"""
rag.py — общее ядро: индексация, поиск, генерация ответа.
Используется и Streamlit-приложением, и Telegram-ботом, и веб-API (api.py).

Генерация ответов полностью через Groq (Llama 3.3 70B) — быстрый, щедрая
бесплатная квота (30 запросов/мин, ~1000+ запросов/день).
"""

import json
import os
import re
import threading

import fitz  # PyMuPDF — нужен для автозагрузки PDF из docs/ при старте
from docx import Document as DocxDocument  # python-docx — для .docx файлов
import numpy as np
from dotenv import load_dotenv
load_dotenv("key.env")

from groq import Groq
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
#  НАСТРОЙКИ
# ──────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY чөйрө өзгөрмөсү табылган жок. "
        "https://console.groq.com/keys дарегинен акысыз ачкыч алып, "
        "'export GROQ_API_KEY=сиздин_ачкыч' деп коюңуз "
        "же хостингдин Environment Variables/Secrets бөлүмүнө кошуңуз."
    )

groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL  = "openai/gpt-oss-120b"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE  = 800
OVERLAP     = 150
TOP_K       = 18
MAX_TOKENS  = 2000
INDEX_FILE  = "index.json"
DOCS_DIR    = "docs"  # сюда кладём PDF, которые должны индексироваться сами при старте

LANG_NAMES = {
    "ky": "кыргызском",
    "ru": "русском",
    "tr": "турецком (Türkçe)",
}

def build_system_prompt(lang: str | None = None) -> str:
    lang_instruction = ""
    if lang and lang in LANG_NAMES:
        lang_instruction = (
            f"\nОБЯЗАТЕЛЬНО отвечай ТОЛЬКО на {LANG_NAMES[lang]} языке, "
            f"даже если документы или контекст на другом языке."
        )
    return f"""Ты — корпоративный ИИ-ассистент компании.
Отвечай ТОЛЬКО на основе предоставленного контекста из документов компании.
Внимательно изучи ВЕСЬ предоставленный контекст перед ответом — информация может быть
сформулирована другими словами, сокращениями или синонимами, чем в вопросе. Если в контексте
есть таблица, список или конкретные данные, относящиеся к теме вопроса — используй их, даже
если формулировка в контексте не дословно совпадает с вопросом.

Говори "информация не найдена" ТОЛЬКО если в контексте действительно нет данных, относящихся
к теме вопроса — не будь излишне осторожным, если релевантная информация присутствует.
Не отвечай на основе общих знаний о том, как это "обычно бывает" в университетах — только на
основе того, что реально есть в контексте. Не придумывай факты, цифры или названия, которых
нет в контексте.

СТРОГО ЗАПРЕЩЕНО смешивать языки и алфавиты внутри одного ответа. Используй ТОЛЬКО буквы того
языка, на котором пишешь ответ.

Отвечай подробно и развёрнуто: раскрывай тему полностью, используй все релевантные детали
из контекста (цифры, условия, исключения, шаги), структурируй ответ по пунктам или абзацам,
если это уместно. Не сокращай ответ искусственно — краткость не приоритет, важна полнота.

Если ты даёшь содержательный ответ на основе контекста — ОБЯЗАТЕЛЬНО укажи в конце:
"📄 Источник: [имя файла], страница [номер]". Если честно говоришь, что информация не найдена —
источник не указывай.

НИКОГДА не используй LaTeX-формулы. Это касается ЛЮБЫХ команд вида \\text{{}}, \\frac{{}}{{}}, \\sum,
\\limits, \\displaystyle, \\times, \\cdot, а также квадратных скобок [ ... ] или круглых \\( ... \\)
как обёртки для формул. Формулы пиши ОДНОЙ строкой обычными символами клавиатуры: ×, ÷, /, Σ,
=, обычные скобки ( ). Пример правильного оформления формулы:
Средний балл = Σ(оценка × кредит) / Σ(кредит)
Пример НЕПРАВИЛЬНОГО оформления (так писать ЗАПРЕЩЕНО):
[ \\text{{Средний балл}} = \\frac{{\\sum(\\text{{оценка}}_i \\times \\text{{кредит}}_i)}}{{\\sum \\text{{кредит}}_i}} ]
Отвечай на том же языке, на котором задан вопрос, если ниже не указано иное.{lang_instruction}"""

# ──────────────────────────────────────────────────────────────────────────────
#  МОДЕЛИ
# ──────────────────────────────────────────────────────────────────────────────

embed_model = SentenceTransformer(EMBED_MODEL)

_documents: list[dict] = []
_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
#  ДИСК
# ──────────────────────────────────────────────────────────────────────────────

def save_index() -> None:
    with _lock:
        data = [
            {
                "text":   d["text"],
                "emb":    d["emb"].tolist(),
                "source": d["source"],
                "page":   d.get("page", 0),
            }
            for d in _documents
        ]
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_index() -> None:
    global _documents
    if not os.path.exists(INDEX_FILE):
        return
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    with _lock:
        _documents = [
            {
                "text":   d["text"],
                "emb":    np.array(d["emb"], dtype=np.float32),
                "source": d["source"],
                "page":   d.get("page", 0),
            }
            for d in data
        ]


def reload_index() -> None:
    load_index()

# ──────────────────────────────────────────────────────────────────────────────
#  ИНДЕКСАЦИЯ
# ──────────────────────────────────────────────────────────────────────────────

def _split_text(text: str) -> list[str]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        candidate = current + sentence + ". "
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
                tail = current[-OVERLAP:] if len(current) > OVERLAP else current
                current = tail + sentence + ". "
            else:
                chunks.append(sentence.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── очистка колонтитулов (номер страницы + название регламента) ─────────────
_FOOTER_RE = re.compile(
    r"Sayfa:\s*\d+\s+[A-ZÇĞİÖŞÜÂÎÛ\s]+|^\s*KTMÜ MEVZUATI\s*$|^\s*\d+\s*$",
    re.MULTILINE,
)

def _clean_page_text(text: str) -> str:
    """Убирает повторяющиеся колонтитулы/номера страниц из текста страницы."""
    return _FOOTER_RE.sub("", text)


# ── подстраховка — вычищаем LaTeX из ответа, если модель всё же
#    использовала его, несмотря на инструкцию в системном промпте ────────────
def _strip_latex(text: str) -> str:
    """Заменяет типичные LaTeX-конструкции на обычный читаемый текст."""
    if not text:
        return text

    # \frac{A}{B} -> (A)/(B) — по одному уровню вложенности, несколько проходов
    frac_re = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(4):  # несколько проходов на случай вложенных \frac
        new_text = frac_re.sub(r"(\1)/(\2)", text)
        if new_text == text:
            break
        text = new_text

    # \text{X} -> X
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)

    # \sum с подстрочными/надстрочными индексами -> Σ
    text = re.sub(r"\\sum(\\limits)?(_\{[^{}]*\})?(\^\{[^{}]*\})?", "Σ", text)
    text = re.sub(r"\\sum(_\S+)?(\^\S+)?", "Σ", text)

    # прочие частые команды
    text = text.replace("\\displaystyle", "")
    text = text.replace("\\limits", "")
    text = text.replace("\\times", "×")
    text = text.replace("\\cdot", "×")
    text = text.replace("\\div", "÷")

    # оставшиеся подстрочные/надстрочные индексы вида _{...} ^{...}
    text = re.sub(r"[_^]\{([^{}]*)\}", r"\1", text)

    # LaTeX-скобки-обёртки формул: \[ \] \( \)
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")

    # любые оставшиеся одиночные "\команда" без аргументов — просто убираем
    # обратный слэш, оставляя слово (на случай редких команд вроде \alpha)
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)

    # убираем случайно оставшиеся фигурные скобки
    text = text.replace("{", "").replace("}", "")

    # схлопываем лишние пробелы, которые могли появиться после замен
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def _extract_docx_text(path: str) -> str:
    """Извлекает весь текст из .docx: обычные абзацы + текст из таблиц."""
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts)


def add_document(text: str, source: str = "unknown") -> int:
    """Индексирует обычный текст (TXT). Номер страницы = 0."""
    chunks = _split_text(text)
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    with _lock:
        for chunk, emb in zip(chunks, embeddings):
            _documents.append({"text": chunk, "emb": emb, "source": source, "page": 0})
    save_index()
    return len(chunks)


def add_pdf_pages(pages: list[tuple[int, str]], source: str) -> int:
    """
    Индексирует PDF, склеивая текст ВСЕХ страниц в единый поток (после очистки
    колонтитулов), чтобы статьи/абзацы, переходящие на следующую страницу, не
    резались "вслепую" по границе страницы, а нарезались по смыслу (_split_text)
    с overlap через границы страниц.
    Номер страницы для каждого чанка вычисляется по его положению в общем тексте.

    pages = [(номер_страницы, текст_страницы), ...]
    """
    cleaned_pages = [(p_num, _clean_page_text(p_text)) for p_num, p_text in pages]

    full_text = ""
    page_offsets: list[tuple[int, int]] = []  # [(offset начала страницы, номер страницы), ...]
    for p_num, p_text in cleaned_pages:
        page_offsets.append((len(full_text), p_num))
        full_text += p_text + "\n"

    chunks = _split_text(full_text)

    def _page_for_pos(pos: int) -> int:
        page = page_offsets[0][1]
        for offset, p_num in page_offsets:
            if offset <= pos:
                page = p_num
            else:
                break
        return page

    all_chunks = []
    search_pos = 0
    for chunk in chunks:
        pos = full_text.find(chunk[:40], search_pos)
        if pos == -1:
            pos = search_pos
        page_num = _page_for_pos(pos)
        search_pos = pos + 1
        all_chunks.append({"text": chunk, "source": source, "page_num": page_num})

    embeddings = embed_model.encode(
        [c["text"] for c in all_chunks], normalize_embeddings=True
    )
    total = len(all_chunks)
    with _lock:
        for c, emb in zip(all_chunks, embeddings):
            _documents.append({
                "text": c["text"], "emb": emb,
                "source": c["source"], "page": c["page_num"],
            })
    save_index()
    return total


def clear_documents() -> None:
    global _documents
    with _lock:
        _documents = []
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)


def get_all_sources() -> dict[str, int]:
    result: dict[str, int] = {}
    with _lock:
        for d in _documents:
            result[d["source"]] = result.get(d["source"], 0) + 1
    return result


def total_chunks() -> int:
    with _lock:
        return len(_documents)


# ── автозагрузка PDF из docs/ при старте, если индекс пуст ───────────────────
def ensure_indexed() -> None:
    """
    Вызывается один раз при импорте модуля (см. самый низ файла).
    Если индекс уже не пуст — ничего не делает.
    Если пуст — сам читает все PDF из папки docs/ и индексирует их через
    add_pdf_pages(), без необходимости вручную загружать файл через Streamlit.
    """
    if total_chunks() > 0:
        print(f"Индекс уже загружен: {total_chunks()} фрагментов.")
        return

    if not os.path.isdir(DOCS_DIR):
        print(f"Папка {DOCS_DIR} не найдена — нечего индексировать.")
        return

    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        lower = fname.lower()
        if lower.endswith(".pdf"):
            print(f"Автозагрузка (PDF): {fname} ...")
            with fitz.open(path) as doc:
                pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
            n = add_pdf_pages(pages, fname)
            print(f"  → {n} фрагментов добавлено.")
        elif lower.endswith(".docx"):
            print(f"Автозагрузка (Word): {fname} ...")
            try:
                text = _extract_docx_text(path)
                n = add_document(text, fname)
                print(f"  → {n} фрагментов добавлено.")
            except Exception as e:
                print(f"  Ошибка чтения {fname}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  ПОИСК И ГЕНЕРАЦИЯ
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_search_query(query: str, history: list[dict] | None, lang: str | None) -> str:
    """
    Издөө үчүн суроону даярдайт: (1) маектин акыркы бөлүгүн эске алып, суроону
    өз алдынча түшүнүктүү кылып кайра жазат, (2) документтер негизинен түркчө
    болгондуктан, түркчөгө которот. Жоптун өзү дайыма колдонуучу тандаган тилде
    кайтарылат — бул которуу жөн эле издөө үчүн гана.
    """
    if not query or not query.strip():
        return query

    history_text = ""
    if history:
        recent = history[-6:]
        lines = []
        for m in recent:
            role = "Kullanıcı" if m.get("role") == "user" else "Asistan"
            text = (m.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        history_text = "\n".join(lines)

    if not history_text and lang == "tr":
        return query

    try:
        prompt = (
            "Aşağıda bir sohbet geçmişi (varsa) ve kullanıcının son sorusu var. "
            "Son soruyu, sohbet geçmişindeki bağlamı da göz önünde bulundurarak, "
            "tek başına anlaşılır, bağımsız bir arama sorgusuna dönüştür ve "
            "TÜRKÇE olarak yaz. SADECE yeniden yazılmış soruyu döndür, başka hiçbir şey yazma.\n\n"
        )
        if history_text:
            prompt += f"Sohbet geçmişi:\n{history_text}\n\n"
        prompt += f"Son soru: {query}"

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        rewritten = (completion.choices[0].message.content or "").strip()

        print(f"[DEBUG] groq rewrite raw='{rewritten}'")

        if not rewritten or len(rewritten) < len(query) * 0.4:
            print(f"[DEBUG] rewritten query өтө кыска/бош, түпнускага кайтабыз")
            return query

        return rewritten
    except Exception as e:
        print(f"[DEBUG] groq translate ERROR: {e}")
        return query


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Возвращает список словарей: score, text, source, page."""
    with _lock:
        docs = list(_documents)
    if not docs:
        return []
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    scored = sorted(
        [
            {
                "score":  float(np.dot(q_emb, d["emb"])),
                "text":   d["text"],
                "source": d["source"],
                "page":   d.get("page", 0),
            }
            for d in docs
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return scored[:top_k]


def ask(query: str, reload: bool = False, lang: str | None = None, history: list[dict] | None = None) -> tuple[str, list]:
    if reload:
        reload_index()

    search_query = _prepare_search_query(query, history, lang)
    print(f"[DEBUG] original='{query}' lang={lang} -> search_query='{search_query}'")

    results = search(search_query)
    print(f"[DEBUG] found {len(results)} results")
    if results:
        for r in results[:5]:
            print(f"[DEBUG]   score={r['score']:.3f} source={r['source']} page={r['page']} text={r['text'][:80]!r}")

    if not results:
        return "Документы не загружены.", []

    context = "\n\n".join(
        [f"[{i+1}] (файл: {r['source']}, стр. {r['page']}) {r['text']}"
         for i, r in enumerate(results)]
    )

    history_note = ""
    if history:
        recent = history[-6:]
        lines = []
        for m in recent:
            role = "Колдонуучу" if m.get("role") == "user" else "Жардамчы"
            text = (m.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        if lines:
            history_note = "Мурунку маек (тактоо үчүн гана, жоопту документтен ал):\n" + "\n".join(lines) + "\n\n"

    user_prompt = f"{history_note}Контекст:\n{context}\n\nВопрос: {query}"

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(lang)},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0,
        )
        answer = completion.choices[0].message.content
        answer = _strip_latex(answer)
        return answer, results
    except Exception as e:
        print(f"[DEBUG] Groq generate ERROR: {e}")
        raise


load_index()
ensure_indexed()