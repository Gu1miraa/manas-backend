"""
rag.py — общее ядро: индексация, поиск, генерация ответа.
Используется и Streamlit-приложением, и Telegram-ботом, и веб-API (api.py).

Генерация ответов полностью через Groq (Llama 3.3 70B) — быстрый, щедрая
бесплатная квота (30 запросов/мин, ~1000+ запросов/день).
"""

import json
import os
import threading

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

GROQ_MODEL  = "llama-3.3-70b-versatile"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE  = 800
OVERLAP     = 150
TOP_K       = 18
MAX_TOKENS  = 2000
INDEX_FILE  = "index.json"

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
    Индексирует PDF постранично.
    pages = [(номер_страницы, текст_страницы), ...]
    """
    total = 0
    all_chunks = []
    for page_num, page_text in pages:
        chunks = _split_text(page_text)
        embeddings = embed_model.encode(chunks, normalize_embeddings=True)
        for chunk, emb in zip(chunks, embeddings):
            all_chunks.append({"text": chunk, "emb": emb, "source": source, "page": page_num})
        total += len(chunks)
    with _lock:
        _documents.extend(all_chunks)
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
        return answer, results
    except Exception as e:
        print(f"[DEBUG] Groq generate ERROR: {e}")
        raise


load_index()