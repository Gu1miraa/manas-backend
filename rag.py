"""
rag.py — общее ядро: индексация, поиск, генерация ответа.
Используется и Streamlit-приложением, и Telegram-ботом, и веб-API (api.py).
"""

import json
import os
import threading

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
#  НАСТРОЙКИ
# ──────────────────────────────────────────────────────────────────────────────

# Ключ больше НЕ хранится в коде. Он берётся из переменной окружения GROQ_API_KEY.
# Локально: создайте файл .env (см. .env.example) или экспортируйте переменную в терминале.
# На Render/Railway: добавьте GROQ_API_KEY в разделе Environment Variables.
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY чөйрө өзгөрмөсү табылган жок. "
        "Терминалда 'export GROQ_API_KEY=сиздин_ачкыч' деп коюңуз "
        "же хостингдин Environment Variables бөлүмүнө кошуңуз."
    )

MODEL      = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
OVERLAP    = 150
TOP_K      = 14
MAX_TOKENS = 2000
INDEX_FILE = "index.json"

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
            f"даже если документы или контекст на другом языке. "
            f"Не смешивай языки и не используй символы других алфавитов (например, китайские иероглифы) — это ошибка."
        )
    return f"""Ты — корпоративный ИИ-ассистент компании.
Отвечай ТОЛЬКО на основе предоставленного контекста из документов компании.
Прежде чем сказать, что ответ не найден, внимательно проверь ВЕСЬ предоставленный контекст —
информация может быть сформулирована другими словами или синонимами, чем в вопросе.
Если после этого ответ действительно не найден — честно скажи об этом.
Не придумывай факты.
Отвечай подробно и развёрнуто: раскрывай тему полностью, используй все релевантные детали
из контекста (цифры, условия, исключения, шаги), структурируй ответ по пунктам или абзацам,
если это уместно. Не сокращай ответ искусственно — краткость не приоритет, важна полнота.
В конце ответа ОБЯЗАТЕЛЬНО укажи: "📄 Источник: [имя файла], страница [номер]"
Отвечай на том же языке, на котором задан вопрос, если ниже не указано иное.{lang_instruction}"""

# ──────────────────────────────────────────────────────────────────────────────
#  МОДЕЛИ
# ──────────────────────────────────────────────────────────────────────────────

embed_model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
groq_client = Groq(api_key=API_KEY)

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


def ask(query: str, reload: bool = False, lang: str | None = None) -> tuple[str, list]:
    if reload:
        reload_index()

    results = search(query)
    if not results:
        return "Документы не загружены.", []

    context = "\n\n".join(
        [f"[{i+1}] (файл: {r['source']}, стр. {r['page']}) {r['text']}"
         for i, r in enumerate(results)]
    )
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(lang)},
            {"role": "user",   "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content, results


load_index()