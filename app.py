"""
app.py — Streamlit-приложение + запуск Telegram-бота в фоновом потоке.

Установка:  pip install streamlit sentence-transformers groq numpy python-telegram-bot
Запуск:     streamlit run app.py
"""

import threading

import streamlit as st

# Импортируем ядро
from rag import add_document, add_pdf_pages, ask, clear_documents, get_all_sources, total_chunks
from bot import run_bot

# ──────────────────────────────────────────────────────────────────────────────
#  ЗАПУСК TELEGRAM-БОТА В ФОНОВОМ ПОТОКЕ
# ──────────────────────────────────────────────────────────────────────────────

if "bot_thread_started" not in st.session_state:
    st.session_state.bot_thread_started = True
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

# ──────────────────────────────────────────────────────────────────────────────
#  СТРАНИЦА
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Система поиска знаний",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    color: #e8eaf0;
}

.main-header {
    background: linear-gradient(90deg, #1e3a5f, #0d47a1);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    border: 1px solid #1565c0;
    box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
}
.main-header h1 { font-family:'Montserrat',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin:0 0 6px; }
.main-header p  { color:#90caf9; margin:0; font-size:0.95rem; font-weight:300; }

.msg-user {
    background: linear-gradient(135deg,#1565c0,#0d47a1);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px; margin: 8px 0 8px 60px;
    color:#fff; font-size:0.95rem; line-height:1.6;
    box-shadow: 0 4px 12px rgba(13,71,161,0.3);
}
.msg-bot {
    background: #1e2535; border:1px solid #2a3550;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px; margin: 8px 60px 8px 0;
    color:#e8eaf0; font-size:0.95rem; line-height:1.7;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.msg-label { font-size:0.72rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; opacity:0.6; }

.doc-card { background:#1e2535; border:1px solid #2a3550; border-radius:12px; padding:16px; margin-bottom:10px; }
.doc-card h4 { color:#64b5f6; font-size:0.8rem; font-weight:600; letter-spacing:0.5px; margin:0 0 8px; }
.doc-card p  { color:#9aa3b8; font-size:0.82rem; line-height:1.5; margin:0; }

section[data-testid="stSidebar"] { background:#12161f !important; border-right:1px solid #1e2535; }

.stButton > button {
    background: linear-gradient(135deg,#1565c0,#0d47a1);
    color:white; border:none; border-radius:10px; font-weight:600; font-family:'Inter',sans-serif; transition:all 0.2s;
}
.stButton > button:hover { background:linear-gradient(135deg,#1976d2,#1565c0); transform:translateY(-1px); box-shadow:0 4px 12px rgba(13,71,161,0.4); }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background:#1e2535 !important; border:1px solid #2a3550 !important;
    border-radius:10px !important; color:#e8eaf0 !important; font-family:'Inter',sans-serif !important;
}
.stFileUploader { background:#1e2535; border:2px dashed #2a3550; border-radius:12px; padding:8px; }

.status-bar { background:#1e2535; border:1px solid #2a3550; border-radius:10px; padding:10px 16px; font-size:0.82rem; color:#64b5f6; margin-bottom:16px; }

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

if "chat"          not in st.session_state: st.session_state.chat          = []
if "loaded_files"  not in st.session_state: st.session_state.loaded_files  = []
if "input_counter" not in st.session_state: st.session_state.input_counter = 0

# ──────────────────────────────────────────────────────────────────────────────
#  САЙДБАР
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Загрузка документов")
    st.markdown("<p style='color:#9aa3b8;font-size:0.82rem;'>Поддерживаются TXT и PDF файлы</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Выберите файлы",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.loaded_files]
        if new_files:
            with st.spinner("Обрабатываю документы... (большой PDF может занять несколько минут)"):
                for file in new_files:
                    raw = file.read()
                    if file.name.lower().endswith(".pdf"):
                        try:
                            import pymupdf
                            doc = pymupdf.open(stream=raw, filetype="pdf")
                            pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
                            add_pdf_pages(pages, file.name)
                        except Exception as e:
                            st.error(f"Ошибка чтения PDF {file.name}: {e}")
                            continue
                    else:
                        text = raw.decode("utf-8-sig", errors="ignore")
                        add_document(text, file.name)
                    st.session_state.loaded_files.append(file.name)
            st.success(f"Загружено {len(new_files)} файл(ов)!")

    sources = get_all_sources()
    if sources:
        st.markdown("---")
        st.markdown("### 📊 Загруженные файлы")
        for fname, cnt in sources.items():
            st.markdown(f"""
            <div class="doc-card">
                <h4>📄 {fname}</h4>
                <p>{cnt} фрагментов</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"<p style='color:#64b5f6;font-size:0.85rem;'>Всего фрагментов: <b>{total_chunks()}</b></p>",
            unsafe_allow_html=True,
        )

        if st.button("🗑 Очистить всё", use_container_width=True):
            clear_documents()
            st.session_state.chat         = []
            st.session_state.loaded_files = []
            st.session_state.input_counter += 1
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:#64b5f6;font-size:0.82rem;'>🤖 Telegram-бот активен — документы доступны и в чате</p>",
        unsafe_allow_html=True,
    )
    if st.button("🗨 Очистить чат", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
#  ОСНОВНАЯ ОБЛАСТЬ
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 Система поиска знаний</h1>
    <p>Интеллектуальный ассистент — задавайте вопросы через веб-интерфейс или Telegram</p>
</div>
""", unsafe_allow_html=True)

# Статус-бар
n = total_chunks()
if n:
    st.markdown(f"""
    <div class="status-bar">
        ✅ Загружено документов: {len(get_all_sources())} &nbsp;|&nbsp; Фрагментов в базе: {n}
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar" style="color:#f39c12;">
        ⚠️ Документы не загружены — добавьте файлы в боковой панели слева
    </div>""", unsafe_allow_html=True)

# История чата
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label">Вы</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-bot">
            <div class="msg-label">🤖 ИИ-ассистент</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander("📎 Источники из документов"):
                for i, src in enumerate(msg["sources"], 1):
                    score = src["score"]
                    text = src["text"]
                    source = src["source"]
                    page = src.get("page", 0)
                    st.markdown(f"""
                    <div class="doc-card">
                        <h4>[{i}] {source}, стр. {page} — релевантность: {score:.2f}</h4>
                        <p>{text[:300]}...</p>
                    </div>""", unsafe_allow_html=True)

# Поле ввода
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        "Вопрос",
        placeholder="Введите вопрос по документам компании...",
        label_visibility="collapsed",
        key=f"query_input_{st.session_state.input_counter}",
    )
with col2:
    send = st.button("Отправить ➤", use_container_width=True)

# Обработка отправки
if send and query.strip():
    user_input = query.strip()
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.spinner("Ищу ответ..."):
        try:
            answer, sources = ask(user_input)
            st.session_state.chat.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as e:
            st.error(f"Ошибка: {e}")

    st.session_state.input_counter += 1
    st.rerun()