from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
import os
import docx  # pip install python-docx
import re

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

messages = []
file_texts = {}  # Словарь: имя файла → текст


# 🔹 Функция ИИ: ищет совпадения по словам в предложениях
def ai_response(question: str):
    # чистим вопрос от спецсимволов, оставляем только буквы и цифры
    question_words = set(re.sub(r'[^a-zA-Zа-яА-Я0-9]+', ' ', question.lower()).split())
    for fname, text in file_texts.items():
        sentences = text.split('.')  # разбиваем текст на предложения
        for sent in sentences:
            sent_words = set(re.sub(r'[^a-zA-Zа-яА-Я0-9]+', ' ', sent.lower()).split())
            if question_words & sent_words:  # хотя бы одно совпадение
                return f"Ответ из файла {fname}: {sent.strip()}."
    return "В файле нет информации по вашему вопросу."


@app.post("/chat")
async def chat(
    username: str = Form(...),
    text: str = Form(...),
    file: UploadFile = File(None)
):
    now = datetime.now().strftime("%H:%M:%S")
    file_path = None

    # 🔹 Сохраняем файл и извлекаем текст
    if file:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # очищаем старый текст файла (если был)
        file_texts[file.filename] = ""

        # читаем .docx
        if file.filename.endswith(".docx"):
            doc = docx.Document(file_path)
            file_texts[file.filename] = "\n".join([p.text for p in doc.paragraphs])
        # читаем .txt
        elif file.filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                file_texts[file.filename] = f.read()

    # 🔹 Сохраняем сообщение пользователя
    messages.append({
        "username": username,
        "text": text,
        "filename": file.filename if file else None,
        "time": now
    })

    # 🔹 Генерируем ответ ИИ
    ai_text = ai_response(text)
    messages.append({
        "username": "ИИ-ассистент",
        "text": ai_text,
        "filename": None,
        "time": now
    })

    return {
        "status": "received",
        "time": now,
        "file_saved": file_path,
        "ai_reply": ai_text
    }


@app.get("/messages")
def get_messages():
    return {"messages": messages}


@app.post("/clear")
def clear_messages():
    messages.clear()
    file_texts.clear()
    return {"status": "chat cleared"}