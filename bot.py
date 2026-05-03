"""
bot.py - Telegram-бот, использующий общее ядро rag.py.
"""

import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag import ask, add_document, total_chunks

logging.basicConfig(level=logging.INFO)

TOKEN = "8678545457:AAFOsuItI1ZcINYnP1NkbrUl0HMF4B1gBbQ"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я корпоративный ИИ-ассистент.\n"
        "Задайте любой вопрос по загруженным документам.\n\n"
        "Команды:\n"
        "/status - сколько документов загружено\n"
        "/add текст - добавить текст прямо из Telegram"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from rag import reload_index, get_all_sources
    reload_index()
    sources = get_all_sources()
    n = total_chunks()
    if not sources:
        await update.message.reply_text("Документов нет. Загрузите файлы через веб-интерфейс.")
        return
    lines = ["Всего фрагментов: " + str(n) + "\n"]
    for src, cnt in sources.items():
        lines.append("  " + src + ": " + str(cnt) + " фрагм.")
    await update.message.reply_text("\n".join(lines))


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Использование: /add текст для добавления")
        return
    n = add_document(text, source="telegram")
    await update.message.reply_text("Добавлено " + str(n) + " фрагментов.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    await update.message.reply_text("Ищу ответ...")

    answer, results = ask(query, reload=True)

    sources_lines = []
    for r in results[:3]:
        page = r.get("page", 0)
        if page > 0:
            sources_lines.append("  " + r["source"] + ", стр. " + str(page))

    if sources_lines:
        answer = answer + "\n\nИсточники:\n" + "\n".join(sources_lines)

    await update.message.reply_text(answer)


def run_bot() -> None:
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    run_bot()