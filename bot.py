"""
bot.py - Telegram-бот, использующий общее ядро rag.py.
"""

import logging
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from rag import ask, add_document, total_chunks

logging.basicConfig(level=logging.INFO)

TOKEN = "8678545457:AAFOsuItI1ZcINYnP1NkbrUl0HMF4B1gBbQ"

# ──────────────────────────────────────────────────────────────────────────────
#  РОЛИ
# ──────────────────────────────────────────────────────────────────────────────

ROLES = {
    "role_teacher": "Преподаватель",
    "role_student": "Студент",
    "role_applicant": "Абитуриент",
}

GREETING_PATTERN = re.compile(
    r"^\s*("
    r"привет(?:ик|ствую)?|"
    r"здравствуй(?:те)?|"
    r"добр(?:ый|ое)\s+(?:день|утро|вечер)|"
    r"хай|"
    r"хеллоу|"
    r"hello|hi|hey"
    r")[\s!.,]*$",
    re.IGNORECASE,
)


def role_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(label, callback_data=key)]
        for key, label in ROLES.items()
    ]
    return InlineKeyboardMarkup(buttons)


async def send_greeting_and_ask_role(update: Update) -> None:
    await update.message.reply_text(
        "Здравствуйте! 👋 Я корпоративный ИИ-ассистент.\n\n"
        "Подскажите, пожалуйста, кто вы?",
        reply_markup=role_keyboard(),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  КОМАНДЫ
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_greeting_and_ask_role(update)


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


# ──────────────────────────────────────────────────────────────────────────────
#  ВЫБОР РОЛИ (нажатие на кнопку)
# ──────────────────────────────────────────────────────────────────────────────

async def handle_role_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    role_label = ROLES.get(query.data)
    if not role_label:
        return

    context.user_data["role"] = role_label

    await query.edit_message_text(
        f"Спасибо! Вы выбрали роль: «{role_label}».\n\n"
        "Теперь можете задать любой вопрос по загруженным документам."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ОБЫЧНЫЕ СООБЩЕНИЯ
# ──────────────────────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()

    # Если это приветствие — отвечаем приветствием и спрашиваем роль
    if GREETING_PATTERN.match(text):
        await send_greeting_and_ask_role(update)
        return

    query = text
    await update.message.reply_text("Ищу ответ...")

    # Храним историю переписки для каждого пользователя отдельно (Telegram
    # сам разделяет context.user_data по каждому собеседнику).
    history = context.user_data.setdefault("history", [])

    answer, results = ask(query, reload=True, history=list(history))

    # Запоминаем этот обмен репликами, чтобы уточняющие вопросы вида
    # "какие ТАМ формулы" понимались в контексте предыдущего вопроса.
    history.append({"role": "user", "text": query})
    history.append({"role": "assistant", "text": answer})
    # Не даём истории расти бесконечно — оставляем последние 12 сообщений.
    del history[:-12]

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
    app.add_handler(CallbackQueryHandler(handle_role_choice, pattern="^role_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    run_bot()