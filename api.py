"""
api.py — жеңил веб-API. Manas AI React интерфейси ушул серверге суроо жиберет.

Локалдуу иштетүү:
    pip install flask flask-cors
    export GROQ_API_KEY=сиздин_ачкыч          (Windows: set GROQ_API_KEY=...)
    python api.py

Production'до (Render/Railway):
    gunicorn api:app   (Procfile ичинде)
"""

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from rag import ask

app = Flask(__name__)

# Кайсы сайттардан бул API'ге кайрылууга уруксат берилет.
# Оболу '*' менен ачык коюп сынап көрүңүз, андан кийин так өз Vercel
# доменинизге алмаштырыңыз (мисалы: "https://frontend-project-psi-green.vercel.app").
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/health", methods=["GET"])
def health():
    """Сервер иштеп жатабы деп текшерүү үчүн."""
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("message") or "").strip()
    # lang азырынча логго гана колдонулат — rag.py суроонун өз тилинде жооп берүүгө аракет кылат
    lang = data.get("lang", "ky")

    if not question:
        return jsonify({"error": "Суроо бош болбошу керек"}), 400

    try:
        answer, sources = ask(question, lang=lang)
    except Exception as e:
        return jsonify({"error": f"Ички ката: {e}"}), 500

    return jsonify(
        {
            "reply": answer,
            "sources": [
                {"source": s["source"], "page": s["page"], "score": round(s["score"], 3)}
                for s in sources
            ],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)