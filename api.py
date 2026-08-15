import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from rag import ask, total_chunks, get_all_sources   # <-- бул саптын аягына кошуу кереk

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/health", methods=["GET"])
def health():
    """Сервер иштеп жатабы деп текшерүү үчүн."""
    return jsonify({"status": "ok"})


@app.route("/api/debug", methods=["GET"])          # <-- ЖАҢЫ ENDPOINT
def debug():
    return jsonify({
        "total_chunks": total_chunks(),
        "sources": get_all_sources(),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("message") or "").strip()
    lang = data.get("lang", "ky")
    history = data.get("history") or []
    if not isinstance(history, list):
        history = []
    history = history[-10:]

    if not question:
        return jsonify({"error": "Суроо бош болбошу керек"}), 400

    try:
        answer, sources = ask(question, lang=lang, history=history)
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