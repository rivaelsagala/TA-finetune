from flask import Blueprint, request, jsonify
from raft_service import load_model, generate_response

bp = Blueprint("routes", __name__)


@bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "RAFT API is running"
    }), 200


@bp.route("/api/load-model", methods=["POST"])
def load_model_endpoint():
    try:
        load_model()
        return jsonify({
            "status": "success",
            "message": "Model berhasil dimuat ke memori."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@bp.route("/api/chat-raft", methods=["POST"])
def chat():
    """
    Endpoint utama untuk tanya jawab dengan context.

    Body JSON:
    {
        "question": "BPD itu apa sih, fungsinya buat apa di desa?",
        "context": "..." | ["...", "..."]   ← opsional
    }

    Response JSON:
    {
        "status": "success",
        "message": "Jawaban berhasil diproses",
        "thought": "...",
        "answer": "..."
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        question = str(data.get("question") or "").strip()

        if not question:
            return jsonify({
                "status": "error",
                "message": "Field 'question' wajib diisi."
            }), 400

        # context/documents bisa berupa string atau list (opsional)
        context = data.get("context")
        documents = data.get("documents")

        result = generate_response(question=question, context=context, documents=documents)

        return jsonify({
            "status": "success",
            "message": "Jawaban berhasil diproses",
            "thought": result["thought"],
            "answer": result["answer"],
        }), 200

    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500