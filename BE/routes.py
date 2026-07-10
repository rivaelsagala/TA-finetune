from flask import Blueprint, request, jsonify
from llama_service import load_model, generate_answer

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
def chat_rag():
    """
    Body JSON:
    {
      "question": "Sebutkan unsur masyarakat dalam Pasal 12 ayat (2) Perdes Majasetra?",
      "documents": [
        "dokumen 1 ...",
        "dokumen 2 ...",
        "dokumen 3 ..."
      ],
      "max_new_tokens": 512
    }
    """
    try:
        data = request.get_json(force=True)

        question = (data.get("question") or "").strip()
        documents = data.get("documents", [])
        max_new_tokens = int(data.get("max_new_tokens", 512))

        if not question:
            return jsonify({
                "status": "error",
                "message": "Field 'question' wajib diisi."
            }), 400

        if not isinstance(documents, list) or len(documents) == 0:
            return jsonify({
                "status": "error",
                "message": "Field 'documents' wajib berupa list dan tidak boleh kosong."
            }), 400

        answer = generate_answer(
            question=question,
            documents=documents,
            max_new_tokens=max_new_tokens
        )

        return jsonify({
            "status": "success",
            "question": question,
            "documents_count": len(documents),
            "answer": answer
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500