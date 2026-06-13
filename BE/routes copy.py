from flask import Blueprint, request, jsonify
from llama_service import generate_answer, load_model

bp = Blueprint("routes", __name__)


@bp.route("/api/load-model", methods=["POST"])
def load_model_endpoint():
    """
    Endpoint untuk load model secara manual.
    Berguna untuk preload model sebelum ada request chat.
    """
    try:
        load_model()
        return jsonify({
            "status": "success",
            "message": "Model berhasil dimuat ke memori."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal memuat model: {str(e)}"
        }), 500


@bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Body JSON:
    {
        "message": "Apa itu kewenangan desa?",
        "system_prompt": "(opsional)",
        "max_new_tokens": 512  (opsional, default 512)
    }
    """
    data = request.get_json(force=True)

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Field 'message' wajib diisi."}), 400

    system_prompt = data.get("system_prompt", None)
    max_new_tokens = int(data.get("max_new_tokens", 512))

    try:
        answer = generate_answer(message, system_prompt, max_new_tokens)
        return jsonify({
            "message": message,
            "answer": answer
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200



