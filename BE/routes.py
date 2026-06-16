from flask import Blueprint, request, jsonify, Response, stream_with_context
from llama_service import (
    generate_answer,
    generate_answer_rag,
    load_model,
    get_model_info,
    get_error_detail,
    MODEL_PATHS,
    BASE_MODEL_NAME,
)
import logging
import os
import json
import time

logger = logging.getLogger(__name__)

bp = Blueprint("routes", __name__)


@bp.route("/api/load-model", methods=["POST"])
def load_model_endpoint():
    """
    Endpoint untuk load model secara manual.
    Berguna untuk preload model sebelum ada request chat.
    """
    try:
        load_model()
        info = get_model_info()
        return jsonify({
            "status": "success",
            "message": "Model berhasil dimuat ke memori.",
            "model_info": info,
        }), 200
    except Exception as e:
        detail = get_error_detail(e)
        logger.error(f"Error loading model:\n{detail}")
        return jsonify({
            "status": "error",
            "message": f"Gagal memuat model: {type(e).__name__}: {str(e) or '(no message)'}",
        }), 500


@bp.route("/api/model-info", methods=["GET"])
def model_info_endpoint():
    """Cek info model yang sedang dipakai."""
    return jsonify(get_model_info()), 200


@bp.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat biasa (tanpa RAG context).

    Body JSON:
    {
        "message": "Apa itu kewenangan desa?",
        "system_prompt": "Kamu adalah asisten..." (opsional),
        "max_new_tokens": 512  (opsional, default 512)
    }
    """
    data = request.get_json(force=True)

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Field 'message' wajib diisi."}), 400

    system_prompt = data.get("system_prompt", None)
    # Jangan pakai string "(opsional)" sebagai system prompt
    if system_prompt and system_prompt.strip().lower() in ("(opsional)", "opsional", ""):
        system_prompt = None
    max_new_tokens = int(data.get("max_new_tokens", 512))

    try:
        answer = generate_answer(message, system_prompt, max_new_tokens)
        return jsonify({
            "message": message,
            "answer": answer,
        }), 200
    except Exception as e:
        detail = get_error_detail(e)
        logger.error(f"Error in /api/chat:\n{detail}")
        return jsonify({
            "error": f"{type(e).__name__}: {str(e) or '(no message)'}",
        }), 500


@bp.route("/api/chat-rag", methods=["POST"])
def chat_rag():
    """
    Chat dengan konteks RAG (model fine-tuned RAFT).

    Body JSON:
    {
        "message": "Apa isi pasal 1?",
        "konteks": "=== DOKUMEN KONTEKS ===\n[DOKUMEN 1] [GOLD_DOCUMENT]\n...\n=== AKHIR DOKUMEN KONTEKS ===",
        "max_new_tokens": 512  (opsional, default 512)
    }

    Format 'konteks' harus sama persis dengan yang dipakai saat fine-tuning:
    - Berisi [GOLD_DOCUMENT] dan [DISTRACTOR]
    - Dipisahkan dengan ---
    """
    data = request.get_json(force=True)

    message = data.get("message", "").strip()
    konteks = data.get("konteks", "").strip()

    if not message:
        return jsonify({"error": "Field 'message' wajib diisi."}), 400
    if not konteks:
        return jsonify({"error": "Field 'konteks' wajib diisi untuk RAG."}), 400

    max_new_tokens = int(data.get("max_new_tokens", 512))

    try:
        answer = generate_answer_rag(message, konteks, max_new_tokens)
        return jsonify({
            "message": message,
            "answer": answer,
            "mode": "rag",
        }), 200
    except Exception as e:
        detail = get_error_detail(e)
        logger.error(f"Error in /api/chat-rag:\n{detail}")
        return jsonify({
            "error": f"{type(e).__name__}: {str(e) or '(no message)'}",
        }), 500


@bp.route("/api/health", methods=["GET"])
def health():
    info = get_model_info()
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "model_loaded": info["loaded"],
        "model_info": info,
        "endpoints": {
            "chat": "POST /api/chat - Generate without RAG context",
            "chat_rag": "POST /api/chat-rag - Generate with RAG context",
            "load_model": "POST /api/load-model - Preload model into memory",
            "model_info": "GET /api/model-info - Get current model info",
            "models": "GET /api/models - List available models on server",
            "health": "GET /api/health - Health check",
        },
    }), 200


@bp.route("/api/models", methods=["GET"])
def list_models():
    """
    List all models available on the server (fine-tuned + base).
    Useful to check which models exist before calling /api/load-model.
    """
    available = []
    for path in MODEL_PATHS:
        resolved = os.path.abspath(path)
        if os.path.isdir(resolved):
            available.append({
                "path": resolved,
                "type": "fine-tuned",
                "exists": True,
            })

    base_exists = os.path.isdir(BASE_MODEL_NAME)
    available.append({
        "path": BASE_MODEL_NAME,
        "type": "base",
        "exists": base_exists,
    })

    current = get_model_info()

    return jsonify({
        "models": available,
        "currently_loaded": current.get("model_path"),
    }), 200
