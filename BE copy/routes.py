from flask import Blueprint, request, jsonify
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

logger = logging.getLogger(__name__)

bp = Blueprint("routes", __name__)


@bp.route("/api/load-model", methods=["POST"])
def load_model_endpoint():
    """
    Endpoint untuk load model secara manual.
    
    Body JSON (opsional):
    {
        "model_path": "/path/to/model"  // load model tertentu untuk perbandingan
    }
    
    Jika body kosong, akan auto-resolve model pertama yang ditemukan.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_path = data.get("model_path", None)
        load_model(model_path)
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
    Generate jawaban TANPA konteks dokumen (plain chat).

    Body JSON:
    {
        "pertanyaan": "Apa yang dimaksud dengan Desa?",
        "max_new_tokens": 512,       // opsional, default 512
        "temperature": 0.1,          // opsional, default 0.1
        "top_p": 0.9,               // opsional, default 0.9
        "repetition_penalty": 1.1    // opsional, default 1.1
    }

    Contoh Postman:
    POST http://localhost:6000/api/chat
    Content-Type: application/json
    {
        "pertanyaan": "Apa yang dimaksud dengan Desa menurut Perdes Loa No. 5/2017?"
    }
    """
    try:
        data = request.get_json(force=True)
        pertanyaan = data.get("pertanyaan", "").strip()

        if not pertanyaan:
            return jsonify({
                "status": "error",
                "message": "Field 'pertanyaan' wajib diisi.",
            }), 400

        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.1)
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.1)

        result = generate_answer(
            pertanyaan=pertanyaan,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return jsonify({
            "status": "success",
            "pertanyaan": pertanyaan,
            "jawaban": result["answer"],
            "model_type": result["model_type"],
            "model_path": result["model_path"],
        }), 200

    except Exception as e:
        detail = get_error_detail(e)
        logger.error(f"Error di /api/chat:\n{detail}")
        return jsonify({
            "status": "error",
            "message": f"Gagal generate jawaban: {type(e).__name__}: {str(e) or '(no message)'}",
        }), 500


@bp.route("/api/chat-rag", methods=["POST"])
def chat_rag():
    """
    Generate jawaban DENGAN konteks dokumen (RAG / RAFT format).
    Model akan menganalisis dokumen yang diberikan dan menjawab berdasarkan
    dokumen yang paling relevan.

    Body JSON:
    {
        "pertanyaan": "Apa rentang usia bayi?",
        "dokumen": [
            "pasal 1\n\n15. bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari",
            "pasal 1\n\n9. pembangunan desa adalah upaya peningkatan kualitas hidup...",
            "pasal 1\n\n25. pemerintah pusat selanjutnya disebut pemerintah..."
        ],
        "max_new_tokens": 512,       // opsional, default 512
        "temperature": 0.1,          // opsional, default 0.1
        "top_p": 0.9,               // opsional, default 0.9
        "repetition_penalty": 1.1    // opsional, default 1.1
    }

    Contoh Postman:
    POST http://localhost:6000/api/chat-rag
    Content-Type: application/json
    {
        "pertanyaan": "Apa rentang usia bayi menurut Perdes Biru No. 07/2015?",
        "dokumen": [
            "pasal 1\n\n15. bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari",
            "pasal 1\n\n9. pembangunan desa adalah upaya peningkatan kualitas hidup...",
            "pasal 1\n\n25. pemerintah pusat selanjutnya disebut pemerintah..."
        ]
    }
    """
    try:
        data = request.get_json(force=True)
        pertanyaan = data.get("pertanyaan", "").strip()
        dokumen = data.get("dokumen", [])

        if not pertanyaan:
            return jsonify({
                "status": "error",
                "message": "Field 'pertanyaan' wajib diisi.",
            }), 400

        if not dokumen or not isinstance(dokumen, list):
            return jsonify({
                "status": "error",
                "message": "Field 'dokumen' wajib berupa list string berisi dokumen konteks.",
            }), 400

        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.15)

        result = generate_answer_rag(
            pertanyaan=pertanyaan,
            dokumen=dokumen,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return jsonify({
            "status": "success",
            "pertanyaan": pertanyaan,
            "analisis": result["analisis"],
            "jawaban": result["jawaban"],
            "raw_response": result["raw_response"],
            "model_type": result["model_type"],
            "model_path": result["model_path"],
            "num_documents": result["num_documents"],
        }), 200

    except Exception as e:
        detail = get_error_detail(e)
        logger.error(f"Error di /api/chat-rag:\n{detail}")
        return jsonify({
            "status": "error",
            "message": f"Gagal generate jawaban RAG: {type(e).__name__}: {str(e) or '(no message)'}",
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
    # Model type labels for comparison
    model_type_labels = {
        "model_merged_raft_perdes": "fine-tuned-raft",
        "model_merged_perdes": "fine-tuned-qa",
    }

    for path in MODEL_PATHS:
        resolved = os.path.abspath(path)
        basename = os.path.basename(resolved)
        model_type = model_type_labels.get(basename, "fine-tuned")
        if os.path.isdir(resolved):
            available.append({
                "path": resolved,
                "type": model_type,
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
