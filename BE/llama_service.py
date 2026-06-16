"""
LLaMA Service untuk RAG Peraturan Desa
========================================

Model path priority (untuk fine-tuned model):
  1. notebooks/model_merged_rag_perdes/  (model fine-tuned RAFT terbaru)
  2. model_merged_rag_perdes/            (alternatif path)
  3. model_merged_legal/                 (model fine-tuned lama)

Base model (fallback jika tidak ada fine-tuned model):
  /workspace/model/Meta-Llama-3.1-8B-Instruct/

Endpoint:
  /api/chat     → model loaded (fine-tuned atau base) + pertanyaan TANPA konteks
  /api/chat-rag → model loaded (fine-tuned atau base) + pertanyaan DENGAN konteks dokumen

Kedua endpoint menggunakan MODEL YANG SAMA. Perbedaannya hanya di prompt:
  - chat:     system_prompt + user_question
  - chat-rag: RAG_system_prompt + dokumen_konteks + user_question
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Priority list untuk model path (resolved absolute paths)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_BASE_DIR)  # /workspace

MODEL_PATHS = [
    os.path.join(_BASE_DIR, "notebooks", "model_merged_rag_perdes"),
    os.path.join(_WORKSPACE_DIR, "notebooks", "model_merged_rag_perdes"),
    os.path.join(_BASE_DIR, "model_merged_rag_perdes"),
    os.path.join(_WORKSPACE_DIR, "model_merged_rag_perdes"),
    os.path.join(_WORKSPACE_DIR, "model_merged_legal"),
    os.path.join(_BASE_DIR, "..", "model_merged_legal"),
]

# Path ke base model lokal (tokenizer + fallback model)
BASE_MODEL_NAME = os.path.join(_WORKSPACE_DIR, "model", "Meta-Llama-3.1-8B-Instruct")

# System prompt untuk RAG (sama seperti yang dipakai saat fine-tuning RAFT)
RAG_SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Kabupaten Bandung.\n\n"
    "Di bawah ini terdapat beberapa dokumen peraturan. Satu dokumen adalah "
    "GOLD_DOCUMENT (sumber jawaban yang benar), sisanya adalah DISTRACTOR "
    "(dokumen tidak relevan yang sengaja disertakan). Gunakan HANYA informasi "
    "dari GOLD_DOCUMENT untuk menjawab pertanyaan."
)

_model = None
_tokenizer = None
_model_path_used = None


def _find_model_path():
    """Cari model path yang tersedia secara lokal."""
    for path in MODEL_PATHS:
        resolved = os.path.abspath(path)
        if os.path.isdir(resolved):
            logger.info(f"Found model at: {resolved}")
            return resolved
    logger.warning("No local model found. Searched paths:")
    for path in MODEL_PATHS:
        logger.warning(f"  - {os.path.abspath(path)} (not found)")
    return None


def _get_hf_token():
    """Get HF token dari environment variable atau .env file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Coba baca dari .env file di workspace root
        env_path = os.path.join(_WORKSPACE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return token


def load_model():
    global _model, _tokenizer, _model_path_used

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    model_path = _find_model_path()
    hf_token = _get_hf_token()

    if model_path:
        logger.info(f"Loading fine-tuned model dari: {model_path}")
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,  # Jangan coba download dari HF
        )
        # Try tokenizer from fine-tuned model first, fallback to base model
        tokenizer_path = model_path if os.path.isfile(os.path.join(model_path, "tokenizer_config.json")) else BASE_MODEL_NAME
        logger.info(f"Loading tokenizer dari: {tokenizer_path}")
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        _model_path_used = model_path
    else:
        # Fallback: pakai base model lokal
        if not os.path.isdir(BASE_MODEL_NAME):
            raise RuntimeError(
                f"Tidak ada fine-tuned model dan base model tidak ditemukan di: "
                f"{BASE_MODEL_NAME}. "
                f"Pastikan folder model ada atau jalankan fine-tuning terlebih dahulu."
            )

        logger.info(f"Loading base model dari: {BASE_MODEL_NAME}")
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        _model_path_used = BASE_MODEL_NAME

    logger.info(f"Model loaded dari: {_model_path_used}")
    return _model, _tokenizer


def generate_answer(
    pertanyaan: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
) -> str:
    """Generate jawaban tanpa konteks RAG (mode biasa)."""
    model, tokenizer = load_model()

    if system_prompt is None:
        system_prompt = (
            "Kamu adalah asisten hukum yang membantu menjawab pertanyaan "
            "tentang peraturan desa dengan bahasa yang mudah dipahami."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pertanyaan},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return response


def generate_answer_rag(
    pertanyaan: str,
    dokumen_konteks: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate jawaban dengan konteks RAG (format RAFT).

    Args:
        pertanyaan: Pertanyaan user
        dokumen_konteks: Teks dokumen yang di-retrieve (GOLD + DISTRACTOR)
            Format yang diharapkan sama seperti saat fine-tuning:
            "=== DOKUMEN KONTEKS ===\n[DOKUMEN 1] ...\n=== AKHIR DOKUMEN KONTEKS ==="
        max_new_tokens: Maksimum token yang di-generate

    Returns:
        Jawaban model yang di-grounding ke dokumen konteks
    """
    model, tokenizer = load_model()

    # Format: system prompt + dokumen konteks (sama seperti training RAFT)
    user_content = f"{dokumen_konteks}\n\n{pertanyaan}"

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return response


def get_model_info() -> dict:
    """Return info tentang model yang sedang dipakai."""
    if _model is None:
        return {"loaded": False, "model_path": None}

    return {
        "loaded": True,
        "model_path": _model_path_used,
        "device": str(next(_model.parameters()).device),
        "dtype": str(next(_model.parameters()).dtype),
    }


def get_error_detail(e: Exception) -> str:
    """Return full error detail termasuk traceback untuk debugging."""
    tb = traceback.format_exc()
    return f"{type(e).__name__}: {str(e) or '(no message)'}\n\nTraceback:\n{tb}"
