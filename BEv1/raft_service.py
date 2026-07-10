"""
llama_service.py
================
Service layer untuk model LLaMA 3.1 8B Instruct.

Menangani:
  - Load model (base / fine-tuned RAFT / fine-tuned Q&A)
  - Generate jawaban tanpa konteks (plain chat)
  - Generate jawaban dengan konteks RAG (chat-rag, format RAFT)
  - Info model yang sedang aktif

PENTING: System prompt saat inference HARUS SAMA PERSIS dengan yang dipakai
saat training (fine-tuning). Mismatch akan menyebabkan model bingung dan
menghasilkan jawaban yang tidak relevan.
"""

import os
import re
import logging
import traceback
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Konfigurasi Path Model
# ============================================================================

# Path model fine-tuned (relatif terhadap file ini)
_NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))

MODEL_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "model_merged_raft_v2")),
]

# ============================================================================
# System Prompts (HARUS SAMA PERSIS dengan training)
# ============================================================================

# System prompt untuk model RAFT (fine-tuned dengan dokumen konteks)
# Digunakan di /api/chat-rag
RAFT_SYSTEM_PROMPT = (
    "Anda adalah asisten AI ahli dalam menjawab pertanyaan berdasarkan dokumen hukum dan peraturan desa.\n"
    "Diberikan sejumlah dokumen referensi, analisislah dokumen tersebut untuk mencari jawaban yang tepat.\n"
    "Tuliskan proses berpikir Anda di dalam tag <thought>...</thought> dengan menjelaskan dokumen mana yang relevan dan tidak relevan (distraktor).\n"
    "Setelah itu, berikan jawaban akhir Anda berdasarkan hasil analisis tersebut."
)

# System prompt untuk model base / Q&A (tanpa dokumen konteks)
# Digunakan di /api/chat
PLAIN_SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Indonesia. Jawab dengan jelas dan lengkap "
    "berdasarkan pengetahuan Anda."
)

# ============================================================================
# Global State
# ============================================================================
_model = None
_tokenizer = None
_model_path = None
_model_type = None  # "base", "raft", "qa"


def _detect_model_type(path: str) -> str:
    """Deteksi tipe model berdasarkan nama path."""
    basename = os.path.basename(os.path.abspath(path))
    if "raft" in basename.lower():
        return "raft"
    elif "perdes" in basename.lower() or "merged" in basename.lower():
        return "qa"
    elif "Llama" in basename or "base" in basename.lower():
        return "base"
    return "unknown"


def _resolve_model_path(requested_path: Optional[str] = None) -> str:
    """
    Resolve path model yang akan di-load.
    Priority:
      1. requested_path (jika diberikan dan valid)
      2. Model fine-tuned pertama yang ditemukan (MODEL_PATHS)
    """
    if requested_path:
        abs_path = os.path.abspath(requested_path)
        if os.path.isdir(abs_path):
            return abs_path
        logger.warning(f"Path tidak ditemukan: {abs_path}, mencoba fallback...")

    # Coba model fine-tuned dulu
    for path in MODEL_PATHS:
        if os.path.isdir(path):
            logger.info(f"Model fine-tuned ditemukan: {path}")
            return path

    raise FileNotFoundError(
        f"Tidak ada model yang ditemukan! "
        f"Fine-tuned paths: {MODEL_PATHS}"
    )


def load_model(requested_path: Optional[str] = None):
    """
    Load model dan tokenizer ke GPU.
    Menggunakan Unsloth FastLanguageModel untuk inference yang efisien.
    """
    global _model, _tokenizer, _model_path, _model_type

    from unsloth import FastLanguageModel

    path = _resolve_model_path(requested_path)
    model_type = _detect_model_type(path)

    logger.info(f"Loading model dari: {path} (tipe: {model_type})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # Set ke inference mode
    FastLanguageModel.for_inference(model)

    _model = model
    _tokenizer = tokenizer
    _model_path = path
    _model_type = model_type

    logger.info(f"Model berhasil di-load: {path}")


def _ensure_model_loaded():
    """Pastikan model sudah di-load. Auto-load jika belum."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        logger.info("Model belum di-load, auto-loading...")
        load_model()


def _format_rag_context(dokumen: list) -> str:
    """
    Format list dokumen menjadi string konteks RAFT.
    Format harus SAMA PERSIS dengan training data.
    """
    formatted_docs = []
    for idx, doc in enumerate(dokumen, start=1):
        doc_text = str(doc).strip() if doc is not None else "[Dokumen kosong]"
        formatted_docs.append(f'<doc id="{idx}">\n{doc_text}\n</doc>')
    return "\n\n".join(formatted_docs)


def generate_answer(pertanyaan: str, max_new_tokens: int = 2048,
                    temperature: float = 0.1, top_p: float = 0.9,
                    repetition_penalty: float = 1.0) -> dict:
    """
    Generate jawaban TANPA konteks dokumen (plain chat).
    Cocok untuk model base atau model Q&A sederhana.

    Args:
        pertanyaan: Pertanyaan user
        max_new_tokens: Jumlah maksimal token yang di-generate
        temperature: Suhu sampling (lebih rendah = lebih deterministik)
        top_p: Top-p (nucleus) sampling
        repetition_penalty: Penalty untuk repetisi

    Returns:
        dict dengan key: answer, model_type, model_path
    """
    _ensure_model_loaded()

    messages = [
        {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": pertanyaan},
    ]

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = _model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    response = _tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return {
        "answer": response.strip(),
        "model_type": _model_type,
        "model_path": _model_path,
    }


def _split_cot_answer(raw_response: str) -> tuple:
    """
    Pisahkan Chain-of-Thought (analisis) dari jawaban akhir.
    Model RAFT menghasilkan: {analisis/CoT}\n\n{jawaban}
    Returns: tuple (analisis, jawaban)
    """
    import re
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL | re.IGNORECASE)
    
    if thought_match:
        analisis = thought_match.group(1).strip()
        jawaban = raw_response[thought_match.end():].strip()
        
        # Bersihkan awalan "Jawaban:" jika ada
        if jawaban.lower().startswith("jawaban:"):
            jawaban = jawaban[len("jawaban:"):].strip()
            
        return analisis, jawaban
        
    return "", raw_response.strip()


def _extract_doc_number(analisis: str) -> Optional[int]:
    """Ekstrak nomor dokumen relevan dari analisis CoT."""
    patterns = [
        r"[Dd]okumen\s+(\d+)\s+.*?[Rr]elevan",
        r"[Dd]okumen\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, analisis)
        if match:
            return int(match.group(1))
    return None


def _extract_doc_content(dokumen: list, doc_number: int) -> str:
    """Ekstrak inti konten dari dokumen relevan (1-based index)."""
    idx = doc_number - 1
    if idx < 0 or idx >= len(dokumen):
        return ""

    content = dokumen[idx].strip()
    lines = content.split("\n")
    body_lines = []
    skip_header = True
    for line in lines:
        stripped = line.strip()
        if skip_header and (stripped.lower().startswith("pasal") or stripped == ""):
            continue
        skip_header = False
        body_lines.append(stripped)

    body = " ".join(body_lines).strip()
    if len(body) > 300:
        last_period = body[:300].rfind(".")
        if last_period > 50:
            body = body[:last_period + 1]
        else:
            body = body[:300].rsplit(" ", 1)[0] + "."
    return body







def generate_answer_rag(pertanyaan: str, dokumen: list,
                        max_new_tokens: int = 2048,
                        temperature: float = 0.7, top_p: float = 0.9,
                        repetition_penalty: float = 1.0) -> dict:
    """
    Generate jawaban DENGAN konteks dokumen (RAG / RAFT format).
    Format prompt HARUS SAMA PERSIS dengan training data RAFT.

    Response dipisah menjadi 2 field:
    - analisis: Chain-of-Thought (evaluasi relevansi dokumen)
    - jawaban: Jawaban akhir yang ringkas dan lengkap

    Returns:
        dict dengan key: analisis, jawaban, raw_response, model_type, model_path, num_documents
    """
    _ensure_model_loaded()

    docs_text = _format_rag_context(dokumen)
    user_message = f"Pertanyaan: {pertanyaan}\n\nDokumen Referensi:\n{docs_text}"

    messages = [
        {"role": "system", "content": RAFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = _model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=50,
        repetition_penalty=repetition_penalty,
        min_p=0.05,
    )

    response = _tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    raw_response = response.strip()
    analisis, jawaban = _split_cot_answer(raw_response)

    # Coba parse analisis menjadi dictionary seperti struktur dataset awal
    import ast
    parsed_analisis = analisis
    try:
        if analisis.strip().startswith("{"):
            cleaned = analisis.strip()
            # Perbaiki common model generation errors (misal kurung kurawal ganda)
            cleaned = cleaned.replace("}},", "},")
            parsed_analisis = ast.literal_eval(cleaned)
    except Exception as e:
        logger.warning(f"Gagal mem-parse analisis menjadi dict: {e}")

    return {
        "analisis": parsed_analisis, # Terstruktur seperti dataset jika berhasil diparse
        "thought_process": parsed_analisis, # Alias agar mirip dengan dataset
        "completion": jawaban, # Alias agar mirip dengan dataset
        "jawaban": jawaban,
        "raw_response": raw_response,
        "model_type": _model_type,
        "model_path": _model_path,
        "num_documents": len(dokumen),
    }


def get_model_info() -> dict:
    """Info model yang sedang aktif."""
    return {
        "loaded": _model is not None,
        "model_path": _model_path,
        "model_type": _model_type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def get_error_detail(e: Exception) -> str:
    """Format error detail untuk logging."""
    return f"{type(e).__name__}: {str(e) or '(no message)'}\n{traceback.format_exc()}"
