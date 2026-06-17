"""
llama_service.py
================
Model loading, inference, and helper utilities for the RAG Perdes API.

Model resolution order (first found is used):
  1. notebooks/model_merged_rag_perdes/   (fine-tuned, relative to /workspace)
  2. model_merged_rag_perdes/             (fine-tuned, relative to BE/)
  3. /workspace/model/Meta-Llama-3.1-8B-Instruct/  (base model fallback)

Uses Unsloth's FastLanguageModel for efficient 4-bit inference (QLoRA style).
"""

import os
import sys
import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Workspace root is one level above BE/
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Fine-tuned merged model candidates (checked in order)
MODEL_PATHS = [
    os.path.join(WORKSPACE_ROOT, "notebooks", "model_merged_rag_perdes"),
    os.path.join(WORKSPACE_ROOT, "model_merged_rag_perdes"),
]

# Base model fallback
BASE_MODEL_NAME = os.path.join(WORKSPACE_ROOT, "model", "Meta-Llama-3.1-8B-Instruct")

# Inference config
MAX_SEQ_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_loaded_model_path: Optional[str] = None
_load_error: Optional[str] = None

# ---------------------------------------------------------------------------
# RAG system prompt (must match what was used during fine-tuning)
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Kabupaten Bandung.\n\n"
    "Di bawah ini terdapat beberapa dokumen peraturan. Satu dokumen adalah "
    "GOLD_DOCUMENT (sumber jawaban yang benar), sisanya adalah DISTRACTOR "
    "(dokumen tidak relevan yang sengaja disertakan). Gunakan HANYA informasi "
    "dari GOLD_DOCUMENT untuk menjawab pertanyaan."
)

DEFAULT_SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Indonesia. Jawab dengan jelas dan lengkap "
    "berdasarkan pengetahuan Anda."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_path() -> str:
    """
    Walk MODEL_PATHS in order; return the first directory that exists.
    Falls back to BASE_MODEL_NAME if none found.
    """
    for path in MODEL_PATHS:
        if os.path.isdir(path):
            return path
    if os.path.isdir(BASE_MODEL_NAME):
        return BASE_MODEL_NAME
    raise FileNotFoundError(
        f"No model directory found. Checked:\n"
        + "\n".join(f"  - {p}" for p in MODEL_PATHS)
        + f"\n  - {BASE_MODEL_NAME} (base fallback)"
    )


def get_error_detail(e: Exception) -> str:
    """Return full traceback string for logging."""
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def get_model_info() -> dict:
    """Return metadata about the currently loaded model."""
    return {
        "loaded": _model is not None,
        "model_path": _loaded_model_path,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_error": _load_error,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[str] = None) -> None:
    """
    Load the model and tokenizer into global state using Unsloth.

    If `model_path` is None, resolves automatically via `_resolve_model_path()`.
    Calling this when a model is already loaded is a no-op (unless path differs).
    """
    global _model, _tokenizer, _loaded_model_path, _load_error

    resolved_path = model_path or _resolve_model_path()

    # Skip if already loaded with the same path
    if _model is not None and _loaded_model_path == resolved_path:
        logger.info(f"Model already loaded from: {resolved_path}")
        return

    _load_error = None
    logger.info(f"Loading model from: {resolved_path}")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=resolved_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,          # auto-detect Float16 / BFloat16
            load_in_4bit=True,   # QLoRA 4-bit inference
            device_map="auto",
        )

        # Switch to inference mode (disables training-only ops)
        FastLanguageModel.for_inference(model)

        _model = model
        _tokenizer = tokenizer
        _loaded_model_path = resolved_path

        logger.info(f"Model loaded successfully: {resolved_path}")

    except Exception as e:
        _load_error = f"{type(e).__name__}: {str(e) or '(no message)'}"
        logger.error(f"Failed to load model:\n{get_error_detail(e)}")
        raise


def _ensure_model_loaded() -> None:
    """Lazy-load model on first call if not already in memory."""
    if _model is None:
        load_model()


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _extract_system_and_docs(instruction_text: str) -> tuple:
    """
    Split a RAFT instruction into (system_prompt, docs_section).

    The boundary marker is '=== DOKUMEN KONTEKS ==='.
    If the marker is absent, returns ("", full_instruction).
    """
    marker = "=== DOKUMEN KONTEKS ==="
    idx = instruction_text.find(marker)
    if idx >= 0:
        system_prompt = instruction_text[:idx].strip()
        docs_section = instruction_text[idx:].strip()
    else:
        system_prompt = ""
        docs_section = instruction_text.strip()
    return system_prompt, docs_section


def _build_messages_chat(message: str, system_prompt: Optional[str] = None) -> list:
    """Build chat messages for plain chat (no RAG context)."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": message},
    ]


def _build_messages_rag(message: str, konteks: str) -> list:
    """
    Build chat messages for RAG mode.

    `konteks` is the full document block including:
      === DOKUMEN KONTEKS ===
      [DOKUMEN 1] [GOLD_DOCUMENT] ...
      ---
      [DOKUMEN 2] [DISTRACTOR] ...
      === AKHIR DOKUMEN KONTEKS ===
    """
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"{konteks}\n\n{message}"},
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _generate(messages: list, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    """
    Run inference given a list of chat messages.

    Returns the decoded assistant response string.
    """
    _ensure_model_loaded()

    eos_token = _tokenizer.eos_token

    # Apply the Llama 3.1 chat template
    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = _model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # Decode only the generated portion (exclude the prompt tokens)
    generated_tokens = output_ids[0][input_ids.shape[1]:]
    response = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def generate_answer(
    message: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate an answer without RAG context (plain chat).

    Args:
        message:        User's question.
        system_prompt:  Optional system prompt override.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated answer string.
    """
    messages = _build_messages_chat(message, system_prompt)
    return _generate(messages, max_new_tokens)


def generate_answer_rag(
    message: str,
    konteks: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Generate an answer with RAG context (fine-tuned RAFT model).

    Args:
        message:        User's question.
        konteks:        Document context block (GOLD + DISTRACTOR docs).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated answer string grounded in the provided context.
    """
    messages = _build_messages_rag(message, konteks)
    return _generate(messages, max_new_tokens)
