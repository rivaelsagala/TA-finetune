"""
RAFT (Retrieval-Augmented Fine-Tuning) Dataset Generator
Two-phase approach:
  Phase 1 — Generate all samples with random style variation (no retry/repair).
  Phase 2 — Scan output, validate, regenerate only failed samples with different style + feedback.
  Final dataset = merge original valid + repaired.

Style system: 4 styles (direct, explanatory, elaborated, structured), equal 25% each,
separate tailored prompts.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Load environment ─────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = (
    DATA_DIR
    / "PERATURAN_DESA_LOA_KECAMATAN_PASEH_KABUPATEN_BANDUNG_NOMOR___5_TAHUN_2017__chunks.json"
)
OUTPUT_FILE = DATA_DIR / "raft_perdes_dataset.jsonl"
REPAIRED_FILE = DATA_DIR / "raft_perdes_dataset_repaired.jsonl"

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    generator_model: str = "openai/gpt-4o-mini"
    max_tokens: int = 1024
    temperature_question: float = 0.8
    temperature_thought: float = 0.6
    temperature_completion: float = 0.7
    top_p: float = 0.9
    request_delay: float = 1.5
    num_distractors: int = 2
    chunks_per_doc: Optional[int] = 5  # None = full
    min_completion_length: int = 80
    max_retries: int = 2
    seed: int = 42


# ─── Style Definitions ────────────────────────────────────────────────────────
# 4 styles, equal 25% probability, each with separate tailored system+user prompt.
# Same base constraints, different format directives.

_BASE_CONSTRAINTS = (
    "Aturan umum (WAJIB ditaati):\n"
    "- Gunakan bahasa Indonesia yang natural dan profesional.\n"
    "- JANGAN pernah menyebut 'Dokumen 1', 'Dokumen 2', 'Dokumen 3', dst. "
    "Gunakan nomor Pasal, nama Peraturan, atau 'Peraturan Desa' sebagai rujukan.\n"
    "- JANGAN mengarang informasi yang tidak ada di dokumen yang diberikan.\n"
    "- Jika dokumen tidak mengandung jawaban yang memadai, nyatakan secara eksplisit "
    "bahwa informasi tidak ditemukan dalam dokumen yang diberikan.\n"
    "- Jawab dalam 2-4 kalimat, minimal 80 karakter.\n"
    "- Sertakan minimal satu fakta spesifik (angka, definisi, nama jabatan, atau istilah hukum) "
    "yang terdapat dalam dokumen sumber.\n"
)

STYLE_PROMPTS: Dict[str, Dict[str, str]] = {
    "direct": {
        "system": (
            "Anda adalah asisten hukum profesional yang menjawab secara langsung dan ringkas.\n\n"
            + _BASE_CONSTRAINTS
            + "\nFormat jawaban (gaya langsung):\n"
            "- Jawaban langsung ke inti, tanpa pembuka panjang.\n"
            "- Gunakan pola: '[Konsep] adalah [definisi/ketentuan], sebagaimana diatur dalam Pasal [N].'\n"
            "- Hindari frasa pembuka seperti 'Berdasarkan dokumen...' atau 'Menurut peraturan...' "
            "kecuali diperlukan untuk kejelasan.\n"
        ),
        "user_suffix": "Jawab secara langsung dan ringkas:",
    },
    "explanatory": {
        "system": (
            "Anda adalah asisten hukum profesional yang menjawab dengan memberikan penjelasan sebab-akibat.\n\n"
            + _BASE_CONSTRAINTS
            + "\nFormat jawaban (gaya penjelasan):\n"
            "- Mulai dengan pernyataan fakta/definisi, lalu jelaskan MENGAPA hal itu penting atau apa akibatnya.\n"
            "- Gunakan pola: '[Konsep] adalah [definisi]. Hal ini penting karena [alasan]...'\n"
            "- Berikan konteks tambahan yang membantu pemahaman pembaca.\n"
        ),
        "user_suffix": "Jawab dengan penjelasan sebab-akibat:",
    },
    "elaborated": {
        "system": (
            "Anda adalah asisten hukum profesional yang menjawab dengan elaborasi dan konteks latar belakang.\n\n"
            + _BASE_CONSTRAINTS
            + "\nFormat jawaban (gaya elaborasi):\n"
            "- Mulai dengan merujuk Pasal/Peraturan, lalu jabarkan isinya, lalu jelaskan tujuan atau latar belakangnya.\n"
            "- Gunakan pola: 'Menurut Pasal [N], [konsep] adalah [definisi]. Ketentuan ini bertujuan untuk [tujuan]...'\n"
            "- Berikan elaborasi yang informatif tanpa mengarang.\n"
        ),
        "user_suffix": "Jawab dengan elaborasi dan konteks:",
    },
    "structured": {
        "system": (
            "Anda adalah asisten hukum profesional yang menjawab dalam format terstruktur/poin.\n\n"
            + _BASE_CONSTRAINTS
            + "\nFormat jawaban (gaya terstruktur):\n"
            "- Sajikan jawaban dalam bentuk poin bernomor di dalam satu kalimat atau beberapa kalimat.\n"
            "- Gunakan pola: '[Konsep] mencakup: (1) [poin 1], (2) [poin 2], (3) [poin 3].'\n"
            "- Setiap poin harus ringkas namun informatif.\n"
        ),
        "user_suffix": "Jawab dalam format terstruktur:",
    },
}

STYLE_NAMES: List[str] = list(STYLE_PROMPTS.keys())
STYLE_WEIGHTS: List[float] = [0.30, 0.30, 0.25, 0.15]  # direct, explanatory, elaborated, structured

# Anti-monotony: rolling window of recent completions for diversity checking
_recent_completions: deque = deque(maxlen=5)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── API Client ───────────────────────────────────────────────────────────────

def call_api(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    max_attempts: int = 3,
) -> str:
    """Call OpenAI-compatible API with retry and exponential backoff.

    Handles 429, 503, timeouts, and general errors.
    Raises RuntimeError after all attempts exhausted.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code in (429, 503):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [API] Rate limited ({resp.status_code}), waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            print(f"  [API] Timeout on attempt {attempt + 1}/{max_attempts}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"  [API] Error on attempt {attempt + 1}/{max_attempts}: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {max_attempts} attempts")


def clean_chunk_content(text: str) -> str:
    """Normalize whitespace and strip chunk text."""
    return re.sub(r"\s+", " ", text).strip()


def load_all_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load chunk JSON file and return list of chunk dicts."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path.name}")
    return chunks


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    """Write samples to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — GENERATION (no retry, no repair)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_question(oracle_content: str, config: GeneratorConfig) -> str:
    """Stage 2: Generate a question answerable from the oracle chunk content."""
    snippet = oracle_content[:400]
    system_prompt = (
        "Anda adalah pembuat pertanyaan untuk dataset fine-tuning hukum. "
        "Buat SATU pertanyaan spesifik dan menantang berdasarkan teks berikut. "
        "Pertanyaan harus:\n"
        "- Hanya bisa dijawab secara lengkap dari teks ini, bukan pertanyaan umum\n"
        "- Menggunakan istilah atau konsep spesifik yang ada dalam teks\n"
        "- Bukan pertanyaan ya/tidak sederhana\n"
        "- Menggunakan bahasa Indonesia formal dan jelas\n"
        "Jawab HANYA dengan satu pertanyaan, tanpa penjelasan tambahan."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Teks sumber:\n{snippet}\n\nBuat satu pertanyaan spesifik:"},
    ]
    question = call_api(
        messages, model=config.generator_model,
        temperature=config.temperature_question, max_tokens=256, top_p=config.top_p,
    )
    time.sleep(config.request_delay)
    return question


def select_distractors(
    oracle_idx: int, all_chunks: List[Dict[str, Any]], num_distractors: int,
) -> List[int]:
    """Stage 3a: Pick random distractor chunk indices (excluding the oracle)."""
    candidates = [i for i in range(len(all_chunks)) if i != oracle_idx]
    return random.sample(candidates, min(num_distractors, len(candidates)))


def arrange_documents(
    oracle_content: str, distractor_contents: List[str],
) -> Tuple[List[str], int]:
    """Stage 3b: Place oracle at a random position among distractors."""
    docs = list(distractor_contents)
    oracle_pos = random.randint(0, len(docs))
    docs.insert(oracle_pos, oracle_content)
    return docs, oracle_pos


def generate_thought_process(
    question: str, documents: List[str], oracle_pos: int, config: GeneratorConfig,
) -> str:
    """Stage 4: Generate chain-of-thought analyzing every document.

    Marks irrelevant as '(Abaikan)', oracle as '(Sangat Relevan)'. Plain text output.
    """
    doc_text = "\n\n".join(f"Dokumen {i+1}:\n{doc}" for i, doc in enumerate(documents))
    system_prompt = (
        "Anda adalah asisten yang menganalisis dokumen untuk menjawab pertanyaan. "
        "Analisis SETIAP dokumen secara berurutan:\n"
        "- Untuk dokumen yang TIDAK relevan: tandai dengan '(Abaikan)' dan jelaskan singkat mengapa.\n"
        "- Untuk dokumen yang PALING relevan: tandai dengan '(Sangat Relevan)' dan jelaskan bagaimana "
        "dokumen tersebut menjawab pertanyaan.\n"
        "- Untuk dokumen yang agak relevan: berikan analisis singkat.\n"
        "Gunakan format teks biasa (bukan JSON)."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Pertanyaan: {question}\n\n{doc_text}\n\n"
                                     "Analisis setiap dokumen di atas."},
    ]
    thought = call_api(
        messages, model=config.generator_model,
        temperature=config.temperature_thought,
        max_tokens=config.max_tokens, top_p=config.top_p,
    )
    time.sleep(config.request_delay)
    return thought


def generate_completion_with_style(
    question: str,
    documents: List[str],
    config: GeneratorConfig,
    style: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Tuple[str, str]:
    """Stage 5: Generate completion using a specific style's tailored prompt.

    Args:
        question: The question to answer.
        documents: Ordered document list.
        config: Generator config.
        style: Style name to use. If None, picks randomly.
        feedback: Optional feedback string appended to user prompt (for repair phase).

    Returns:
        (completion_text, style_used)
    """
    if style is None:
        style = random.choices(STYLE_NAMES, weights=STYLE_WEIGHTS, k=1)[0]

    style_info = STYLE_PROMPTS[style]
    doc_text = "\n\n".join(f"Dokumen {i+1}:\n{doc}" for i, doc in enumerate(documents))

    user_content = f"{doc_text}\n\nPertanyaan: {question}\n\n{style_info['user_suffix']}"
    if feedback:
        user_content += f"\n\nCatatan perbaikan: {feedback}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": style_info["system"]},
        {"role": "user", "content": user_content},
    ]
    completion = call_api(
        messages, model=config.generator_model,
        temperature=config.temperature_completion,
        max_tokens=config.max_tokens, top_p=config.top_p,
    )
    time.sleep(config.request_delay)
    return completion, style


def generate_one_sample(
    oracle_idx: int,
    all_chunks: List[Dict[str, Any]],
    config: GeneratorConfig,
) -> Dict[str, Any]:
    """Run the full generation pipeline for one oracle chunk (Phase 1, no retry)."""
    oracle_content = clean_chunk_content(all_chunks[oracle_idx].get("content", ""))

    question = generate_question(oracle_content, config)

    distractor_idxs = select_distractors(oracle_idx, all_chunks, config.num_distractors)
    distractor_contents = [
        clean_chunk_content(all_chunks[i].get("content", "")) for i in distractor_idxs
    ]
    documents, oracle_pos = arrange_documents(oracle_content, distractor_contents)

    thought_process = generate_thought_process(question, documents, oracle_pos, config)

    completion, style_used = generate_completion_with_style(question, documents, config)

    context = "\n\n".join(f"Dokumen {i+1}:\n{doc}" for i, doc in enumerate(documents))

    return {
        "instruction": question,
        "input": context,
        "output": completion,
        "thought_process": thought_process,
        "oracle_index": oracle_idx,
        "oracle_position": oracle_pos,
        "style_used": style_used,
    }


def run_generation(config: GeneratorConfig) -> List[Dict[str, Any]]:
    """Phase 1: Generate all samples with random styles, no retry/repair."""
    random.seed(config.seed)
    chunks = load_all_chunks(CHUNKS_FILE)

    if config.chunks_per_doc is not None:
        indices = random.sample(range(len(chunks)), min(config.chunks_per_doc, len(chunks)))
    else:
        indices = list(range(len(chunks)))

    print(f"[Phase 1] Generating {len(indices)} samples...")

    samples: List[Dict[str, Any]] = []
    for idx in tqdm(indices, desc="Phase 1: Generating"):
        try:
            sample = generate_one_sample(idx, chunks, config)
            samples.append(sample)
        except Exception as e:
            print(f"\n  [!] Failed for chunk {idx}: {e}")
            continue

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — VALIDATION & REPAIR
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Validation helpers ───────────────────────────────────────────────────────

def _word_set(text: str) -> Set[str]:
    """Lowercase word set for set-based comparisons."""
    return set(re.findall(r"\w+", text.lower()))


def validate_completion(
    completion: str,
    instruction: str,
    oracle_content: str,
    recent_completions: List[str],
) -> Tuple[bool, List[str]]:
    """Validate a generated completion against quality criteria.

    Returns (is_valid, list_of_failed_check_names).
    Checks: min_length, no_doc_reference, has_specific_fact, not_formulaic, anti_monotony.
    """
    failed: List[str] = []

    # min_length
    if len(completion) < 80:
        failed.append("min_length")

    # no_doc_reference
    if re.search(r"[Dd]okumen\s+\d", completion):
        failed.append("no_doc_reference")

    # has_specific_fact — overlap of key terms/numbers with oracle
    oracle_key = {w for w in _word_set(oracle_content) if len(w) >= 4}
    completion_words = _word_set(completion)
    overlap = completion_words & oracle_key
    oracle_nums = set(re.findall(r"\d+", oracle_content))
    completion_nums = set(re.findall(r"\d+", completion))
    if len(overlap) < 2 and len(oracle_nums & completion_nums) < 1:
        failed.append("has_specific_fact")

    # not_formulaic — rigid template as ENTIRE completion
    if re.match(
        r"^.+,\s*sesuai dengan ketentuan dalam Pasal \d+ Peraturan Desa .+ No\.\s*\d+ Tahun \d+\.?$",
        completion,
    ):
        failed.append("not_formulaic")

    # anti_monotony — Jaccard similarity against recent completions
    if recent_completions:
        new_words = _word_set(completion)
        for prev in recent_completions:
            prev_words = _word_set(prev)
            union = new_words | prev_words
            if not union:
                continue
            jaccard = len(new_words & prev_words) / len(union)
            if jaccard > 0.7:
                failed.append("anti_monotony")
                break

    return (len(failed) == 0, failed)


def _extract_oracle_content(sample: Dict[str, Any]) -> str:
    """Extract oracle document text from a sample's 'input' field."""
    oracle_pos = sample.get("oracle_position", 0)
    # Parse documents from the context string
    parts = re.split(r"Dokumen \d+:\n", sample.get("input", ""))
    # parts[0] is empty, parts[1..n] are doc contents
    docs = [p.strip() for p in parts[1:]]
    if 0 <= oracle_pos < len(docs):
        return docs[oracle_pos]
    return ""


def _build_feedback(failed_checks: List[str]) -> str:
    """Build human-readable feedback string from failed check names."""
    feedback_parts = []
    if "min_length" in failed_checks:
        feedback_parts.append("jawaban terlalu pendek (minimal 80 karakter)")
    if "no_doc_reference" in failed_checks:
        feedback_parts.append(
            "jangan menyebut 'Dokumen 1/2/3', gunakan nomor Pasal atau 'Peraturan Desa'"
        )
    if "has_specific_fact" in failed_checks:
        feedback_parts.append(
            "sertakan fakta spesifik (angka, nama jabatan, istilah) dari dokumen sumber"
        )
    if "not_formulaic" in failed_checks:
        feedback_parts.append("jangan gunakan format kaku/berulang, buat lebih natural")
    if "anti_monotony" in failed_checks:
        feedback_parts.append(
            "jawaban terlalu mirip dengan jawaban sebelumnya, gunakan gaya dan variasi kata yang berbeda"
        )
    return (
        "Jawaban sebelumnya tidak memenuhi kriteria: "
        + "; ".join(feedback_parts)
        + ". Perbaiki."
    )


def repair_sample(
    sample: Dict[str, Any],
    failed_checks: List[str],
    config: GeneratorConfig,
    recent_completions: List[str],
) -> Optional[Dict[str, Any]]:
    """Attempt to repair a failed sample by regenerating completion with a different style.

    Picks a style DIFFERENT from the original, includes feedback in the prompt.
    Validates the new completion. Returns repaired sample or None if still failing.
    """
    original_style = sample.get("style_used", "direct")
    # Use weighted random style (different from original if possible)
    other_styles = [s for s in STYLE_NAMES if s != original_style]
    other_weights = [STYLE_WEIGHTS[STYLE_NAMES.index(s)] for s in other_styles]
    if not other_styles:
        return None

    new_style = random.choices(other_styles, weights=other_weights, k=1)[0]
    feedback = _build_feedback(failed_checks)

    # Re-parse documents and question from the sample
    question = sample["instruction"]
    parts = re.split(r"Dokumen \d+:\n", sample.get("input", ""))
    documents = [p.strip() for p in parts[1:] if p.strip()]
    oracle_content = _extract_oracle_content(sample)

    if not documents:
        return None

    # Regenerate completion with different style + feedback
    new_completion, style_used = generate_completion_with_style(
        question, documents, config, style=new_style, feedback=feedback,
    )

    # Validate the new completion
    is_valid, new_failed = validate_completion(
        new_completion, question, oracle_content, recent_completions,
    )

    if not is_valid:
        return None

    # Build repaired sample (keep original question, thought, context)
    repaired = dict(sample)
    repaired["output"] = new_completion
    repaired["style_used"] = style_used
    repaired["repaired"] = True
    return repaired


def run_validation_and_repair(
    samples: List[Dict[str, Any]], config: GeneratorConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Phase 2: Scan all samples, validate, repair failures.

    Returns (valid_samples, repaired_samples).
    Final dataset = valid_samples + repaired_samples.
    """
    print(f"\n[Phase 2] Validating {len(samples)} samples...")

    valid: List[Dict[str, Any]] = []
    repaired: List[Dict[str, Any]] = []
    failed_unrecoverable: List[Dict[str, Any]] = []

    # Build rolling window of recent completions (kept for API compatibility)
    recent_completions: List[str] = []
    _recent_completions.clear()  # Reset shared deque for fresh run

    for i, sample in enumerate(tqdm(samples, desc="Phase 2: Validating")):
        oracle_content = _extract_oracle_content(sample)
        is_valid, failed_checks = validate_completion(
            sample["output"], sample["instruction"], oracle_content,
            recent_completions[-5:],
        )

        if is_valid:
            valid.append(sample)
            recent_completions.append(sample["output"])
            _recent_completions.append(sample["output"])
        else:
            # Try to repair
            fixed = repair_sample(sample, failed_checks, config, recent_completions[-5:])
            if fixed is not None:
                repaired.append(fixed)
                recent_completions.append(fixed["output"])
                _recent_completions.append(fixed["output"])
                print(f"  [Repaired] sample {i}: style={sample.get('style_used')} -> {fixed.get('style_used')}, failed={failed_checks}")
            else:
                failed_unrecoverable.append(sample)
                print(f"  [Failed]   sample {i}: checks={failed_checks}, could not repair")

    print(f"\nPhase 2 summary: {len(valid)} valid, {len(repaired)} repaired, "
          f"{len(failed_unrecoverable)} unrecoverable")
    return valid, repaired


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT & STATS
# ═══════════════════════════════════════════════════════════════════════════════

def print_stats(samples: List[Dict[str, Any]], label: str = "Dataset") -> None:
    """Print dataset statistics and 2 sample outputs."""
    if not samples:
        print(f"[{label}] No samples.")
        return
    n = len(samples)
    avg_q = sum(len(s["instruction"]) for s in samples) / n
    avg_c = sum(len(s["output"]) for s in samples) / n
    avg_t = sum(len(s["thought_process"]) for s in samples) / n
    min_c = min(len(s["output"]) for s in samples)
    max_c = max(len(s["output"]) for s in samples)

    # Style distribution
    style_counts: Dict[str, int] = {}
    for s in samples:
        st = s.get("style_used", "unknown")
        style_counts[st] = style_counts.get(st, 0) + 1

    print(f"\n{'='*50}")
    print(f"[{label}]")
    print(f"Total samples  : {n}")
    print(f"Avg question   : {avg_q:.0f} chars")
    print(f"Avg completion : {avg_c:.0f} chars (min={min_c}, max={max_c})")
    print(f"Avg thought    : {avg_t:.0f} chars")
    print(f"{'='*50}")

    # Style variation report
    print(f"\n  Style variation report:")
    for style_name in STYLE_NAMES:
        count = style_counts.get(style_name, 0)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {style_name:<12} : {count:>4} ({pct:5.1f}%) {bar}")
    repaired_count = sum(1 for s in samples if s.get("repaired"))
    print(f"    Repaired       : {repaired_count:>4} ({repaired_count / n * 100:5.1f}%)")

    for i, idx in enumerate([0, -1] if n > 1 else [0]):
        s = samples[idx]
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {s['instruction']}")
        print(f"A: {s['output'][:300]}...")
        if s.get("repaired"):
            print(f"   (repaired, style={s.get('style_used')})")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point: Phase 1 (generate) -> Phase 2 (validate+repair) -> merge -> write."""
    config = GeneratorConfig()
    random.seed(config.seed)

    print("=" * 50)
    print("RAFT Dataset Generator (Two-Phase)")
    print("=" * 50)
    print(f"Model       : {config.generator_model}")
    print(f"Chunks file : {CHUNKS_FILE}")
    print(f"Output      : {OUTPUT_FILE}")
    print(f"Repaired    : {REPAIRED_FILE}")
    print(f"Samples     : {config.chunks_per_doc or 'all'}")
    print(f"Styles      : {', '.join(STYLE_NAMES)} (weighted: {dict(zip(STYLE_NAMES, STYLE_WEIGHTS))})")
    print()

    # Test API connection
    print("Testing API connection...")
    try:
        result = call_api(
            [{"role": "user", "content": "Say 'OK' in one word."}],
            model=config.generator_model, temperature=0.0, max_tokens=5, max_attempts=2,
        )
        print(f"  API OK: '{result}'")
    except Exception as e:
        print(f"  API test FAILED: {e}")
        return

    # Backup existing output
    if OUTPUT_FILE.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUTPUT_FILE.with_suffix(f".jsonl.bak_{ts}")
        shutil.copy2(OUTPUT_FILE, backup)
        print(f"Backed up existing output to {backup.name}")

    # ── Phase 1: Generate all samples ────────────────────────────────────────
    raw_samples = run_generation(config)
    # Write raw output (before repair)
    write_jsonl(raw_samples, OUTPUT_FILE)
    print_stats(raw_samples, label="Phase 1 — Raw")

    # ── Phase 2: Validate & Repair ───────────────────────────────────────────
    valid, repaired = run_validation_and_repair(raw_samples, config)

    # Write repaired-only file
    if repaired:
        write_jsonl(repaired, REPAIRED_FILE)

    # ── Final dataset: merge valid + repaired ────────────────────────────────
    final = valid + repaired
    write_jsonl(final, OUTPUT_FILE)
    print_stats(final, label="Final Dataset")


if __name__ == "__main__":
    main()
