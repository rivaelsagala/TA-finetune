"""
=============================================================================
Improve RAFT Dataset Completions (Tanpa Regenerate Ulang)
=============================================================================

Script ini memperbaiki field `completion`/`output` pada dataset RAFT yang sudah ada
TANPA perlu men-generate ulang dataset dari awal.

Masalah yang diperbaiki:
  1. Completion yang mengandung referensi "Dokumen N" / "(Dokumen N)"
  2. Completion yang terlalu pendek (< 80 karakter)
  3. Completion yang formulaic atau terlalu mirip dengan completion sebelumnya
  4. Completion tanpa fakta spesifik dari dokumen sumber

Cara kerja:
  - Membaca raft_perdes_dataset.jsonl yang ada
  - Validasi setiap completion menggunakan validate_completion (shared validator)
  - Untuk completion bermasalah, regenerate dengan style acak (weighted) + feedback
  - Simpan hasil ke file baru (original di-backup)

Usage:
  python improve_completions.py

Requirements:
  pip install requests python-dotenv tqdm

=============================================================================
"""

import os
import sys
import json
import re
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv

# ============================================================================
# KONFIGURASI — Load .env SEBELUM import generator (agar env vars tersedia)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "raft_perdes_dataset.jsonl"
OUTPUT_FILE = DATA_DIR / "raft_perdes_dataset_improved.jsonl"
BACKUP_FILE = DATA_DIR / "raft_perdes_dataset_backup.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# Import shared utilities dari generator
# ============================================================================

# Tambahkan parent directory agar bisa import dari notebooks package
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_raft_dataset as gen

# Shared utilities
call_api = gen.call_api
validate_completion = gen.validate_completion
_build_feedback = gen._build_feedback
_extract_oracle_content = gen._extract_oracle_content
_word_set = gen._word_set
STYLE_PROMPTS = gen.STYLE_PROMPTS
STYLE_NAMES = gen.STYLE_NAMES
STYLE_WEIGHTS = gen.STYLE_WEIGHTS
GeneratorConfig = gen.GeneratorConfig
_recent_completions = gen._recent_completions

from tqdm import tqdm

REQUEST_DELAY = 1.5  # detik antara request

print(f"Model  : {GeneratorConfig().generator_model}")
print(f"Input  : {INPUT_FILE}")
print(f"Output : {OUTPUT_FILE}")


# ============================================================================
# Helper: Parse documents dari field 'input' (context string)
# ============================================================================

def parse_documents_from_input(input_text: str) -> List[str]:
    """Parse 'Dokumen N:\n...' format dari field input menjadi list dokumen."""
    parts = re.split(r"Dokumen \d+:\n", input_text)
    return [p.strip() for p in parts[1:] if p.strip()]


def extract_oracle_content_from_sample(sample: Dict) -> str:
    """Extract oracle text dari sample, support both 'input' dan 'documents' field."""
    if "documents" in sample and isinstance(sample["documents"], list):
        # Format lama: documents sebagai list
        oracle_pos = sample.get("oracle_position", 0)
        docs = sample["documents"]
        if 0 <= oracle_pos < len(docs):
            return docs[oracle_pos]
        return docs[0] if docs else ""
    elif "input" in sample:
        # Format baru: input sebagai context string
        return _extract_oracle_content(sample)
    return ""


def get_completion(sample: Dict) -> str:
    """Get completion text, support both 'output' dan 'completion' field."""
    return sample.get("output", sample.get("completion", ""))


def set_completion(sample: Dict, text: str) -> None:
    """Set completion text ke field yang sesuai."""
    if "output" in sample:
        sample["output"] = text
    else:
        sample["completion"] = text


def get_documents(sample: Dict) -> List[str]:
    """Get document list dari sample."""
    if "documents" in sample and isinstance(sample["documents"], list):
        return sample["documents"]
    elif "input" in sample:
        return parse_documents_from_input(sample["input"])
    return []


# ============================================================================
# Perbaiki Completion Menggunakan AI (shared generator style)
# ============================================================================

def improve_completion(
    instruction: str,
    documents: List[str],
    thought_process: str,
    old_completion: str,
    failed_checks: List[str],
    config: GeneratorConfig,
) -> Optional[str]:
    """Perbaiki satu completion menggunakan AI dengan style acak (weighted).

    Menggunakan style system prompt dari generator dan feedback dari validator.
    """
    # Pilih style secara weighted random
    style = random.choices(STYLE_NAMES, weights=STYLE_WEIGHTS, k=1)[0]
    style_info = STYLE_PROMPTS[style]

    # Bangun konteks dokumen
    doc_text = "\n\n".join(f"Dokumen {i+1}:\n{doc[:600]}" for i, doc in enumerate(documents))

    # Build feedback dari failed checks
    feedback = _build_feedback(failed_checks)

    user_content = (
        f"{doc_text}\n\n"
        f"Pertanyaan: {instruction}\n\n"
        f"Analisis (untuk konteks): {thought_process[:300]}\n\n"
        f"Jawaban LAMA yang perlu diperbaiki:\n\"{old_completion}\"\n\n"
        f"{style_info['user_suffix']}\n\n"
        f"Catatan perbaikan: {feedback}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": style_info["system"]},
        {"role": "user", "content": user_content},
    ]

    try:
        result = call_api(
            messages,
            model=config.generator_model,
            temperature=0.4,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        if result and len(result.strip()) >= 50:
            return result.strip()
    except Exception as e:
        print(f"  [!] API error saat memperbaiki completion: {e}")

    return None


# ============================================================================
# PIPELINE UTAMA
# ============================================================================

def main():
    config = GeneratorConfig()

    print("\n" + "=" * 70)
    print("IMPROVE RAFT DATASET COMPLETIONS")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[1] Loading dataset: {INPUT_FILE}")
    samples: List[Dict] = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"    Total sampel: {len(samples)}")

    # 2. Klasifikasi semua completion menggunakan shared validator
    print(f"\n[2] Menganalisis completion dengan validate_completion()...")
    fix_needed: List[tuple] = []  # list of (index, failed_checks)
    check_counts: Dict[str, int] = {}

    for i, s in enumerate(samples):
        completion_text = get_completion(s)
        oracle_content = extract_oracle_content_from_sample(s)
        is_valid, failed_checks = validate_completion(
            completion_text,
            s.get("instruction", ""),
            oracle_content,
            [],  # No recent completions during initial scan
        )
        if not is_valid:
            fix_needed.append((i, failed_checks))
            for chk in failed_checks:
                check_counts[chk] = check_counts.get(chk, 0) + 1

    ok_count = len(samples) - len(fix_needed)
    print(f"    ✅ Completion baik     : {ok_count}")
    print(f"    ⚠️  Perlu diperbaiki   : {len(fix_needed)}")
    print(f"\n    Breakdown masalah:")
    for chk, cnt in sorted(check_counts.items(), key=lambda x: -x[1]):
        print(f"      {chk}: {cnt}")

    if not fix_needed:
        print("\n✅ Semua completion sudah baik!")
        return

    # 3. Backup file original
    print(f"\n[3] Backup original → {BACKUP_FILE}")
    import shutil
    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    # 4. Test API connection
    print(f"\n[4] Testing API connection...")
    try:
        test = call_api(
            [{"role": "user", "content": "Katakan 'OK' dalam satu kata."}],
            model=config.generator_model,
            temperature=0.0,
            max_tokens=10,
            max_attempts=2,
        )
        print(f"    ✅ API OK: {test}")
    except Exception as e:
        print(f"    ❌ API tidak bisa dihubungi! {e}")
        sys.exit(1)

    # 5. Perbaiki completion satu per satu
    print(f"\n[5] Memperbaiki {len(fix_needed)} completion...")
    print("=" * 70)

    # Reset shared deque untuk tracking anti-monotony
    _recent_completions.clear()

    repaired = 0
    failed = 0

    pbar = tqdm(fix_needed, desc="Improving completions")
    for idx, failed_checks in pbar:
        sample = samples[idx]
        old_completion = get_completion(sample)
        instruction = sample.get("instruction", "")
        documents = get_documents(sample)
        thought_process = sample.get("thought_process", "")
        oracle_content = extract_oracle_content_from_sample(sample)

        new_completion = improve_completion(
            instruction=instruction,
            documents=documents,
            thought_process=thought_process,
            old_completion=old_completion,
            failed_checks=failed_checks,
            config=config,
        )

        if new_completion:
            # Re-validate the new completion (with anti-monotony check)
            is_valid, new_failed = validate_completion(
                new_completion,
                instruction,
                oracle_content,
                list(_recent_completions),
            )

            if is_valid:
                set_completion(sample, new_completion)
                _recent_completions.append(new_completion)
                repaired += 1
            else:
                # Still failing after retry — keep old completion
                failed += 1
        else:
            failed += 1

        pbar.set_postfix(ok=repaired, fail=failed)
        time.sleep(REQUEST_DELAY)

    pbar.close()

    # 6. Simpan hasil
    print(f"\n[6] Menyimpan hasil → {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 7. Verifikasi final
    print(f"\n[7] Verifikasi final...")
    final_issues = 0
    for s in samples:
        completion_text = get_completion(s)
        oracle_content = extract_oracle_content_from_sample(s)
        is_valid, _ = validate_completion(
            completion_text,
            s.get("instruction", ""),
            oracle_content,
            [],
        )
        if not is_valid:
            final_issues += 1

    print(f"\n{'=' * 70}")
    print(f"HASIL PERBAIKAN")
    print(f"{'=' * 70}")
    print(f"  Total sampel   : {len(samples)}")
    print(f"  Diperbaiki     : {repaired}")
    print(f"  Gagal          : {failed}")
    print(f"  Masih bermasalah: {final_issues}")
    print(f"{'=' * 70}")
    print(f"\n  📄 Output      : {OUTPUT_FILE}")
    print(f"  📦 Backup      : {BACKUP_FILE}")
    print(f"\nUntuk mengganti dataset lama:")
    print(f"  cp {OUTPUT_FILE} {INPUT_FILE}")


if __name__ == "__main__":
    main()
