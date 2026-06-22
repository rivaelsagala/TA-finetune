"""
=============================================================================
Improve RAFT Dataset Completions (Tanpa Regenerate Ulang)
=============================================================================

Script ini memperbaiki field `completion` pada dataset RAFT yang sudah ada
TANPA perlu men-generate ulang dataset dari awal.

Masalah yang diperbaiki:
  1. Completion yang mengandung referensi "Dokumen N" / "(Dokumen N)"
     → Referensi ini tidak bermakna bagi end-user
  2. Completion yang terlalu pendek (< 80 karakter)
     → Diperluas menjadi jawaban yang lengkap dan deskriptif
  3. Completion yang dimulai dengan "Dokumen N menyebutkan..."
     → Diubah menjadi jawaban langsung

Cara kerja:
  - Membaca raft_perdes_dataset.jsonl yang ada
  - Untuk setiap completion bermasalah, kirim ke AI API (OpenAI-compatible)
    dengan konteks instruction + documents + thought_process
  - AI menulis ulang completion yang lebih baik
  - Simpan hasil ke file baru (original tidak dihapus)

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
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

# ============================================================================
# KONFIGURASI
# ============================================================================

# Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "raft_perdes_dataset.jsonl"
OUTPUT_FILE = DATA_DIR / "raft_perdes_dataset_improved.jsonl"
BACKUP_FILE = DATA_DIR / "raft_perdes_dataset_backup.jsonl"

# API Configuration (dari .env)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL = "openai/gpt-4o-mini"
REQUEST_DELAY = 1.5  # detik antara request
MAX_RETRIES = 3

print(f"API URL: {API_URL}")
print(f"Model  : {GENERATOR_MODEL}")
print(f"Input  : {INPUT_FILE}")
print(f"Output : {OUTPUT_FILE}")


# ============================================================================
# API Client
# ============================================================================

import requests

def call_api(
    messages: List[Dict[str, str]],
    model: str = GENERATOR_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.4,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Panggil OpenAI-compatible API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError:
            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  Rate limited. Menunggu {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 503:
                wait = (attempt + 1) * 10
                print(f"  Model loading. Menunggu {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(3)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


# ============================================================================
# Deteksi Masalah Completion
# ============================================================================

def classify_completion(completion: str) -> Tuple[bool, List[str]]:
    """
    Klasifikasi apakah completion perlu diperbaiki.
    Returns: (needs_fix, list_of_issues)
    """
    issues = []
    c = completion.strip()

    if not c:
        issues.append("EMPTY")
        return True, issues

    # Terlalu pendek
    if len(c) < 80:
        issues.append("TOO_SHORT")

    # Dimulai dengan "Dokumen N menyebutkan/menjelaskan..."
    if re.match(r'^Dokumen\s+\d+\s+(menyebutkan|menjelaskan|menyatakan|berisi)', c, re.IGNORECASE):
        issues.append("STARTS_WITH_DOK_REF")

    # Hanya berisi "Dokumen N" atau "Jawaban merujuk pada Dokumen N"
    if re.match(r'^(Jawaban\s+)?(akhir\s+)?(merujuk|mengacu|berdasarkan|sesuai)?\s*(pada\s+)?Dokumen\s+\d+\.?$', c, re.IGNORECASE):
        issues.append("ONLY_DOK_REF")

    # Mengandung referensi "(Dokumen N)" di tengah/akhir
    if re.search(r'\(Dokumen\s+\d+\)', c):
        issues.append("HAS_DOK_PAREN_REF")

    # Mengandung "Dokumen N" sebagai referensi inline
    # (bukan di awal kalimat sebagai subjek)
    if re.search(r',\s*sebagaimana\s+(dijelaskan|tercantum|diatur)\s+dalam\s+Dokumen\s+\d+', c, re.IGNORECASE):
        issues.append("HAS_DOK_INLINE_REF")

    # Ending dengan "Dokumen N." atau "(Dokumen N)."
    if re.search(r'Dokumen\s+\d+\.?\s*$', c):
        issues.append("ENDS_WITH_DOK_REF")

    needs_fix = len(issues) > 0
    return needs_fix, issues


# ============================================================================
# Perbaiki Completion Menggunakan AI
# ============================================================================

IMPROVE_SYSTEM_PROMPT = """\
Anda adalah editor profesional dataset fine-tuning AI untuk domain peraturan desa Indonesia.

TUGAS: Tulis ulang jawaban di bawah ini agar menjadi jawaban yang SEMPURNA untuk fine-tuning.

ATURAN KETAT:
1. JANGAN menyebutkan "Dokumen 1", "Dokumen 2", "Dokumen 3" atau variasi apapun.
   - HAPUS semua referensi seperti "(Dokumen 1)", "sebagaimana dijelaskan dalam Dokumen 2",
     "Dokumen 3 menyebutkan", "merujuk pada Dokumen 1", dll.
   - Ganti dengan referensi ke PASAL/PERATURAN yang sebenarnya jika tersedia.
     Contoh: "sebagaimana diatur dalam Pasal 5" atau "sesuai ketentuan Peraturan Desa..."
2. Jawaban HARUS menjawab pertanyaan secara LANGSUNG dan LENGKAP.
3. Jawaban harus bisa BERDIRI SENDIRI tanpa perlu konteks tambahan.
4. Sertakan FAKTA SPESIFIK: definisi, angka, syarat, nama jabatan dari dokumen sumber.
5. Gunakan Bahasa Indonesia yang formal, jelas, dan informatif.
6. Panjang ideal: 2-4 kalimat (80-300 karakter).
7. JANGAN menambah informasi yang TIDAK ADA di dokumen sumber.
8. JANGAN mengawali dengan "Berdasarkan dokumen..." atau "Menurut dokumen..."

CONTOH PERBAIKAN:
- SEBELUM: "Bayi baru lahir adalah anak usia 0 hari sampai 20 hari (Dokumen 1)."
- SESUDAH: "Bayi baru lahir atau neonatal adalah anak usia 0 hari sampai dengan 20 hari, sesuai definisi dalam Pasal 1 Peraturan Desa Biru No. 07 Tahun 2015."

- SEBELUM: "Dokumen 3 menyatakan bahwa penggantian anggota tim Kibbla berdasarkan hasil MMD."
- SESUDAH: "Penggantian anggota tim Kibbla dilakukan berdasarkan hasil Musyawarah Masyarakat Desa (MMD), sesuai ketentuan Pasal 13 Peraturan Desa."

- SEBELUM: "Masa bakti kepengurusan tim Kibbla adalah 3 (tiga) tahun (Dokumen 3)."
- SESUDAH: "Masa bakti kepengurusan tim Kibbla adalah 3 (tiga) tahun. Hal ini menunjukkan bahwa tim Kibbla memiliki periode waktu yang jelas untuk menjalankan tugas dan tanggung jawabnya dalam pengelolaan kesehatan ibu, bayi, dan anak di desa."

Output HANYA jawaban yang sudah diperbaiki, tanpa penjelasan tambahan."""


def improve_completion(
    instruction: str,
    documents: List[str],
    thought_process: str,
    old_completion: str,
    issues: List[str],
) -> Optional[str]:
    """Perbaiki satu completion menggunakan AI."""

    # Bangun konteks dokumen (untuk AI tahu fakta yang benar)
    docs_context = ""
    for i, doc in enumerate(documents, 1):
        docs_context += f"\n--- Dokumen sumber {i} ---\n{doc[:600]}\n"

    # Identifikasi pasal dari dokumen
    pasal_info = ""
    for doc in documents:
        pasal_match = re.search(r'pasal\s+\d+', doc, re.IGNORECASE)
        if pasal_match:
            pasal_info += f"  - {pasal_match.group()}\n"

    issue_desc = ", ".join(issues)

    user_prompt = f"""Pertanyaan: {instruction}

Dokumen sumber (untuk referensi fakta, JANGAN sebutkan "Dokumen 1/2/3"):
{docs_context}

Pasal-pasal yang tersedia:
{pasal_info if pasal_info else "  (tidak teridentifikasi)"}

Analisis (untuk konteks): {thought_process[:300]}

Jawaban LAMA yang perlu diperbaiki:
"{old_completion}"

Masalah: {issue_desc}

Tulis ulang jawaban di atas sesuai aturan. Output HANYA jawaban baru:"""

    messages = [
        {"role": "system", "content": IMPROVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = call_api(messages, temperature=0.3, max_tokens=400)

    if result:
        # Bersihkan hasil
        cleaned = result.strip().strip('"').strip("'").strip()
        # Pastikan tidak masih ada referensi Dokumen N
        cleaned = re.sub(r'\s*\(Dokumen\s+\d+\)\s*', ' ', cleaned).strip()
        cleaned = re.sub(r'\s*Dokumen\s+\d+\.?\s*$', '', cleaned).strip()
        # Pastikan tidak terlalu pendek
        if len(cleaned) >= 50:
            return cleaned

    return None


# ============================================================================
# PIPELINE UTAMA
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("IMPROVE RAFT DATASET COMPLETIONS")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[1] Loading dataset: {INPUT_FILE}")
    samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"    Total sampel: {len(samples)}")

    # 2. Klasifikasi semua completion
    print(f"\n[2] Menganalisis completion...")
    fix_needed = []  # list of (index, issues)
    issue_counts = {}

    for i, s in enumerate(samples):
        needs_fix, issues = classify_completion(s["completion"])
        if needs_fix:
            fix_needed.append((i, issues))
            for iss in issues:
                issue_counts[iss] = issue_counts.get(iss, 0) + 1

    ok_count = len(samples) - len(fix_needed)
    print(f"    ✅ Completion baik     : {ok_count}")
    print(f"    ⚠️  Perlu diperbaiki   : {len(fix_needed)}")
    print(f"\n    Breakdown masalah:")
    for iss, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"      {iss}: {cnt}")

    if not fix_needed:
        print("\n✅ Semua completion sudah baik!")
        return

    # 3. Backup file original
    print(f"\n[3] Backup original → {BACKUP_FILE}")
    import shutil
    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    # 4. Test API connection
    print(f"\n[4] Testing API connection...")
    test = call_api(
        [{"role": "user", "content": "Katakan 'OK' dalam satu kata."}],
        max_tokens=10, temperature=0.1
    )
    if not test:
        print("    ❌ API tidak bisa dihubungi! Periksa OPENAI_API_KEY dan OPENAI_BASE_URL di .env")
        sys.exit(1)
    print(f"    ✅ API OK: {test}")

    # 5. Perbaiki completion satu per satu
    print(f"\n[5] Memperbaiki {len(fix_needed)} completion...")
    print("=" * 70)

    repaired = 0
    failed = 0
    skipped = 0

    pbar = tqdm(fix_needed, desc="Improving completions")
    for idx, issues in pbar:
        sample = samples[idx]
        old = sample["completion"]

        new_completion = improve_completion(
            instruction=sample["instruction"],
            documents=sample["documents"],
            thought_process=sample["thought_process"],
            old_completion=old,
            issues=issues,
        )

        if new_completion:
            # Verifikasi: pastikan tidak masih ada "Dokumen N"
            still_has_ref = bool(re.search(r'Dokumen\s+\d+', new_completion))
            if still_has_ref:
                # Strip referensi secara paksa
                new_completion = re.sub(r'\s*\(Dokumen\s+\d+\)\s*', ' ', new_completion)
                new_completion = re.sub(r',?\s*sebagaimana\s+(dijelaskan|tercantum|diatur)\s+dalam\s+Dokumen\s+\d+\.?', '.', new_completion)
                new_completion = re.sub(r',?\s*sesuai\s+dengan\s+(ketentuan\s+)?Dokumen\s+\d+\.?', '.', new_completion)
                new_completion = re.sub(r',?\s*merujuk\s+pada\s+Dokumen\s+\d+\.?', '.', new_completion)
                new_completion = re.sub(r'\s*Dokumen\s+\d+\s*', ' ', new_completion)
                new_completion = re.sub(r'\s{2,}', ' ', new_completion).strip()
                # Pastikan diakhiri titik
                if new_completion and not new_completion.endswith('.'):
                    new_completion += '.'

            samples[idx]["completion"] = new_completion
            repaired += 1
            pbar.set_postfix(ok=repaired, fail=failed)
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
        needs_fix, _ = classify_completion(s["completion"])
        if needs_fix:
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
