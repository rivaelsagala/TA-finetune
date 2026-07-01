"""
augment_existing_raft.py
=========================
Augmentasi dataset RAFT single-oracle yang SUDAH ADA (raft_dataset_final.jsonl)
menjadi sebagian multi-oracle, TANPA generate dataset dari nol.

LATAR BELAKANG
--------------
raft_dataset_final.jsonl Anda dibuat dengan asumsi 1 pertanyaan = 1 dokumen
oracle (oracle_index = int tunggal). Akibatnya model belajar "ambil 1 dokumen
paling cocok, abaikan sisanya" — padahal banyak pertanyaan nyata jawabannya
tersebar di beberapa ayat/pasal yang berkaitan.

DARIPADA generate ulang semua dari nol (mahal: re-generate instruction,
re-pilih distraktor, re-validasi semuanya), script ini HANYA:
  1. Mengambil sample yang sudah oracle_present=True dari dataset lama.
  2. Mencari "tetangga" oracle tersebut di file chunk mentah (*_chunks.json):
     ayat lain dalam pasal yang sama, yang BELUM masuk ke documents sample itu.
  3. Jika ketemu tetangga valid -> susun ulang `documents` (oracle asli +
     1-2 tetangga + distraktor lama yang sudah ada di sample), lalu HANYA
     panggil LLM untuk re-generate `thought_process` (format per-dokumen)
     dan `completion` (sintesis semua oracle). instruction TIDAK diubah
     secara default (lihat REPHRASE_INSTRUCTION di bawah).
  4. Validasi pakai LLM-as-judge (mode multi-oracle).
  5. Sample yang TIDAK punya tetangga valid tetap dipertahankan APA ADANYA
     (tidak digenerate ulang, tidak hilang) di output gabungan.

INI BUKAN PENGGANTI raft_dataset_final.jsonl, TAPI VERSI AUGMENTED-NYA.
Output: raft_dataset_final_augmented.jsonl (gabungan sample lama + yang sudah
dimodifikasi jadi multi-oracle).

DISTRAKTOR
----------
Folder/file yang mengandung "dis" di document_id (mis. perdes_disbiru_07_2015)
DIANGGAP DISTRAKTOR MURNI dan TIDAK PERNAH dijadikan sumber oracle/tetangga,
konsisten dengan desain pipeline asli Anda. Variabel EXCLUDE_DOC_ID_PATTERN
mengatur ini.

DRY RUN
-------
Jalankan dulu dengan DRY_RUN=True (default) untuk melihat berapa sample yang
akan diaugmentasi dan PREVIEW dokumen barunya TANPA memanggil LLM sama sekali
-- supaya Anda bisa cek logikanya benar sebelum membakar kuota API.

    python augment_existing_raft.py            # dry run, gratis, tanpa API
    python augment_existing_raft.py --live      # jalankan sungguhan, panggil LLM
"""

import os, sys, json, time, random, re, argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
HF_BASE_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL = "openai/gpt-4.1-mini"
VALIDATOR_MODEL = "openai/gpt-4.1-mini"

DATASET_DIR   = PROJECT_ROOT / "data" / "dataset"
CHUNKS_DIR    = PROJECT_ROOT / "data" / "processed"   # folder berisi semua *_chunks.json (oracle pool)

INPUT_DATASET_PATH  = DATASET_DIR / "raft_dataset_final.jsonl"
OUTPUT_DATASET_PATH = DATASET_DIR / "raft_dataset_final_augmented.jsonl"

# Document_id yang mengandung salah satu pattern ini dianggap distraktor murni,
# TIDAK PERNAH dijadikan oracle/tetangga (konsisten dgn pipeline generator asli).
EXCLUDE_DOC_ID_PATTERN = re.compile(r"dis", re.IGNORECASE)

MAX_NEIGHBORS_TO_ADD     = 2     # maksimal berapa tetangga ditambahkan per sample
MAX_AUGMENT_PER_DOC      = None  # None = tanpa limit; set angka utk batasi per dokumen
REQUEST_DELAY            = 0.2
COMPLETION_STYLES = [
    "Formal (Gunakan bahasa hukum, kutip pasalnya dengan rapi dan padat).",
    "Natural (Gunakan bahasa yang lebih natural untuk warga awam, namun tetap akurat dan bersumber langsung dari dokumen).",
]

random.seed(3407)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAS
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages, temperature=0.3, max_tokens=1024, retries=3, model=GENERATOR_MODEL) -> Optional[str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": 0.9, "stream": False}
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError:
            time.sleep((attempt + 1) * 5)
        except Exception:
            time.sleep(3)
    return None


def clean_content(text: str) -> str:
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()


def get_doc_label(chunk: Dict) -> str:
    title = chunk.get("metadata", {}).get("title", "Dokumen tidak diketahui")
    pasal = chunk.get("pasal", "") or chunk.get("metadata", {}).get("section", "")
    return f"{title}, {pasal.title()}" if pasal else title


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    if not path.exists():
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_all_chunks(directory: Path) -> Dict[str, List[Dict]]:
    """
    Load semua *_chunks.json, kembalikan dict {document_id: [chunk, ...]}.
    Dokumen yang document_id-nya match EXCLUDE_DOC_ID_PATTERN (distraktor)
    DILEWATI -- tidak pernah jadi sumber oracle/tetangga.
    """
    chunk_index: Dict[str, List[Dict]] = {}
    if not directory.exists():
        print(f"PERINGATAN: folder {directory} tidak ditemukan.")
        return chunk_index
    for fpath in directory.glob("*_chunks.json"):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            continue
        doc_id = chunks[0]["metadata"]["document_id"]
        if EXCLUDE_DOC_ID_PATTERN.search(doc_id):
            continue  # distraktor murni, skip
        chunk_index[doc_id] = chunks
    return chunk_index


def find_oracle_chunk(oracle_text_clean: str, doc_id: str,
                       chunk_index: Dict[str, List[Dict]]) -> Optional[Dict]:
    """
    Cocokkan teks oracle dari sample lama ke chunk mentah aslinya.
    Pakai exact match dulu, fallback ke prefix-match (toleransi artefak
    pemotongan teks minor yang kadang terjadi di dataset lama).
    """
    if doc_id not in chunk_index:
        return None
    for c in chunk_index[doc_id]:
        cc = clean_content(c["content"])
        if cc == oracle_text_clean:
            return c
    # fallback: prefix match (toleransi beda di ekor teks)
    for c in chunk_index[doc_id]:
        cc = clean_content(c["content"])
        prefix_len = min(100, len(cc), len(oracle_text_clean))
        if prefix_len >= 30 and cc[:prefix_len] == oracle_text_clean[:prefix_len]:
            return c
    return None


def build_pasal_index(chunk_index: Dict[str, List[Dict]]) -> Dict[Tuple[str, str], List[Dict]]:
    pasal_index = defaultdict(list)
    for doc_id, chunks in chunk_index.items():
        for c in chunks:
            pasal_index[(doc_id, c.get("pasal", ""))].append(c)
    return pasal_index


def find_neighbor_chunks(oracle_chunk: Dict, doc_id: str,
                          pasal_index: Dict[Tuple[str, str], List[Dict]],
                          existing_texts_clean: set, max_n: int) -> List[Dict]:
    """
    Cari ayat tetangga dalam pasal yang sama dengan oracle_chunk, yang BELUM
    ada di documents sample (existing_texts_clean), maksimal max_n chunk.
    """
    pasal = oracle_chunk.get("pasal", "")
    if not pasal:
        return []
    siblings = pasal_index.get((doc_id, pasal), [])
    oracle_text_clean = clean_content(oracle_chunk["content"])

    candidates = []
    for c in siblings:
        cc = clean_content(c["content"])
        if cc == oracle_text_clean:
            continue
        if cc in existing_texts_clean:
            continue
        candidates.append(c)

    random.shuffle(candidates)
    return candidates[:max_n]


# ─────────────────────────────────────────────────────────────────────────────
# RE-GENERATE thought_process + completion (MULTI-ORACLE)
# ─────────────────────────────────────────────────────────────────────────────

def generate_multi_thought_and_completion(
    question: str, docs: List[str], doc_labels: List[str],
    oracle_indices_0based: List[int], style: str,
) -> Tuple[Optional[str], Optional[str]]:
    docs_fmt = "".join(
        f"\n--- Dokumen {i+1}: {l} ---\n{d}\n" for i, (d, l) in enumerate(zip(docs, doc_labels))
    )
    oracle_nums = [i + 1 for i in oracle_indices_0based]

    hint = (
        f"\n[INTERNAL] Dokumen {', '.join(map(str, oracle_nums))} SEMUA mengandung "
        "bagian dari jawaban (jawaban harus menggabungkan SEMUA dokumen ini)."
    )

    t_instr = (
        "Buat 'thought_process' yang WAJIB mengevaluasi SETIAP dokumen satu per satu, "
        "satu baris per dokumen, format persis:\n"
        '"Dokumen <n>: relevan, <alasan singkat>." atau "Dokumen <n>: tidak relevan, <alasan singkat>."\n'
        f"Total ada {len(docs)} dokumen, jadi harus ada {len(docs)} baris evaluasi, "
        "berurutan dari Dokumen 1 sampai Dokumen terakhir. "
        f"Dokumen {', '.join(map(str, oracle_nums))} HARUS dievaluasi relevan; sisanya tidak relevan."
    )

    c_instr = (
        f"Gaya: {style}\n"
        f"Jawab dengan MENYINTESIS informasi dari SEMUA dokumen relevan "
        f"(Dokumen {', '.join(map(str, oracle_nums))}) menjadi SATU jawaban utuh dan koheren. "
        "JANGAN hanya mengutip satu dokumen saja — gabungkan semua poin dari semua dokumen relevan. "
        "DILARANG KERAS menambahkan opini atau fakta yang tidak ada di dokumen-dokumen tersebut."
    )

    sys_msg = (
        "Anda AI pembuat data RAFT, mode MULTI-ORACLE (lebih dari satu dokumen relevan).\n"
        "Output JSON valid:\n"
        '{"thought_process": "...", "completion": "..."}\n'
        f"ATURAN THOUGHT:\n{t_instr}\nATURAN COMPLETION:\n{c_instr}"
    )

    res = call_llm(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Q: {question}\n{docs_fmt}\n{hint}"}],
        temperature=0.3, max_tokens=1024,
    )
    if not res:
        return None, None
    for attempt in [res, re.search(r'\{[\s\S]*\}', res)]:
        try:
            raw = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
            if not raw:
                continue
            data = json.loads(raw)
            return data.get("thought_process", "").strip(), data.get("completion", "").strip()
        except Exception:
            continue
    return None, None


def validate_multi_sample(sample: Dict) -> Optional[Dict]:
    oracle_nums = [i + 1 for i in sample["metadata_extra"]["oracle_index"]]
    sys_msg = (
        "Anda adalah evaluator ketat dataset RAFT MULTI-ORACLE (>1 dokumen relevan). "
        "Validasi sampel berikut:\n"
        "1. instruction_answered: true HANYA JIKA completion menjawab instruction secara lengkap.\n"
        "2. grounded: true HANYA JIKA seluruh isi completion berasal murni dari dokumen.\n"
        "3. uses_all_oracles: true HANYA JIKA completion menggabungkan informasi dari SEMUA "
        f"dokumen oracle ({', '.join(map(str, oracle_nums))}), bukan cuma salah satu saja.\n"
        "4. thought_correct: true JIKA thought_process punya satu baris evaluasi PER dokumen "
        "dan menandai dokumen oracle relevan, sisanya tidak relevan.\n"
        "Output JSON:\n"
        '{"pass": boolean, "instruction_answered": boolean, "grounded": boolean, '
        '"uses_all_oracles": boolean, "thought_correct": boolean, "score": float, "reason": "..."}'
    )
    user_msg = (
        f"Instruction: {sample['instruction']}\n"
        f"Documents: {json.dumps(sample['documents'], ensure_ascii=False)}\n"
        f"Thought Process: {sample['thought_process']}\n"
        f"Completion: {sample['completion']}\n"
        f"Oracle indices (1-based): {oracle_nums}"
    )
    res = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                    temperature=0.0, max_tokens=256, model=VALIDATOR_MODEL)
    if not res:
        return None
    try:
        match = re.search(r'\{[\s\S]*\}', res)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASI PER-SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def try_augment_sample(
    sample: Dict, chunk_index: Dict[str, List[Dict]],
    pasal_index: Dict[Tuple[str, str], List[Dict]],
    dry_run: bool = True,
) -> Tuple[Optional[Dict], str]:
    """
    Coba augmentasi 1 sample lama menjadi multi-oracle.

    Returns: (sample_baru_atau_None, status_string)
    status_string salah satu dari:
      "no_oracle"        - sample ini oracle_present=False, tidak diaugmentasi
      "oracle_not_found" - oracle text tidak match chunk mentah manapun
      "no_neighbors"     - tidak ada tetangga valid (pasal cuma 1 ayat / semua sudah dipakai)
      "augmented"        - berhasil disusun ulang documents (dry_run: belum panggil LLM)
      "llm_failed"       - LLM gagal generate / tidak lolos validasi (hanya saat live run)
    """
    meta = sample.get("metadata_extra", {})
    if not meta.get("oracle_present"):
        return None, "no_oracle"

    oracle_idx = meta["oracle_index"]
    oracle_doc_id = meta["oracle_doc_id"]
    oracle_text_clean = clean_content(sample["documents"][oracle_idx])

    oracle_chunk = find_oracle_chunk(oracle_text_clean, oracle_doc_id, chunk_index)
    if not oracle_chunk:
        return None, "oracle_not_found"

    existing_texts_clean = set(clean_content(d) for d in sample["documents"])
    neighbors = find_neighbor_chunks(
        oracle_chunk, oracle_doc_id, pasal_index, existing_texts_clean, MAX_NEIGHBORS_TO_ADD
    )
    if not neighbors:
        return None, "no_neighbors"

    # Susun ulang documents: oracle asli (label) + tetangga + SISA distraktor lama
    # yang sudah ada di sample (kecuali oracle lama, karena posisi oracle berubah).
    # PENTING: dataset lama Anda menyimpan documents yang SUDAH dibersihkan dari
    # header [dokumen: ...] [desa: ...] dst (lihat clean_content() di generator
    # asli). Chunk mentah dari folder processed/ BELUM dibersihkan, jadi kita
    # harus clean_content() teks oracle baru & tetangga supaya formatnya konsisten
    # dengan distraktor lama yang sudah bersih.
    old_oracle_text = clean_content(sample["documents"][oracle_idx])
    old_distractors = [d for i, d in enumerate(sample["documents"]) if i != oracle_idx]

    neighbor_texts = [clean_content(n["content"]) for n in neighbors]
    neighbor_labels = [get_doc_label(n) for n in neighbors]

    oracle_label = get_doc_label(oracle_chunk)

    # gabung & acak posisi: oracle + tetangga jadi oracle group, sisanya distraktor
    oracle_group_texts = [old_oracle_text] + neighbor_texts
    oracle_group_labels = [oracle_label] + neighbor_labels

    distractor_labels = [f"Distraktor {i+1}" for i in range(len(old_distractors))]  # label generik, tidak krusial

    combined = [(t, l, True) for t, l in zip(oracle_group_texts, oracle_group_labels)] + \
               [(t, l, False) for t, l in zip(old_distractors, distractor_labels)]
    random.shuffle(combined)

    new_documents = [c[0] for c in combined]
    new_labels = [c[1] for c in combined]
    new_oracle_indices = [i for i, c in enumerate(combined) if c[2]]

    new_sample_skeleton = {
        "instruction": sample["instruction"],   # instruction TIDAK diubah secara default
        "documents": new_documents,
        "thought_process": None,   # akan diisi LLM
        "completion": None,        # akan diisi LLM
        "metadata_extra": {
            "question_type": meta.get("question_type"),
            "answer_type": "abstractive",
            "oracle_present": True,
            "oracle_doc_id": oracle_doc_id,
            "oracle_index": new_oracle_indices,   # LIST -- ini perbedaan kunci dgn skema lama
            "multi_oracle": True,
            "augmented_from_single_oracle": True,
            "num_neighbors_added": len(neighbors),
        },
    }

    if dry_run:
        # Dry run: tidak panggil LLM, cukup kembalikan skeleton + info preview
        new_sample_skeleton["thought_process"] = "[DRY_RUN - belum digenerate]"
        new_sample_skeleton["completion"] = "[DRY_RUN - belum digenerate]"
        return new_sample_skeleton, "augmented"

    style = random.choice(COMPLETION_STYLES)
    thought, completion = generate_multi_thought_and_completion(
        new_sample_skeleton["instruction"], new_documents, new_labels,
        new_oracle_indices, style,
    )
    time.sleep(REQUEST_DELAY)

    if not thought or not completion:
        return None, "llm_failed"

    new_sample_skeleton["thought_process"] = thought
    new_sample_skeleton["completion"] = completion

    validation_res = validate_multi_sample(new_sample_skeleton)
    time.sleep(REQUEST_DELAY)

    if not (
        validation_res
        and validation_res.get("pass", False)
        and validation_res.get("instruction_answered", False)
        and validation_res.get("grounded", False)
        and validation_res.get("uses_all_oracles", False)
        and validation_res.get("thought_correct", False)
    ):
        return None, "llm_failed"

    new_sample_skeleton["validation"] = validation_res
    return new_sample_skeleton, "augmented"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(dry_run: bool = True, limit: Optional[int] = None):
    print(f"Mode: {'DRY RUN (tanpa panggil LLM)' if dry_run else 'LIVE (memanggil LLM sungguhan)'}")
    print(f"Memuat dataset lama dari: {INPUT_DATASET_PATH}")
    old_samples = load_jsonl(INPUT_DATASET_PATH)
    print(f"Total sample lama: {len(old_samples)}")

    print(f"Memuat chunk mentah dari: {CHUNKS_DIR}")
    chunk_index = load_all_chunks(CHUNKS_DIR)
    print(f"Dokumen non-distraktor termuat: {len(chunk_index)}")
    pasal_index = build_pasal_index(chunk_index)

    status_counter = defaultdict(int)
    augmented_samples = []
    untouched_samples = []

    candidates = old_samples if limit is None else old_samples[:limit]

    for sample in tqdm(candidates, desc="Mencoba augmentasi"):
        new_sample, status = try_augment_sample(sample, chunk_index, pasal_index, dry_run=dry_run)
        status_counter[status] += 1
        if new_sample is not None:
            augmented_samples.append(new_sample)
        else:
            untouched_samples.append(sample)

    print("\n=== RINGKASAN ===")
    for status, count in status_counter.items():
        print(f"  {status}: {count}")
    print(f"\nTotal berhasil diaugmentasi: {len(augmented_samples)}")
    print(f"Total dipertahankan apa adanya: {len(untouched_samples)}")

    if dry_run:
        print("\n[DRY RUN] Contoh 2 sample yang BERHASIL diaugmentasi (preview, blm ada thought/completion asli):")
        for s in augmented_samples[:2]:
            print("-" * 80)
            print("Instruction:", s["instruction"])
            print("Jumlah dokumen baru:", len(s["documents"]))
            print("Oracle index (1-based):", [i + 1 for i in s["metadata_extra"]["oracle_index"]])
            for i, doc in enumerate(s["documents"], 1):
                tag = "ORACLE" if (i - 1) in s["metadata_extra"]["oracle_index"] else "distraktor"
                print(f"  [{tag}] Dokumen {i}: {doc[:100]}...")
        print("\nJalankan ulang dengan --live untuk benar-benar generate thought_process & completion via LLM.")
        return

    # gabungkan: sample yang berhasil diaugmentasi (sekarang multi-oracle) + sample lama yang tidak disentuh
    final_dataset = untouched_samples + augmented_samples

    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDataset augmented tersimpan di: {OUTPUT_DATASET_PATH}")
    print(f"Total sample akhir: {len(final_dataset)} "
          f"({len(augmented_samples)} multi-oracle baru + {len(untouched_samples)} sample lama apa adanya)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Jalankan sungguhan (panggil LLM). Default: dry run.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah sample yang diproses (utk testing cepat).")
    args = parser.parse_args()

    run(dry_run=not args.live, limit=args.limit)