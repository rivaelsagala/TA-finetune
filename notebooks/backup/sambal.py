"""
generate_raft_dataset.py
========================
Generate RAFT (Retrieval Augmented Fine-Tuning) dataset dari chunk dokumen
peraturan desa, sesuai metodologi Zhang et al. (2024).

Referensi:
  Zhang, T., Shao, F., Garg, S., et al. (2024). "RAFT: Adapting Language
  Model to Domain Specific RAG." Transactions on ML Research.
  https://arxiv.org/abs/2403.10131

Output: JSONL file di data/ dengan format RAFT standar.

Cara pakai:
  cd notebooks/
  python generate_raft_dataset.py
"""

import json
import random
import os
from datetime import datetime
from collections import Counter

# ============================================================================
# Konfigurasi
# ============================================================================
random.seed(42)

# Path
CHUNKS_PATH = "../data/PERATURAN_DESA_LOA_KECAMATAN_PASEH_KABUPATEN_BANDUNG_NOMOR___5_TAHUN_2017__chunks.json"
OUTPUT_PATH = "../data/raft_dataset_loa_5_2017.jsonl"

# RAFT hyperparameters (Zhang et al., 2024)
P_ORACLE = 0.80          # 80% sampel memiliki oracle document
N_DISTRACTORS = 3        # Jumlah distractor per sampel (K=3)
MIN_CONTENT_CHARS = 120  # Minimum karakter content untuk layak jadi oracle

# Metadata peraturan
TITLE = "Peraturan Desa Loa No. 5 Tahun 2017 - Rencana Kerja Pembangunan Desa (Rkp-Desa) Tahun Anggaran 2018"
VILLAGE = "loa"
REGENCY = "bandung"
PERDES_NUMBER = "5"
PERDES_YEAR = "2017"

# System prompt (harus sama persis dengan yang dipakai saat inference)
SYSTEM_PROMPT_ORACLE = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Kabupaten Bandung.\n\n"
    "Di bawah ini terdapat beberapa potongan dokumen peraturan desa. "
    "Tidak semua dokumen relevan dengan pertanyaan. Baca seluruh isi dokumen, "
    "lalu gunakan HANYA informasi dari dokumen yang relevan untuk menjawab "
    "pertanyaan."
)

SYSTEM_PROMPT_NO_ORACLE = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Kabupaten Bandung.\n\n"
    "Di bawah ini terdapat beberapa potongan dokumen peraturan desa. "
    "PERHATIAN: tidak ada dokumen yang relevan dengan pertanyaan. "
    "Jika isi dokumen tidak menjawab pertanyaan, katakan secara jujur "
    "bahwa informasi tersebut tidak ditemukan."
)

# Jawaban untuk no-oracle samples
NO_ORACLE_COT = (
    "##Reason: Tidak ada dokumen relevan di antara konteks yang diberikan.\n"
    "##Answer: Informasi yang relevan untuk menjawab pertanyaan ini tidak "
    "ditemukan dalam dokumen yang tersedia. Disarankan untuk merujuk langsung "
    "pada dokumen Peraturan Desa terkait."
)

NO_ORACLE_ANSWER = (
    "Informasi yang relevan untuk menjawab pertanyaan ini tidak "
    "ditemukan dalam dokumen yang tersedia. Disarankan untuk merujuk langsung "
    "pada dokumen Peraturan Desa terkait."
)


# ============================================================================
# Helper functions
# ============================================================================

def load_chunks(path: str) -> list:
    """Load chunks dari file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_clean_text(chunk: dict) -> str:
    """
    Ambil full content chunk (termasuk header metadata).
    Ini adalah format yang dihasilkan retriever RAG saat inference.
    Format: [dokumen: ...] [desa: ...] [kabupaten: ...] [nomor: ...]\n\n{content}
    """
    return chunk["content"].strip()


def get_pasal(chunk: dict) -> str:
    """Ambil nama pasal dari metadata."""
    return chunk["metadata"].get("section", chunk.get("pasal", "pasal 1"))


def get_pasal_title(chunk: dict) -> str:
    """Ambil judul pasal dari content (baris pertama setelah header)."""
    text = get_clean_text(chunk)
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("pasal"):
            return line
    return get_pasal(chunk)


def format_doc_block(chunk: dict) -> str:
    """
    Format satu dokumen dalam konteks block.
    Menggunakan raw content dari chunk (persis output retriever RAG).
    TANPA label GOLD/DISTRACTOR.
    """
    return get_clean_text(chunk)


def build_context_block(docs: list) -> str:
    """
    Bangun blok === DOKUMEN KONTEKS === dari list raw content string.
    """
    parts = ["=== DOKUMEN KONTEKS ===\n"]
    for i, doc in enumerate(docs, 1):
        parts.append(f"[DOKUMEN {i}]\n{doc}")
        if i < len(docs):
            parts.append("\n---\n")
    parts.append("\n=== AKHIR DOKUMEN KONTEKS ===")
    return "\n".join(parts)


# ============================================================================
# Question generation
# ============================================================================

# Template pertanyaan - divariasikan agar model robust terhadap berbagai formulasi
QUESTION_TEMPLATES = [
    "Apa isi dari {pasal} dalam {title}?",
    "Bagaimana bunyi {pasal} dari {title}?",
    "Jelaskan isi {pasal} pada {title}",
    "Tolong berikan penjelasan tentang {pasal} di {title}",
    "Apa yang diatur dalam {pasal} {title}?",
    "Apa bunyi {pasal} dari {title}?",
    "Bagaimana ketentuan {pasal} dalam {title}?",
    "Apa maksud dari {pasal} yang tercantum dalam {title}?",
    "Di Desa Loa, apa isi {pasal} dalam {title}?",
    "Apa saja poin-poin dalam {pasal} {title}?",
]

# Template pertanyaan spesifik untuk content definisi (pasal 1)
DEFINITION_TEMPLATES = [
    "Menurut {title}, apa yang dimaksud dengan {term}?",
    "Apa definisi {term} dalam {title}?",
    "Bagaimana {title} mendefinisikan {term}?",
    "Dalam {title}, apa pengertian dari {term}?",
]

# Template pertanyaan untuk list items (pasal 6 sub-items)
LIST_ITEM_TEMPLATES = [
    "Apa saja contoh {category} dalam {title}?",
    "Sebutkan bidang {category} menurut {title}",
    "Dalam {title}, apa saja yang termasuk {category}?",
]


def extract_definition_term(text: str) -> str:
    """
    Untuk content definisi (pasal 1), ekstrak istilah yang didefinisikan.
    Contoh: '9. pembangunan desa adalah upaya...' -> 'pembangunan desa'
    """
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        # Skip baris 'pasal X'
        if line.lower().startswith("pasal"):
            continue
        # Cari pola: "N. istilah adalah ..."
        if " adalah " in line:
            # Hapus nomor di depan
            parts = line.split(".", 1)
            if len(parts) > 1:
                term_part = parts[1].strip()
                term = term_part.split(" adalah ")[0].strip()
                if len(term) > 3:
                    return term
    return ""


def get_category_for_pasal6(chunk: dict) -> str:
    """Tentukan kategori untuk chunk pasal 6 berdasarkan content."""
    text = get_clean_text(chunk).lower()
    if "infrastruktur" in text or "jalan" in text or "lingkungan" in text or "tambatan" in text:
        return "pembangunan infrastruktur desa"
    elif "kesehatan" in text or "posyandu" in text or "sanitasi" in text or "air bersih" in text:
        return "sarana dan prasarana kesehatan"
    elif "pendidikan" in text or "taman bacaan" in text or "pelatihan" in text or "seni" in text:
        return "sarana dan prasarana pendidikan dan kebudayaan"
    elif "pasar" in text or "bum desa" in text or "padi" in text or "ternak" in text or "ekonomi" in text:
        return "pengembangan usaha ekonomi produktif"
    elif "hijau" in text or "hutan" in text or "sungai" in text or "lingkungan hidup" in text:
        return "pelestarian lingkungan hidup"
    elif "pemberdayaan" in text or "kader" in text or "kelompok" in text or "perempuan" in text:
        return "pemberdayaan masyarakat"
    elif "pembinaan" in text or "ketentraman" in text or "kerukunan" in text:
        return "pembinaan kemasyarakatan"
    else:
        return "pembangunan desa"


def generate_question(chunk: dict) -> str:
    """Generate pertanyaan berdasarkan tipe dan content chunk."""
    pasal = get_pasal(chunk)
    text = get_clean_text(chunk)
    content_len = len(text)

    # Untuk content definisi pendek dengan term yang jelas
    if pasal == "pasal 1" and content_len < 400:
        term = extract_definition_term(text)
        if term:
            template = random.choice(DEFINITION_TEMPLATES)
            return template.format(title=TITLE, term=term)

    # Untuk list items di pasal 6 (content pendek)
    if pasal == "pasal 6" and content_len < 200:
        category = get_category_for_pasal6(chunk)
        template = random.choice(LIST_ITEM_TEMPLATES)
        return template.format(title=TITLE, category=category)

    # Default: template umum
    template = random.choice(QUESTION_TEMPLATES)
    return template.format(pasal=pasal, title=TITLE)


# ============================================================================
# CoT Answer generation (synthesized, NOT raw document dump)
# ============================================================================

def _strip_metadata_header(text: str) -> str:
    """
    Hapus baris metadata header '[dokumen: ...] [desa: ...] ...' dari awal text.
    Kembalikan hanya isi konten pasal.
    """
    lines = text.split("\n")
    content_lines = []
    past_header = False
    for line in lines:
        if not past_header and line.strip().startswith("[dokumen:"):
            continue  # skip metadata header line
        if not past_header and line.strip() == "":
            continue  # skip blank lines before content
        past_header = True
        content_lines.append(line)
    return "\n".join(content_lines).strip()


def _extract_definitions(text: str) -> list:
    """
    Ekstrak definisi dari konten pasal 1.
    Return list of (number, term, definition) tuples.
    Contoh: '1. desa adalah kesatuan masyarakat...' -> ('1', 'desa', 'kesatuan masyarakat...')
    """
    content = _strip_metadata_header(text)
    definitions = []
    current_def = ""
    current_num = ""
    current_term = ""

    for line in content.split("\n"):
        line_stripped = line.strip()
        # Skip pasal header
        if line_stripped.lower().startswith("pasal"):
            continue
        # Skip intro line like 'dalam peraturan desa ini yang dimaksud dengan :'
        if "yang dimaksud dengan" in line_stripped:
            continue
        # Detect new definition: starts with 'N. term adalah ...'
        import re
        match = re.match(r'^(\d+)\.\s+(.+?)\s+adalah\s+(.*)', line_stripped)
        if match:
            # Save previous definition if any
            if current_num:
                definitions.append((current_num, current_term, current_def.strip()))
            current_num = match.group(1)
            current_term = match.group(2).strip()
            current_def = match.group(3).strip()
        elif current_num and line_stripped:
            # Continuation of current definition
            current_def += " " + line_stripped

    # Save last definition
    if current_num:
        definitions.append((current_num, current_term, current_def.strip()))

    return definitions


def _extract_list_items(text: str) -> tuple:
    """
    Ekstrak list items dari konten pasal.
    Return (intro_line, [(number, item_text), ...])
    """
    content = _strip_metadata_header(text)
    lines = content.split("\n")
    intro = ""
    items = []
    current_item = ""
    current_num = ""

    import re
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith("pasal"):
            continue
        # Detect list item: 'N. text' or 'N) text'
        match = re.match(r'^(\d+)[.)]\s+(.*)', line_stripped)
        if match:
            if current_num:
                items.append((current_num, current_item.strip()))
            current_num = match.group(1)
            current_item = match.group(2).strip()
        elif current_num and line_stripped:
            current_item += " " + line_stripped
        elif not items and not current_num and line_stripped:
            intro = line_stripped

    if current_num:
        items.append((current_num, current_item.strip()))

    return intro, items


def _synthesize_definition_answer(text: str, pasal: str) -> str:
    """Buat jawaban sintetis untuk pasal definisi (pasal 1)."""
    defs = _extract_definitions(text)
    if not defs:
        return _synthesize_general_answer(text, pasal)

    if len(defs) == 1:
        _, term, definition = defs[0]
        return f"{term.capitalize()} adalah {definition}."
    else:
        parts = []
        for num, term, definition in defs:
            parts.append(f"({num}) {term.capitalize()} adalah {definition}")
        return f"Pasal 1 mendefinisikan {len(defs)} istilah, yaitu: " + "; ".join(parts) + "."


def _synthesize_list_answer(text: str, pasal: str) -> str:
    """Buat jawaban sintetis untuk pasal berbentuk list."""
    intro, items = _extract_list_items(text)
    if not items:
        return _synthesize_general_answer(text, pasal)

    pasal_label = pasal.replace("pasal ", "Pasal ")
    if intro:
        result = f"{pasal_label} mengatur bahwa {intro.lower()} "
    else:
        result = f"{pasal_label} mengatur hal-hal berikut: "

    item_texts = [f"({num}) {item}" for num, item in items]
    result += "; ".join(item_texts) + "."
    return result


def _synthesize_general_answer(text: str, pasal: str) -> str:
    """Buat jawaban sintetis umum dari konten pasal."""
    content = _strip_metadata_header(text)
    pasal_label = pasal.replace("pasal ", "Pasal ")

    # Ambil inti konten (skip baris 'pasal X')
    lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().lower().startswith("pasal")]
    if not lines:
        return f"{pasal_label} tidak memuat ketentuan spesifik dalam potongan dokumen yang tersedia."

    # Gabungkan menjadi kalimat ringkas (maks 3 kalimat)
    core = " ".join(lines)
    # Batasi panjang
    if len(core) > 500:
        # Potong di titik terakhir sebelum 500 karakter
        last_period = core[:500].rfind(".")
        if last_period > 100:
            core = core[:last_period + 1]
        else:
            core = core[:500].rsplit(" ", 1)[0] + "."

    return f"{pasal_label} mengatur bahwa {core[0].lower()}{core[1:]}"


def generate_cot_answer(chunk: dict) -> tuple:
    """
    Generate (cot_answer, answer) dari oracle chunk.
    Format: ##Reason: ... ##Answer: ...

    Jawaban disintesis secara ringkas, BUKAN dump dokumen mentah.
    """
    pasal = get_pasal(chunk)
    text = get_clean_text(chunk)

    # Tentukan reason template
    reason_templates = [
        f"Informasi ditemukan pada {pasal} dari {TITLE}.",
        f"Berdasarkan {TITLE}, pada bagian {pasal} disebutkan.",
        f"Dari {TITLE} {pasal}, dapat diidentifikasi.",
        f"Merujuk pada {pasal} dari {TITLE}.",
        f"Dalam {TITLE}, {pasal} mengatur hal tersebut.",
    ]
    reason = random.choice(reason_templates)

    # Sintesis jawaban berdasarkan tipe konten
    content = _strip_metadata_header(text)

    if pasal == "pasal 1":
        synthesized = _synthesize_definition_answer(text, pasal)
    elif "meliputi" in content or "meliputi:" in content or "terdiri atas" in content:
        synthesized = _synthesize_list_answer(text, pasal)
    elif _extract_list_items(text)[1]:  # has numbered list items
        synthesized = _synthesize_list_answer(text, pasal)
    else:
        synthesized = _synthesize_general_answer(text, pasal)

    cot_answer = f"##Reason: {reason}\n##Answer: {synthesized}"
    answer = synthesized

    return cot_answer, answer


# ============================================================================
# Distractor selection
# ============================================================================

def select_distractors(chunks: list, exclude_idx: int, n: int = N_DISTRACTORS) -> list:
    """
    Pilih N distractor chunks (bukan oracle).
    Prioritas: pasal berbeda, lalu pasal sama tapi butir berbeda.
    """
    candidates = [i for i in range(len(chunks)) if i != exclude_idx]
    if len(candidates) <= n:
        return [chunks[i] for i in candidates]

    # Shuffle dan ambil n
    random.shuffle(candidates)

    # Prioritaskan pasal berbeda
    oracle_pasal = get_pasal(chunks[exclude_idx])
    diff_pasal = [i for i in candidates if get_pasal(chunks[i]) != oracle_pasal]
    same_pasal = [i for i in candidates if get_pasal(chunks[i]) == oracle_pasal]

    selected = []
    # Ambil dari pasal berbeda dulu
    for idx in diff_pasal[:n]:
        selected.append(chunks[idx])

    # Jika masih kurang, ambil dari pasal sama
    for idx in same_pasal:
        if len(selected) >= n:
            break
        selected.append(chunks[idx])

    return selected[:n]


def select_no_oracle_distractors(chunks: list, n: int = N_DISTRACTORS) -> list:
    """Pilih N distractor untuk no-oracle sample."""
    indices = random.sample(range(len(chunks)), min(n, len(chunks)))
    return [chunks[i] for i in indices]


# ============================================================================
# Sample builder
# ============================================================================

def make_doc_block(chunk: dict) -> str:
    """Buat formatted document block dari chunk (raw content)."""
    return get_clean_text(chunk)


def build_oracle_sample(chunk: dict, chunks: list, chunk_idx: int, sample_id: str) -> dict:
    """
    Bangun satu sampel RAFT dengan oracle document.

    Struktur (Zhang et al., 2024):
    {Q + D* + D1...Dk → A*}
    - Q: pertanyaan
    - D*: oracle document (gold)
    - D1...Dk: distractor documents
    - A*: jawaban dengan CoT
    """
    # Pilih distractors
    distractors = select_distractors(chunks, chunk_idx)
    distractor_ids = [d["metadata"]["chunk_index"] for d in distractors]

    # Gold position: random 0..N_DISTRACTORS
    gold_position = random.randint(0, N_DISTRACTORS)

    # Bangun list dokumen: insert oracle di gold_position
    distractor_blocks = [make_doc_block(d) for d in distractors]
    oracle_block = make_doc_block(chunk)

    all_doc_blocks = list(distractor_blocks)
    all_doc_blocks.insert(gold_position, oracle_block)

    # Bangun context array (raw text) dari chunk objects
    all_chunks_ordered = list(distractors)
    all_chunks_ordered.insert(gold_position, chunk)
    context_array = [get_clean_text(c) for c in all_chunks_ordered]

    # Bangun instruction
    context_block = build_context_block(all_doc_blocks)
    instruction = f"{SYSTEM_PROMPT_ORACLE}\n\n{context_block}"

    # Generate question & answer
    question = generate_question(chunk)
    cot_answer, answer = generate_cot_answer(chunk)

    return {
        "id": sample_id,
        "instruction": instruction,
        "question": question,
        "cot_answer": cot_answer,
        "answer": answer,
        "context": context_array,
        "oracle_context": get_clean_text(chunk),
        "metadata": {
            "gold_doc_id": f"chunk_{chunk['metadata']['chunk_index']}",
            "has_oracle": True,
            "gold_position": gold_position,
            "distractor_ids": [f"chunk_{did}" for did in distractor_ids],
            "n_distractors": N_DISTRACTORS,
            "village_name": VILLAGE,
            "title": TITLE,
            "section": get_pasal(chunk),
            "generated_by": "raft_generator",
            "created_at": datetime.now().isoformat(),
        },
    }


def build_no_oracle_sample(chunks: list, sample_id: str, target_pasal: str = None) -> dict:
    """
    Bangun sampel tanpa oracle document.
    Zhang et al. (2024): (1-P)% data tanpa oracle untuk memorization training.
    """
    # Pilih distractors
    distractors = select_no_oracle_distractors(chunks, N_DISTRACTORS)
    distractor_ids = [d["metadata"]["chunk_index"] for d in distractors]

    # Semua dokumen adalah distractor
    all_doc_blocks = [make_doc_block(d) for d in distractors]

    # Bangun instruction (pakai prompt no-oracle)
    context_block = build_context_block(all_doc_blocks)
    instruction = f"{SYSTEM_PROMPT_NO_ORACLE}\n\n{context_block}"

    # Pertanyaan tentang pasal yang TIDAK ada di konteks
    if target_pasal:
        question = f"Apa isi dari {target_pasal} dalam {TITLE}?"
    else:
        # Cari pasal yang tidak ada di distractors
        used_pasals = set(get_pasal(d) for d in distractors)
        all_pasals = ["pasal 1", "pasal 2", "pasal 3", "pasal 4", "pasal 5", "pasal 6"]
        missing = [p for p in all_pasals if p not in used_pasals]
        target = random.choice(missing) if missing else "pasal 3"
        question = f"Apa isi dari {target} dalam {TITLE}?"

    context_array = [get_clean_text(d) for d in distractors]

    return {
        "id": sample_id,
        "instruction": instruction,
        "question": question,
        "cot_answer": NO_ORACLE_COT,
        "answer": NO_ORACLE_ANSWER,
        "context": context_array,
        "oracle_context": "",
        "metadata": {
            "gold_doc_id": None,
            "has_oracle": False,
            "gold_position": None,
            "distractor_ids": [f"chunk_{did}" for did in distractor_ids],
            "n_distractors": N_DISTRACTORS,
            "village_name": VILLAGE,
            "title": TITLE,
            "section": "N/A",
            "generated_by": "raft_generator",
            "created_at": datetime.now().isoformat(),
        },
    }


# ============================================================================
# Main generation
# ============================================================================

def generate_dataset(chunks: list) -> list:
    """Generate dataset RAFT lengkap dari chunks."""
    samples = []
    sample_idx = 0

    # Filter chunks yang layak jadi oracle
    usable_chunks = [
        (i, c) for i, c in enumerate(chunks)
        if len(get_clean_text(c)) >= MIN_CONTENT_CHARS
    ]

    print(f"  Total chunks        : {len(chunks)}")
    print(f"  Chunks usable       : {len(usable_chunks)} (>= {MIN_CONTENT_CHARS} chars)")

    # Hitung target no-oracle
    n_oracle = len(usable_chunks)
    n_no_oracle = max(1, int(n_oracle * (1 - P_ORACLE) / P_ORACLE))
    n_total = n_oracle + n_no_oracle

    print(f"  Target oracle       : {n_oracle} ({P_ORACLE*100:.0f}%)")
    print(f"  Target no-oracle    : {n_no_oracle} ({(1-P_ORACLE)*100:.0f}%)")
    print(f"  Target total        : {n_total}")
    print()

    # Generate oracle samples
    print("[1/2] Generating oracle samples...")
    for idx, chunk in usable_chunks:
        sample_id = f"sample_{sample_idx:05d}"
        sample = build_oracle_sample(chunk, chunks, idx, sample_id)
        samples.append(sample)
        sample_idx += 1

    print(f"  Generated: {len(samples)} oracle samples")

    # Generate no-oracle samples
    print("[2/2] Generating no-oracle samples...")
    # Pilih pasal target yang beragam
    no_oracle_pasals = ["pasal 3", "pasal 5", "pasal 1", "pasal 6", "pasal 4",
                        "pasal 2", "pasal 6", "pasal 1", "pasal 3", "pasal 5"]
    for i in range(n_no_oracle):
        sample_id = f"sample_{sample_idx:05d}"
        target = no_oracle_pasals[i % len(no_oracle_pasals)]
        sample = build_no_oracle_sample(chunks, sample_id, target)
        samples.append(sample)
        sample_idx += 1

    print(f"  Generated: {n_no_oracle} no-oracle samples")
    print(f"\n  TOTAL: {len(samples)} samples")

    return samples


def print_statistics(samples: list):
    """Cetak statistik dataset."""
    n_total = len(samples)
    n_oracle = sum(1 for s in samples if s["metadata"]["has_oracle"])
    n_no_oracle = n_total - n_oracle

    gold_positions = Counter(
        s["metadata"]["gold_position"]
        for s in samples
        if s["metadata"]["has_oracle"]
    )

    sections = Counter(
        s["metadata"]["section"]
        for s in samples
        if s["metadata"]["has_oracle"]
    )

    print(f"\n{'='*60}")
    print("STATISTIK DATASET RAFT")
    print(f"{'='*60}")
    print(f"  Total sampel        : {n_total}")
    print(f"  Dengan oracle (P%)  : {n_oracle} ({100*n_oracle/n_total:.1f}%)")
    print(f"  Tanpa oracle (1-P)% : {n_no_oracle} ({100*n_no_oracle/n_total:.1f}%)")
    print(f"  Distractors/sample  : {N_DISTRACTORS}")
    print()
    print(f"  Distribusi gold position:")
    for pos in sorted(gold_positions.keys()):
        print(f"    Position {pos}: {gold_positions[pos]} samples")
    print()
    print(f"  Distribusi pasal (oracle):")
    for sec, count in sections.most_common():
        print(f"    {sec}: {count} samples")
    print()

    # Average content lengths
    oracle_contents = [
        len(s["oracle_context"])
        for s in samples if s["metadata"]["has_oracle"]
    ]
    if oracle_contents:
        print(f"  Oracle content length:")
        print(f"    Min  : {min(oracle_contents)} chars")
        print(f"    Max  : {max(oracle_contents)} chars")
        print(f"    Mean : {sum(oracle_contents)/len(oracle_contents):.0f} chars")

    print(f"{'='*60}")


def save_dataset(samples: list, path: str):
    """Simpan dataset ke JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nDataset disimpan ke: {path}")
    print(f"  Format : JSONL (satu JSON object per baris)")
    print(f"  Size   : {os.path.getsize(path) / 1024:.1f} KB")


def print_preview(samples: list, n: int = 3):
    """Print preview N sampel pertama."""
    print(f"\n{'='*60}")
    print(f"PREVIEW ({n} sampel pertama)")
    print(f"{'='*60}")

    for s in samples[:n]:
        print(f"\n--- {s['id']} ---")
        print(f"  Oracle     : {s['metadata']['has_oracle']}")
        print(f"  Gold pos   : {s['metadata']['gold_position']}")
        print(f"  Section    : {s['metadata']['section']}")
        print(f"  Question   : {s['question']}")
        print(f"  CoT Answer : {s['cot_answer'][:150]}...")
        print(f"  Docs       : {len(s['context'])} dokumen")
        if s["metadata"]["has_oracle"]:
            print(f"  Oracle ctx : {s['oracle_context'][:100]}...")


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 60)
    print("RAFT DATASET GENERATOR")
    print("Zhang et al. (2024) - Adapting Language Model to Domain RAG")
    print("=" * 60)

    # Load chunks
    print(f"\nLoading chunks dari: {CHUNKS_PATH}")
    if not os.path.exists(CHUNKS_PATH):
        print(f"[ERROR] File tidak ditemukan: {CHUNKS_PATH}")
        print("Pastikan script dijalankan dari folder notebooks/")
        return

    chunks = load_chunks(CHUNKS_PATH)
    print(f"  Loaded {len(chunks)} chunks")

    # Generate dataset
    print(f"\nGenerating RAFT dataset...")
    samples = generate_dataset(chunks)

    # Print statistics
    print_statistics(samples)

    # Save
    save_dataset(samples, OUTPUT_PATH)

    # Preview
    print_preview(samples, n=3)

    print(f"\n{'='*60}")
    print("SELESAI!")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Samples : {len(samples)}")
    print(f"  Format  : JSONL (siap untuk fine-tuning)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
