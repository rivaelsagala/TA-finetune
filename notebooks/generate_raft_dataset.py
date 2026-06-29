"""
RAFT Dataset Generator — Implementasi Standar Riset Terbaik
===========================================================
Berdasarkan Paper "RAFT: Adapting Language Model to Domain Specific RAG" (UC Berkeley & Microsoft).

Pembaruan Utama:
1. Memisahkan secara tegas Oracle Pool (dokumen valid) dan Distractor Pool (dokumen usang dari folder distraktor).
2. Distractor dari folder `distraktor` TIDAK AKAN PERNAH dijadikan Oracle, memastikan model tidak belajar aturan kedaluwarsa.
3. Implementasi rasio 80/20 (Oracle Present/Absent) untuk melatih model menolak halusinasi (refusal behavior).
4. Pembuatan Chain-of-Thought (CoT) yang mewajibkan verbatim citation (pengutipan kalimat asli) saat Oracle ada.
"""

import os, sys, json, time, random, re, shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util
    print("Memuat model embedding (all-MiniLM-L6-v2)...")
    EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except ImportError:
    EMBEDDING_MODEL = None
    print("Warning: 'sentence-transformers' tidak ditemukan. Menggunakan fallback lexical (Jaccard).")

EMBED_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP & KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
HF_BASE_URL = f"{_base}/chat/completions" if not _base.endswith("/chat/completions") else _base

GENERATOR_MODEL  = "openai/gpt-4.1-mini"
VALIDATOR_MODEL  = "openai/gpt-4.1-mini"
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
DISTRAKTOR_DIR   = PROCESSED_DIR / "distraktor"
DATASET_DIR      = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS           = 1024
REQUEST_DELAY        = 0.2
NUM_DISTRACTORS      = 4
ORACLE_PRESENT_RATIO = 0.80   # 80% oracle hadir, 20% absent

# ─────────────────────────────────────────────────────────────────────────────
# TIPE PERTANYAAN & GAYA
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "factual": {
        "weight": 0.20,
        "prompt": "Buat pertanyaan FAKTUAL spesifik tentang angka, tanggal, atau fakta yang HANYA ada di dokumen ini. Wajib sebut nama peraturan dan nomor pasal."
    },
    "definition": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan DEFINISIONAL tentang istilah teknis dalam dokumen. Sertakan nama peraturan."
    },
    "procedural": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan PROSEDURAL tentang langkah/tata cara operasional. Boleh sebut peraturannya, boleh tidak."
    },
    "reasoning": {
        "weight": 0.15,
        "prompt": "Buat pertanyaan INFERENSI/KONDISIONAL tentang syarat atau konsekuensi. Sebutkan nama peraturannya."
    },
    "comparative": {
        "weight": 0.10,
        "prompt": "Buat pertanyaan KOMPARATIF yang membandingkan kewajiban/hak dalam dokumen tersebut."
    },
    "natural_awam": {
        "weight": 0.25,
        "prompt": "Buat pertanyaan NATURAL bergaya bahasa sehari-hari (awam) seolah-olah ditanyakan oleh warga desa biasa yang tidak tahu hukum. JANGAN menyebutkan nama peraturan, JANGAN sebut nomor pasal, dan gunakan bahasa santai/kolokial. Contoh: 'Gimana sih syaratnya kalau mau jadi anggota BPD?' atau 'Kalau kades ngelanggar aturan, sanksinya apa ya?'"
    }
}

COMPLETION_STYLES = [
    "Formal (Gunakan bahasa hukum, kutip pasalnya dengan rapi dan padat).",
    "Natural (Gunakan bahasa yang lebih natural untuk warga awam, namun tetap akurat dan bersumber langsung dari dokumen)."
]

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages: List[Dict], temperature: float = 0.7, max_tokens: int = MAX_TOKENS, retries: int = 3, model: str = GENERATOR_MODEL) -> Optional[str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.9, "stream": False
    }
    for attempt in range(retries):
        try:
            r = requests.post(HF_BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            wait = (attempt + 1) * 5
            time.sleep(wait)
        except Exception:
            time.sleep(3)
    return None

def clean_content(text: str) -> str:
    text = re.sub(r"^\[dokumen:.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(desa|kabupaten|nomor|tahun):.*?\]\s*", "", text)
    return text.strip()

def is_substantive(content: str, min_chars: int = 150) -> bool:
    c = clean_content(content)
    if len(c) < min_chars: return False
    
    words = c.split()
    return len(words) >= 5

def get_doc_label(chunk: Dict) -> str:
    title = chunk.get("metadata", {}).get("title", "Dokumen tidak diketahui")
    pasal = chunk.get("pasal", "") or chunk.get("metadata", {}).get("section", "")
    return f"{title}, {pasal.title()}" if pasal else title

def load_chunks_from_dir(directory: Path, is_distractor: bool = False) -> List[Tuple[str, Dict, bool]]:
    loaded = []
    if not directory.exists(): return loaded
    for fpath in directory.glob("*_chunks.json"):
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            doc_id = chunks[0]["metadata"]["document_id"] if chunks else fpath.stem
            for c in chunks:
                loaded.append((doc_id, c, is_distractor))
    return loaded

def semantic_similarity_score(text_a: str, text_b: str) -> float:
    if EMBEDDING_MODEL:
        if text_a not in EMBED_CACHE:
            EMBED_CACHE[text_a] = EMBEDDING_MODEL.encode(text_a, convert_to_tensor=True)
        if text_b not in EMBED_CACHE:
            EMBED_CACHE[text_b] = EMBEDDING_MODEL.encode(text_b, convert_to_tensor=True)
        return util.pytorch_cos_sim(EMBED_CACHE[text_a], EMBED_CACHE[text_b]).item()
    else:
        stopwords = {"yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini", "itu", "atau", "tidak", "dalam", "adalah", "oleh", "pasal", "nomor", "ayat", "tahun"}
        def tokenize(t):
            return set(w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in stopwords)
        a, b = tokenize(text_a), tokenize(text_b)
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC PEMILIHAN DISTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_distractors(oracle_text: str, oracle_doc_id: str, distractor_pool: List[Tuple[str, Dict, bool]], n: int) -> List[Dict]:
    """
    Pilih distractor dengan overlap moderat.
    Sangat diprioritaskan untuk mengambil dari pure_distractors (dokumen di folder distraktor).
    """
    candidates = []
    for d_id, chunk, is_pure_distractor in distractor_pool:
        if d_id == oracle_doc_id: continue
        c_text = clean_content(chunk["content"])
        score = semantic_similarity_score(oracle_text, c_text)
        
        # Boost skor untuk pure_distractor (dokumen usang) agar lebih sering terpilih
        if is_pure_distractor:
            score += 0.2 if EMBEDDING_MODEL else 0.5
            
        # Kita ingin overlap moderat (topiknya mirip, tapi isinya berbeda)
        lower_bound = 0.35 if EMBEDDING_MODEL else 0.1
        upper_bound = 0.75 if EMBEDDING_MODEL else 0.9
        
        if lower_bound <= score <= upper_bound:
            candidates.append((score, chunk))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Filter for diversity (hindari duplicate pasal/dokumen yang persis sama)
    selected_chunks = []
    seen_labels = set()
    
    for score, chunk in candidates:
        lbl = get_doc_label(chunk)
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            selected_chunks.append(chunk)
        if len(selected_chunks) >= max(n * 2, 8):
            break
            
    random.shuffle(selected_chunks)
    return selected_chunks[:n]

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def generate_question(oracle_content: str, doc_title: str) -> Optional[Tuple[str, str]]:
    q_type = random.choices(list(QUESTION_TYPES.keys()), weights=[v["weight"] for v in QUESTION_TYPES.values()])[0]
    prompt = QUESTION_TYPES[q_type]["prompt"]
    
    system = (
        "Anda adalah pakar pembuat dataset RAFT RAG bidang Hukum Desa.\n"
        f"TUGAS: {prompt}\n"
        "ATURAN: Pertanyaan spesifik bisa dijawab 100% dari dokumen, output HANYA pertanyaan tanpa teks pembuka/penutup."
    )
    res = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Dokumen: {doc_title}\n{oracle_content}\nBuat pertanyaan:"}
    ])
    if not res: return None
    q = re.sub(r"^(\d+[\.\)]\s*|pertanyaan[:\s]*)", "", res, flags=re.IGNORECASE).strip().strip('"\'')
    return (q_type, q) if len(q) > 20 else None

def generate_thought_and_completion(
    question: str, docs: List[str], doc_labels: List[str], oracle_idx: int, style: str, answer_type: str
) -> Tuple[Optional[str], Optional[str]]:
    
    docs_fmt = "".join(f"\n--- Dokumen {i+1}: {l} ---\n{d}\n" for i, (d, l) in enumerate(zip(docs, doc_labels)))
    oracle_present = oracle_idx >= 0
    
    hint = (
        f"\n[INTERNAL] Dokumen {oracle_idx+1} mengandung jawaban." if oracle_present else
        "\n[INTERNAL] TIDAK ADA dokumen yang menjawab pertanyaan ini."
    )
    
    t_instr = (
        "Buat 'thought_process' SANGAT SINGKAT (maksimal 1-2 kalimat). "
        "Contoh: 'Dokumen 2 berisi definisi pemerintahan desa sehingga relevan.' "
        "Tidak perlu membahas dokumen lain yang tidak relevan."
    )
    
    if oracle_present:
        if answer_type == "extractive":
            c_instr = (
                f"Gaya: {style}\nJawab SECARA EKSTRAKTIF. Kutip langsung kalimat dari dokumen (Berdasarkan Pasal X...). "
                "DILARANG KERAS menambahkan opini, contoh buatan (halusinasi), atau kata pengantar bertele-tele seperti 'Jadi...' atau 'Misalnya...' yang tidak ada di dokumen."
            )
        else:
            c_instr = (
                f"Gaya: {style}\nJawab SECARA ABSTRAKTIF. Parafrase jawaban agar natural, "
                "TETAPI SEMUA INFORMASI WAJIB 100% BERSUMBER DARI DOKUMEN. DILARANG KERAS berhalusinasi atau menambahkan fakta luar."
            )
    else:
        c_instr = "Tolak menjawab secara halus. Sebutkan bahwa dokumen yang diberikan membahas topik lain dan tidak mengandung informasi untuk menjawab pertanyaan tersebut."

    sys_msg = (
        "Anda AI pembuat data RAFT.\nOutput JSON valid:\n"
        '{"thought_process": "...", "completion": "..."}\n'
        f"ATURAN THOUGHT:\n{t_instr}\nATURAN COMPLETION:\n{c_instr}"
    )
    
    res = call_llm(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Q: {question}\n{docs_fmt}\n{hint}"}],
        temperature=0.3, max_tokens=1024
    )
    
    if not res: return None, None
    for attempt in [res, re.search(r'\{[\s\S]*\}', res)]:
        try:
            raw = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
            if not raw: continue
            data = json.loads(raw)
            return data.get("thought_process", "").strip(), data.get("completion", "").strip()
        except: continue
    return None, None

def validate_sample(sample: Dict) -> Optional[Dict]:
    sys_msg = (
        "Anda adalah evaluator ketat dataset RAFT. "
        "Validasi kualitas sampel QA berikut berdasarkan pedoman ini:\n"
        "1. instruction_answered: true HANYA JIKA completion benar-benar menjawab instruction (Misal: jika ditanya syarat, berikan syarat, bukan sekadar definisi).\n"
        "2. grounded: true HANYA JIKA completion murni dari dokumen. JIKA completion mengandung informasi sekecil apapun yang tidak muncul di dokumen (opini, tambahan kata pengantar 'Misalnya..'), maka grounded=false.\n"
        "3. oracle_consistency: true JIKA (oracle_present=false -> completion menolak) atau sebaliknya.\n"
        "4. thought_correct: true JIKA thought_process menunjuk ke dokumen yang benar (cek oracle_index jika ada).\n"
        "5. answer_type_correct: true JIKA gaya jawaban sesuai answer_type (extractive tidak boleh diparafrase, abstractive harus diparafrase dari dokumen).\n"
        "Output JSON:\n"
        '{"pass": boolean, "instruction_answered": boolean, "grounded": boolean, "oracle_consistency": boolean, "thought_correct": boolean, "answer_type_correct": boolean, "score": float, "reason": "Alasan spesifik jika false"}'
    )
    
    user_msg = (
        f"Instruction: {sample['instruction']}\n"
        f"Documents: {json.dumps(sample['documents'], ensure_ascii=False)}\n"
        f"Thought Process: {sample['thought_process']}\n"
        f"Completion: {sample['completion']}\n"
        f"Metadata: {json.dumps(sample['metadata_extra'], ensure_ascii=False)}"
    )
    
    res = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], temperature=0.0, max_tokens=256, model=VALIDATOR_MODEL)
    if not res: return None
    try:
        match = re.search(r'\{[\s\S]*\}', res)
        if match: return json.loads(match.group())
    except: pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_raft_pipeline(output_path: Path):
    print("Memuat dokumen utama...")
    oracle_pool = load_chunks_from_dir(PROCESSED_DIR, is_distractor=False)
    print(f"Memuat pure distractors (dari {DISTRAKTOR_DIR.name})...")
    pure_distractors = load_chunks_from_dir(DISTRAKTOR_DIR, is_distractor=True)
    
    distractor_pool = oracle_pool + pure_distractors
    valid_oracles = [(d, c) for d, c, is_dis in oracle_pool if not is_dis and is_substantive(c["content"])]
    
    print(f"Total Chunks: {len(oracle_pool)} utama, {len(pure_distractors)} pure distractors")
    print(f"Valid Oracles yang siap diproses: {len(valid_oracles)}\n")

    dataset, failed = [], 0
    with open(output_path, "w", encoding="utf-8") as f:
        pass # Reset file

    for doc_id, chunk in tqdm(valid_oracles, desc="Generating RAFT Dataset"):
        oracle_text = clean_content(chunk["content"])
        oracle_lbl = get_doc_label(chunk)
        
        success = False
        for attempt in range(5): # Coba maksimal 5 kali jika tidak pass
            # 1. Generate Q
            q_res = generate_question(oracle_text, oracle_lbl)
            if not q_res: continue
            q_type, question = q_res
            time.sleep(REQUEST_DELAY)
            
            # 2. Select Distractors
            d_chunks = select_distractors(oracle_text, doc_id, distractor_pool, NUM_DISTRACTORS)
            
            # Fallback padding if not enough distractors
            while len(d_chunks) < NUM_DISTRACTORS and len(distractor_pool) > 0:
                random_fallback = random.choice(distractor_pool)[1]
                if random_fallback not in d_chunks:
                    d_chunks.append(random_fallback)

            d_texts = [clean_content(c["content"]) for c in d_chunks]
            d_lbls = [get_doc_label(c) for c in d_chunks]
            
            # 3. Setup Oracle Present vs Absent
            is_present = random.random() < ORACLE_PRESENT_RATIO
            if is_present:
                pos = random.randint(0, len(d_texts))
                docs = d_texts[:pos] + [oracle_text] + d_texts[pos:]
                lbls = d_lbls[:pos] + [oracle_lbl] + d_lbls[pos:]
            else:
                docs, lbls, pos = d_texts, d_lbls, -1
                
            style = random.choice(COMPLETION_STYLES)
            answer_type = "extractive" if random.random() < 0.7 else "abstractive"
            
            # 4. Generate Thought + Completion
            thought, completion = generate_thought_and_completion(question, docs, lbls, pos, style, answer_type)
            time.sleep(REQUEST_DELAY)
            
            if not thought or not completion: continue
                
            sample = {
                "instruction": question,
                "documents": docs,
                "thought_process": thought,
                "completion": completion,
                "metadata_extra": {
                    "question_type": q_type,
                    "answer_type": answer_type if is_present else None,
                    "oracle_present": is_present,
                    "oracle_doc_id": doc_id if is_present else None,
                    "oracle_index": pos if is_present else None
                }
            }
            
            # 5. Validation (LLM-as-a-Judge)
            validation_res = validate_sample(sample)
            time.sleep(REQUEST_DELAY)
            
            if (
                validation_res 
                and validation_res.get("pass", False) 
                and validation_res.get("instruction_answered", False)
                and validation_res.get("grounded", False)
                and validation_res.get("oracle_consistency", False)
                and validation_res.get("thought_correct", False)
                and validation_res.get("answer_type_correct", False)
            ):
                sample["validation"] = validation_res
                dataset.append(sample)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                success = True
                break # Berhasil, keluar dari attempt loop
                
        if not success:
            failed += 1
            
    print(f"\\nSelesai! Berhasil: {len(dataset)}, Gagal: {failed}")
    print(f"Tersimpan di: {output_path}")

if __name__ == "__main__":
    output = DATASET_DIR / "raft_dataset_final.jsonl"
    run_raft_pipeline(output)
