# Project Documentation
## RAFT Fine-Tuned LLM for Village Regulation (Perdes) Q&A System
---
# Project Overview
Proyek ini merupakan **Tugas Akhir (TA)** yang membangun sistem *Question & Answer* cerdas berbasis **Large Language Model (LLM)** untuk domain hukum **Peraturan Desa (Perdes)** di Indonesia. Permasalahan utama yang diselesaikan adalah keterbatasan akses masyarakat desa terhadap isi peraturan lokal mereka — dokumen Perdes seringkali bersifat teknis, tersebar, dan sulit dipahami oleh warga awam.
Solusi yang diusulkan menggabungkan dua pendekatan mutakhir: **Retrieval-Augmented Generation (RAG)** dan **Retrieval-Augmented Fine-Tuning (RAFT)**. Model Llama 3.1 8B Instruct di-*fine-tune* secara domain-spesifik menggunakan metode RAFT, sehingga mampu menerima beberapa potongan dokumen Perdes sebagai konteks, mengidentifikasi dokumen yang relevan (gold document), mengabaikan dokumen yang tidak relevan (distractor), dan menghasilkan jawaban yang akurat disertai proses penalaran Chain-of-Thought (CoT). Sistem ini diekspos melalui REST API sehingga dapat diintegrasikan dengan aplikasi frontend atau sistem RAG yang sudah ada.
---
# Technology Stack
| Teknologi | Versi | Fungsi |
|---|---|---|
| **Meta Llama 3.1 8B Instruct** | - | Base LLM yang di-fine-tune untuk domain Perdes |
| **Unsloth** | >= 2024.5 | Mempercepat fine-tuning 2x dan efisiensi VRAM |
| **QLoRA (4-bit NF4)** | PEFT >= 0.10 | Kuantisasi 4-bit + Low-Rank Adaptation untuk fine-tuning hemat memori |
| **TRL / SFTTrainer** | >= 0.8.0 | Supervised Fine-Tuning trainer |
| **HuggingFace Transformers** | >= 4.40.0 | Tokenizer, model loading, & training infrastructure |
| **HuggingFace Datasets** | >= 2.18.0 | Loading dan preprocessing dataset JSONL |
| **BitsAndBytes** | >= 0.43.0 | Kuantisasi 4-bit NF4 untuk QLoRA |
| **Accelerate** | >= 0.27.0 | Distributed training & device mapping |
| **PyTorch** | >= 2.0.0 | Deep learning framework & CUDA runtime |
| **Flask** | 3.0.0 | REST API server untuk serving model inference |
| **Flask-CORS** | 4.0.0 | Cross-Origin Resource Sharing untuk akses eksternal |
| **Python** | 3.x | Bahasa pemrograman utama |
---
# System Architecture
Sistem terdiri dari **tiga lapisan utama** yang saling terhubung:
```
+--------------------------------------------------------------+
|                       CLIENT / RAG SYSTEM                    |
|          (Frontend App / Postman / External Service)         |
+-----------------------------+--------------------------------+
                              | HTTP REST (Port 6000)
+-----------------------------v--------------------------------+
|                    BACKEND API LAYER (Flask)                 |
|  +--------------+  +------------------+  +---------------+  |
|  |  app.py      |  |   routes.py      |  |  run.py       |  |
|  | (App Init +  |  | (Endpoint Router)|  | (Entry Point) |  |
|  |   CORS)      |  |  /api/chat       |  |               |  |
|  +--------------+  |  /api/chat-rag   |  +---------------+  |
|                    |  /api/load-model |                      |
|                    |  /api/model-info |                      |
|                    |  /api/models     |                      |
|                    |  /api/health     |                      |
|                    +--------+---------+                      |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                    SERVICE LAYER (llama_service.py)          |
|  +--------------------------------------------------------+  |
|  |  load_model()          -> Load model + tokenizer ke GPU|  |
|  |  generate_answer()     -> Plain chat (no RAG context)  |  |
|  |  generate_answer_rag() -> RAFT inference + CoT parsing |  |
|  |  _split_cot_answer()   -> Pisah analisis vs jawaban    |  |
|  |  _enrich_jawaban()     -> Post-processing output       |  |
|  +--------------------------------------------------------+  |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                    MODEL LAYER (GPU / CUDA)                  |
|  +------------------------+  +----------------------------+  |
|  |  model_merged_raft_    |  |  Meta-Llama-3.1-8B-Instruct|  |
|  |  perdes/               |  |  (Base Model / Fallback)   |  |
|  |  (RAFT Fine-tuned)     |  +----------------------------+  |
|  +------------------------+                                  |
+--------------------------------------------------------------+
```
**Komponen utama:**
- **Training Pipeline** (`notebooks/finetune_lora_rag.py`): Pipeline offline untuk fine-tuning model dengan dataset RAFT Perdes.
- **Backend API** (`BE/`): Server Flask yang melayani inference request dari luar.
- **LLM Service** (`BE/llama_service.py`): Inti logika inference — load model, format prompt, generate, dan post-process output.
- **Dataset** (`data/`): Dataset RAFT dalam format JSONL berisi pasangan Q&A dengan dokumen gold + distractor.
---
# System Workflow
### A. Training Workflow (Offline)
```
1. Persiapan Lingkungan
   └── Verifikasi CUDA & GPU tersedia
2. Load Base Model (QLoRA 4-bit)
   └── Unsloth FastLanguageModel.from_pretrained()
   └── Konfigurasi LoRA (r=16, alpha=16, target all projection layers)
3. Load & Format Dataset RAFT
   └── Baca raft_perdes_dataset.jsonl
   └── Format: instruction + documents -> thought_process + completion
   └── Template: Llama 3.1 Chat Format (<|begin_of_text|>...<|eot_id|>)
   └── Anti-monoton: variasi transition phrase & analysis opener per sampel
4. Training (SFTTrainer)
   └── LR = 2e-4, Epochs = 15, Batch = 2, Grad Accum = 4 (Eff. Batch = 8)
   └── Optimizer: AdamW 8-bit, Scheduler: Linear Decay
   └── Warmup 5 steps, Weight Decay = 0.02
5. Simpan Model
   └── Simpan LoRA Adapter -> lora_adapter_raft_perdes/
   └── Merge ke base model (16-bit) -> model_merged_raft_perdes/
6. Uji Inferensi (Sanity Check)
   └── Tes dengan sampel RAFT format untuk validasi output
```
### B. Inference Workflow (Online / API)
```
1. Client mengirim POST /api/chat-rag
   Body: { "pertanyaan": "...", "dokumen": ["...", "...", "..."] }
2. routes.py -> validate input (pertanyaan & dokumen wajib ada)
3. llama_service.generate_answer_rag()
   └── Format dokumen: "Dokumen 1:\n{doc1}\n\nDokumen 2:\n{doc2}..."
   └── Susun messages dengan RAFT_SYSTEM_PROMPT
   └── apply_chat_template() -> tokenize -> GPU
4. model.generate()
   └── max_new_tokens=512, temperature=0.7, top_p=0.9
   └── top_k=50, min_p=0.05, repetition_penalty=1.15
5. Post-processing output
   └── _split_cot_answer(): pisah [analisis CoT] dari [jawaban akhir]
   └── _enrich_jawaban(): bersihkan referensi "Dokumen N", perkaya jawaban
6. Return JSON Response
   { "analisis": "...", "jawaban": "...", "raw_response": "...",
     "model_type": "raft", "num_documents": 3 }
```
---
# Fine-Tuning Pipeline (`finetune_lora_rag.py`)
Bagian ini mendokumentasikan implementasi lengkap pipeline fine-tuning LoRA yang terdapat pada file `finetune_lora_rag.py`.

### Ringkasan Tahapan

| Step | Modul | Fungsi |
|---|---|---|
| 0 | Instalasi dependensi | Menyiapkan environment (unsloth, trl, peft, dll.) |
| 1 | `Config` dataclass | Sentralisasi semua hyperparameter |
| 2 | Load Model + LoRA | Load base model 4-bit + pasang LoRA adapter |
| 3 | Format Prompt RAFT | Konversi sampel dataset ke format messages Llama-3 Chat |
| 4 | Load & Proses Dataset | Baca JSONL → tokenisasi → train/eval split |
| 5 | Training (SFTTrainer) | Supervised fine-tuning dengan SFTConfig |
| 6 | Simpan LoRA Adapter | Ekspor adapter ke direktori output |
| 7 | Merge ke Full Model | Fuse LoRA ke base model untuk deployment |
| 8 | Quick Test | Sanity check inference pasca-training |

---

### Step 1 — Konfigurasi (`Config` Dataclass)
Semua hyperparameter dipusatkan dalam satu dataclass untuk kemudahan eksperimen:

```python
@dataclass
class Config:
    # Model
    base_model: str = "../model/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True           # QLoRA 4-bit NF4

    # LoRA
    lora_r: int = 16                    # rank adapter
    lora_alpha: int = 32                # scaling = alpha/r = 2×
    lora_dropout: float = 0.05
    lora_target_modules: list = [       # semua projection layer
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    use_gradient_checkpointing: str = "unsloth"  # hemat VRAM 30%

    # Dataset
    dataset_path: str = "../data/raft_perdes_dataset.jsonl"
    test_size: float = 0.05             # 5% untuk validasi

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Output
    output_dir: str = "model/raft-llama3-lora"
    merged_output_dir: str = "./raft-llama3-merged"
```

---

### Step 2 — Load Model + LoRA Adapter
Model dimuat menggunakan **Unsloth `FastLanguageModel`** untuk optimasi kecepatan dan efisiensi memori:

```python
# Load base model dengan kuantisasi 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg.base_model,
    max_seq_length=cfg.max_seq_length,
    dtype=None,          # auto-detect bf16/fp16
    load_in_4bit=cfg.load_in_4bit,
)

# Pasang LoRA adapter (hanya ~1-2% parameter yang dilatih)
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=cfg.lora_target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=cfg.seed,
    use_rslora=False,
    loftq_config=None,
)

# Set tokenizer dengan template Llama-3 chat
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

**Catatan LoRA:**
| Parameter | Nilai | Keterangan |
|---|---|---|
| `r` (rank) | 16 | Dimensi matriks low-rank; lebih besar = lebih ekspresif |
| `lora_alpha` | 32 | Scaling factor; efektif scaling = alpha/r = 2× |
| `lora_dropout` | 0.05 | Regularisasi untuk mencegah overfitting adapter |
| `target_modules` | 7 layer | Semua Q, K, V, O proj + MLP gate/up/down |
| `gradient_checkpointing` | `"unsloth"` | Mode khusus Unsloth: hemat VRAM ~30% |

---

### Step 3 — Format Prompt RAFT
Prompt dibangun menggunakan struktur **Llama-3 Chat Template** dengan tiga role: `system`, `user`, `assistant`.

**System Prompt:**
```
Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen
peraturan desa. Gunakan HANYA informasi dari dokumen yang diberikan.
Jika dokumen tidak mengandung jawaban, katakan bahwa informasi tidak tersedia.
Sebelum menjawab, analisis terlebih dahulu relevansi setiap dokumen.
```

**Format User Message:**
```
### Instruksi
{pertanyaan}

### Dokumen Konteks
[Dokumen 1]
{isi_dokumen_1}

[Dokumen 2]
{isi_dokumen_2}

[Dokumen 3]
{isi_dokumen_3}

### Pertanyaan
{pertanyaan}
```

**Format Assistant Message (Target Training):**
```
<CoT>
{thought_process — analisis relevansi tiap dokumen}
</CoT>

{completion — jawaban akhir berdasarkan gold document}
```

Fungsi kunci:
```python
def format_documents(docs: list[str]) -> str:
    """Ubah list dokumen menjadi string bernomor [Dokumen N]."""

def build_messages(sample: dict, include_answer: bool = True) -> list[dict]:
    """Bangun struktur messages untuk satu sampel RAFT."""

def tokenize_sample(sample: dict) -> dict:
    """Tokenisasi satu sampel; hanya hitung loss pada bagian Assistant."""
```

> **Penting:** Tag `<CoT>...</CoT>` membungkus proses analisis, sementara jawaban akhir ditempatkan **setelah** tag penutup. Saat inference, model akan menghasilkan keduanya sekaligus dan dipisahkan oleh `_split_cot_answer()` di backend.

---

### Step 4 — Load & Proses Dataset

```python
# Baca file JSONL
raw_data: list[dict] = []
with open(cfg.dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_data.append(json.loads(line))

# Cek statistik oracle (apakah gold document selalu hadir)
oracle_true  = sum(1 for d in raw_data if d.get("metadata_extra", {}).get("oracle_present", True))
oracle_false = len(raw_data) - oracle_true
# Rekomendasi: oracle_absent ≥ 20% agar model belajar menolak konteks tidak relevan

# Tokenisasi semua sampel
hf_dataset = Dataset.from_list(raw_data)
hf_dataset = hf_dataset.map(tokenize_sample, remove_columns=hf_dataset.column_names)

# Train/Eval split (95% / 5%)
split = hf_dataset.train_test_split(test_size=cfg.test_size, seed=cfg.seed)
train_dataset = split["train"]
eval_dataset  = split["test"]
```

**Statistik dataset yang dikontrol:**
- Total sampel: ~179 record
- Oracle present vs oracle absent: dicek otomatis, disarankan ≥ 20% oracle absent
- Train/Eval split: 95% / 5% (deterministik dengan `seed=42`)

---

### Step 5 — Training dengan SFTTrainer

```python
training_args = SFTConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,           # 3 epoch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,                   # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",                      # cosine decay
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16 / bf16=True,                                # auto-detect hardware
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataset_text_field="text",
    max_seq_length=4096,
    packing=False,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    args=training_args,
)

trainer_stats = trainer.train()
```

**Ringkasan Hyperparameter Training:**
| Hyperparameter | Nilai | Alasan |
|---|---|---|
| `num_train_epochs` | 3 | Default awal; disesuaikan menjadi 15 pada versi akhir |
| `learning_rate` | 2e-4 | Standar QLoRA; lebih tinggi dari full fine-tuning |
| `lr_scheduler_type` | cosine | Decay halus; mencegah drop loss mendadak di akhir |
| `warmup_ratio` | 0.05 | Pemanasan 5% dari total steps |
| `weight_decay` | 0.01 | Regularisasi L2; dinaikkan ke 0.02 pada versi final |
| `gradient_accumulation_steps` | 4 | Efektif batch size = 8 tanpa memerlukan VRAM lebih besar |
| `load_best_model_at_end` | True | Otomatis kembali ke checkpoint terbaik (eval_loss terendah) |
| `packing` | False | Tidak digabung antar sampel (sampel panjang, mengandung CoT) |

---

### Step 6 — Simpan LoRA Adapter

```python
model.save_pretrained(cfg.output_dir)       # simpan adapter (bukan full model)
tokenizer.save_pretrained(cfg.output_dir)
# → output: model/raft-llama3-lora/
```

Output adapter terdiri dari:
- `adapter_config.json` — konfigurasi LoRA (r, alpha, target_modules, dll.)
- `adapter_model.safetensors` — bobot adapter yang telah dilatih
- File tokenizer (`tokenizer.json`, `special_tokens_map.json`, dll.)

---

### Step 7 — Merge LoRA ke Full Model (Deployment)

```python
MERGE_MODEL = True  # set False jika hanya butuh adapter

if MERGE_MODEL:
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(
        cfg.merged_output_dir,
        tokenizer,
        save_method="merged_16bit",  # opsi: "merged_4bit", "lora"
    )
    # → output: ./raft-llama3-merged/
```

**Perbedaan opsi `save_method`:**
| Opsi | Deskripsi | Ukuran File |
|---|---|---|
| `"merged_16bit"` | Fuse adapter + base model dalam format BF16 | ~16 GB |
| `"merged_4bit"` | Fuse dalam format 4-bit (lebih hemat, sedikit presisi turun) | ~5 GB |
| `"lora"` | Simpan hanya adapter (butuh base model saat loading) | ~50-100 MB |

Model merged digunakan oleh backend (`llama_service.py`) sebagai `model_merged_raft_perdes/`.

---

### Step 8 — Quick Test Inference (Sanity Check)

Setelah training, dilakukan uji cepat untuk memvalidasi output model:

```python
test_sample = {
    "instruction": "Apa yang dimaksud dengan APBDes?",
    "documents": [
        "pasal 1\n\n9. anggaran pendapatan dan belanja desa, selanjutnya disingkat apbdes ...",
        "pasal 20\n\nhal hal yang belum cukup diatur dalam peraturan desa ini ...",
        "pasal 5\n\nkepala desa bertugas menyelenggarakan pemerintahan desa.",
    ],
}

# Build prompt tanpa include_answer (mode inference)
messages = build_messages(test_sample, include_answer=False)
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Generate
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    use_cache=True,
)

# Decode hanya token baru (output model saja, bukan prompt)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
```

**Yang divalidasi pada quick test:**
- Model berhasil mengidentifikasi Dokumen 1 sebagai gold document (berisi definisi APBDes)
- Output mengandung blok `<CoT>...</CoT>` diikuti jawaban akhir
- Tidak ada looping atau output tidak koheren

---
# Key Features
| Fitur | Deskripsi |
|---|---|
| **RAFT Fine-Tuning** | Model dilatih dengan format RAFT: menerima pertanyaan + beberapa dokumen (gold + distractor), belajar memilih dokumen relevan secara eksplisit |
| **Chain-of-Thought (CoT) Reasoning** | Model menghasilkan proses analisis dokumen sebelum memberikan jawaban akhir, meningkatkan transparansi dan akurasi |
| **Distractor Robustness** | Dataset training menyertakan dokumen distractor (irrelevant) untuk melatih model agar tidak mudah terpengaruh konteks yang tidak relevan |
| **Dual-Mode Inference** | Mendukung dua mode: *plain chat* (tanpa konteks) dan *RAG chat* (dengan dokumen konteks) |
| **Post-Processing Pipeline** | Otomatis memisahkan analisis CoT dari jawaban akhir, menghapus referensi "Dokumen N" dari output, dan memperkaya jawaban yang terlalu singkat |
| **Anti-Monoton Training** | Variasi format respons training (transition phrases & analysis openers) untuk mencegah output yang kaku dan robotic |
| **Multi-Model Support** | API dapat melayani tiga variasi model: base, RAFT fine-tuned, dan Q&A sederhana — bisa di-switch via `/api/load-model` |
| **QLoRA Efficiency** | Kuantisasi 4-bit NF4 + LoRA hanya melatih ~1-2% parameter total, memungkinkan fine-tuning 8B model pada GPU consumer |
| **REST API** | Endpoint Flask yang bersih dengan CORS support untuk integrasi mudah dengan sistem RAG atau frontend apapun |
| **System Prompt Consistency** | System prompt saat training **identik** dengan inference untuk mencegah train-inference mismatch |
---
# Database Design
Proyek ini tidak menggunakan relational database. Data dikelola dalam format **JSONL (JSON Lines)** flat-file.
### Dataset: `raft_perdes_dataset.jsonl`
Setiap baris merepresentasikan **satu sampel training** RAFT:
| Field | Tipe | Deskripsi |
|---|---|---|
| `instruction` | `string` | Pertanyaan tentang Peraturan Desa |
| `documents` | `list[string]` | List dokumen konteks (1 gold + 2 distractor, posisi diacak) |
| `thought_process` | `string` | Chain-of-Thought: analisis relevansi setiap dokumen |
| `completion` | `string` | Jawaban akhir yang benar berdasarkan gold document |
**Contoh record:**
```json
{
  "instruction": "Apa rentang usia bayi menurut Perdes?",
  "documents": [
    "pasal 1\n\n15. bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari",
    "pasal 1\n\n9. pembangunan desa adalah upaya peningkatan kualitas hidup...",
    "pasal 1\n\n25. pemerintah pusat selanjutnya disebut pemerintah..."
  ],
  "thought_process": "Dokumen 1 relevan karena mendefinisikan bayi...",
  "completion": "Rentang usia bayi adalah 0 bulan sampai 11 bulan 28 hari."
}
```
### Dataset: `raft_perdes_dataset_improved.jsonl`
Versi yang sudah dibersihkan — referensi eksplisit "Dokumen N" dihapus dari field `completion` untuk meningkatkan naturalness jawaban saat inference.
---
# API / Integration
### REST API Endpoints (Base URL: `http://<server>:6000`)
| Method | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/api/chat` | Jawab pertanyaan **tanpa** dokumen konteks (plain LLM) |
| `POST` | `/api/chat-rag` | Jawab pertanyaan **dengan** dokumen konteks RAFT (utama) |
| `POST` | `/api/load-model` | Load / ganti model ke memori GPU secara manual |
| `GET` | `/api/model-info` | Info model yang sedang aktif (path, tipe, GPU info) |
| `GET` | `/api/models` | List semua model yang tersedia di server |
| `GET` | `/api/health` | Health check + daftar semua endpoint |
### Contoh Request `/api/chat-rag`
```json
POST /api/chat-rag
Content-Type: application/json
{
  "pertanyaan": "Apa rentang usia bayi menurut Perdes Biru No. 07/2015?",
  "dokumen": [
    "pasal 1\n\n15. bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari",
    "pasal 1\n\n9. pembangunan desa adalah upaya peningkatan kualitas hidup...",
    "pasal 1\n\n25. pemerintah pusat selanjutnya disebut pemerintah..."
  ]
}
```
### Contoh Response `/api/chat-rag`
```json
{
  "status": "success",
  "pertanyaan": "Apa rentang usia bayi menurut Perdes Biru No. 07/2015?",
  "analisis": "Dokumen 1 relevan karena mendefinisikan 'bayi'...",
  "jawaban": "Rentang usia bayi adalah 0 bulan sampai 11 bulan 28 hari.",
  "raw_response": "...",
  "model_type": "raft",
  "model_path": "/path/to/model_merged_raft_perdes",
  "num_documents": 3
}
```
### Integrasi Eksternal & Hardware
| Komponen | Detail |
|---|---|
| **GPU (CUDA)** | Dibutuhkan NVIDIA GPU dengan VRAM >= 12GB untuk inference QLoRA 4-bit |
| **Meta Llama 3.1 8B** | Model diunduh secara lokal (offline) via `download_model.sh`, tidak memanggil API HuggingFace saat runtime |
| **Unsloth Runtime** | Library inference khusus untuk mempercepat generate Llama — digunakan sebagai drop-in replacement FastLanguageModel |
| **CORS Policy** | API dikonfigurasi `origins: "*"` sehingga dapat dipanggil dari aplikasi web manapun |
---
# Challenges & Solutions
| # | Tantangan | Solusi yang Diterapkan |
|---|---|---|
| 1 | **Train-Inference Mismatch** — System prompt berbeda antara training dan production menyebabkan model menghasilkan output yang tidak konsisten | System prompt di `finetune_lora_rag.py` dan `llama_service.py` dibuat **identik secara karakter** dan didokumentasikan eksplisit sebagai `RAFT_SYSTEM_PROMPT` |
| 2 | **Output Monoton & Robotic** — Model cenderung menghasilkan respons dengan pola yang sama persis | Diterapkan *anti-monoton strategy*: variasi `TRANSITION_PHRASES` dan `ANALYSIS_OPENERS` yang dipilih secara deterministik per sampel (`random.Random(i * 3407 + 42)`) |
| 3 | **Referensi "Dokumen N" Bocor ke Jawaban** — Model menghasilkan jawaban seperti "Lihat Dokumen 1" yang tidak natural | Post-processing `_enrich_jawaban()` membersihkan semua referensi `Dokumen N` via regex, dan dataset diperbaiki menjadi `raft_perdes_dataset_improved.jsonl` |
| 4 | **Distractor Robustness Rendah** — Tanpa dokumen distractor, model mudah terpengaruh konteks tidak relevan | Dataset dirancang dengan format RAFT: 1 gold document + 2 distractor per sampel, posisi diacak. Referensi: Shi et al. (2023), Cuconasu et al. (2024) |
| 5 | **VRAM Terbatas** — Model 8B parameter membutuhkan GPU besar untuk fine-tuning penuh | Kombinasi **QLoRA (4-bit NF4)** + **Unsloth** + **gradient checkpointing** memungkinkan fine-tuning hanya ~1-2% parameter trainable |
| 6 | **Overfitting pada Dataset Kecil** — Dataset ~179 sampel rentan overfitting | Epoch dikurangi (20 -> 15), weight decay dinaikkan (0.01 -> 0.02), ditambah variasi format data untuk regularisasi implisit |
| 7 | **Jawaban Terlalu Singkat Pasca-CoT Parsing** — Setelah dipisah dari CoT, jawaban final kadang hanya fragmentasi kalimat | `_split_cot_answer()` menerapkan validasi: jika jawaban < 10 karakter, mundur ke separator sebelumnya |
---
# My Contributions
Sebagai pengembang utama pada proyek Tugas Akhir ini, kontribusi meliputi:
1. **Perancangan Arsitektur Sistem End-to-End** — Merancang keseluruhan pipeline dari dataset preparation, fine-tuning, hingga deployment via REST API, memastikan konsistensi format antara training dan inference.
2. **Implementasi Training Pipeline RAFT** — Mengimplementasikan pipeline fine-tuning dengan pendekatan RAFT menggunakan QLoRA + Unsloth, termasuk konfigurasi hyperparameter yang optimal (LoRA r=16, lr=2e-4, epoch=15) dan strategi anti-overfitting.
3. **Dataset Engineering** — Merancang dan menyiapkan dataset `raft_perdes_dataset.jsonl` dalam format RAFT yang menyertakan dokumen gold dan distractor, terinspirasi dari jurnal-jurnal terkini (Shi et al. 2023, Cuconasu et al. 2024, Gao et al. 2024).
4. **Anti-Monoton Training Strategy** — Merancang mekanisme variasi format respons secara deterministik untuk mencegah model menghasilkan output yang kaku dan seragam.
5. **Pembangunan Backend API** — Membangun REST API Flask yang bersih dan terdokumentasi dengan baik, mencakup endpoint untuk inference, model switching, health check, dan model listing.
6. **Post-Processing Pipeline** — Mengembangkan pipeline pembersihan output model: pemisahan Chain-of-Thought dari jawaban akhir, penghapusan referensi dokumen eksplisit, dan pengayaan jawaban yang kurang substantif.
7. **Optimasi & Debugging** — Mengidentifikasi dan menyelesaikan masalah train-inference mismatch, output degradation akibat second-pass generation, dan ketidaksesuaian hyperparameter antara training dan inference.
---
# Conclusion
Proyek ini berhasil mengembangkan sistem Q&A berbasis LLM yang mampu menjawab pertanyaan hukum tentang Peraturan Desa dengan akurat dan berdasarkan konteks dokumen yang relevan. Dengan menggabungkan fine-tuning RAFT, teknik QLoRA yang efisien, dan pipeline post-processing yang robust, sistem ini memberikan manfaat ganda: di satu sisi meningkatkan aksesibilitas hukum desa bagi masyarakat, di sisi lain membuktikan feasibilitas fine-tuning LLM skala besar pada hardware terbatas untuk domain hukum lokal Indonesia yang spesifik.
