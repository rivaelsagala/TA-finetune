"""
=============================================================================
Fine-Tuning Llama 3.1 8B Instruct dengan LoRA - RAFT (Retrieval-Augmented Fine-Tuning)
=============================================================================

Tujuan:
  Fine-tune model Llama 3.1 8B Instruct menggunakan LoRA dengan pendekatan
  RAFT (Retrieval-Augmented Fine-Tuning) pada dataset raft_perdes_dataset.jsonl
  berisi Q&A + dokumen konteks (gold + distractor) dari Peraturan Desa.

  Model hasil fine-tuning ini akan dibandingkan dengan:
    1. Model base + RAG vanilla (tanpa fine-tune)
    2. Model Q&A sederhana (fine-tuned tanpa konteks dokumen)

Pendekatan:
  - RAFT: Model dilatih dengan dokumen konteks (gold + distractor) agar
    belajar memilih dokumen relevan dan mengabaikan distractor
  - LoRA full precision (BF16): VRAM B200 cukup, tidak perlu QLoRA 4-bit
  - Unsloth: Library untuk mempercepat fine-tuning 2x lebih cepat
  - Format data: instruction + documents -> thought_process + completion

Referensi Jurnal:
  [1] Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language
      Models." In ICLR 2022. https://arxiv.org/abs/2106.09685
  [2] Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized
      Language Models." In NeurIPS 2023. https://arxiv.org/abs/2305.14314
  [3] Zhang, T., et al. (2024). "RAFT: Adapting Language Model to Domain
      Specific RAG." In TMLR. https://arxiv.org/abs/2403.10131

PERUBAHAN dari versi sebelumnya:
  - LOAD_IN_4BIT = False  : B200 punya 191 GB VRAM, tidak perlu QLoRA
  - DTYPE = bfloat16      : B200 support BF16 native, lebih stabil dari FP16
  - BATCH_SIZE = 16       : Manfaatkan VRAM besar (effective batch = 32)
  - SFTConfig             : Gantikan TrainingArguments (API TRL >= 0.9.0)
  - Hapus dataset_text_field dari SFTTrainer (pindah ke SFTConfig)
  - Tambah save_steps + save_total_limit : Checkpoint berkala agar aman
  - Tambah logging_dir    : TensorBoard logging
  - Perbaiki statistik training (key metrics yang benar)

=============================================================================
"""

# ============================================================================
# STEP 1: Cek Lingkungan (GPU & Library)
# ============================================================================
import torch
import transformers
import trl

print("=" * 60)
print("CEK LINGKUNGAN")
print("=" * 60)
print(f"PyTorch version   : {torch.__version__}")
print(f"Transformers ver  : {transformers.__version__}")
print(f"TRL version       : {trl.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")
print(f"GPU count         : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}            : {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory    : {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute cap     : {props.major}.{props.minor}")

print("=" * 60)

assert torch.cuda.is_available(), "CUDA tidak tersedia! Pastikan GPU terdeteksi."

# ============================================================================
# STEP 2: Load Model dengan Unsloth — Full BF16 (tanpa QLoRA)
# ============================================================================
# FIX: LOAD_IN_4BIT = False karena B200 punya 191 GB VRAM.
#      QLoRA 4-bit justru memperlambat di GPU besar karena overhead dequantization.
#      Llama 3.1 8B BF16 hanya butuh ~16 GB, jauh di bawah kapasitas.
# ============================================================================
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16   # FIX: Set eksplisit BF16 (B200 support native)
LOAD_IN_4BIT = False     # FIX: Matikan QLoRA, gunakan full precision BF16

BASE_MODEL_NAME = "../model/Meta-Llama-3.1-8B-Instruct"  # sesuaikan path jika perlu

print("\n[STEP 2] Loading model dengan BF16 (full precision, tanpa quantization)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    device_map="auto",
)

# Konfigurasi LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Hemat VRAM meski sudah besar
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Cetak info parameter
trainable, total = 0, 0
for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

print(f"\n{'='*60}")
print(f"Total parameters     : {total:,}")
print(f"Trainable (LoRA)     : {trainable:,}")
print(f"Rasio trainable      : {100*trainable/total:.2f}%")
print(f"LoRA rank (r)        : {LORA_R}")
print(f"LoRA alpha           : {LORA_ALPHA}")
print(f"Precision            : BF16 (full, tanpa quantization)")
print(f"{'='*60}")

# ============================================================================
# STEP 3: Siapkan Dataset RAFT (raft_perdes_dataset.jsonl)
# ============================================================================
"""
Format data raft_perdes_dataset.jsonl (RAFT format):
  Setiap baris berisi JSON object dengan field:
  - instruction    : Pertanyaan tentang Peraturan Desa
  - documents      : List dokumen konteks (1 gold + 2 distractor, posisi diacak)
  - thought_process: Chain-of-Thought reasoning (analisis relevansi dokumen)
  - completion     : Jawaban akhir berdasarkan dokumen yang relevan

  Contoh:
  {
    "instruction": "Apa rentang usia bayi?",
    "documents": ["pasal 1\n\n15. bayi adalah...", "pasal 1\n\n9. pembangunan...", ...],
    "thought_process": "Langkah analisis: evaluasi setiap dokumen...",
    "completion": "Rentang usia bayi adalah 0 bulan sampai 11 bulan 28 hari."
  }
"""
import json
from datasets import Dataset

DATA_PATH = "../data/dataset/raft_dataset_final.jsonl"

print("\n[STEP 3] Memuat dataset RAFT (raft_perdes_dataset.jsonl)...")
raw_data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

data = []
for item in raw_data:
    data.append({
        "instruction": item["instruction"],
        "documents": item["documents"],
        "thought_process": item["thought_process"],
        "completion": item["completion"],
    })

dataset = Dataset.from_list(data)
print(f"Jumlah sampel training: {len(dataset)}")

# Statistik dataset
avg_q_len    = sum(len(d["instruction"])    for d in data) / len(data)
avg_a_len    = sum(len(d["completion"])     for d in data) / len(data)
avg_doc_count = sum(len(d["documents"])     for d in data) / len(data)

print(f"  Rata-rata panjang pertanyaan : {avg_q_len:.0f} karakter")
print(f"  Rata-rata panjang jawaban    : {avg_a_len:.0f} karakter")
print(f"  Rata-rata jumlah dokumen     : {avg_doc_count:.1f} per sampel")

print("\n--- CONTOH DATA TRAINING (RAFT) ---")
for i in range(min(3, len(dataset))):
    print(f"\n  [{i+1}] Pertanyaan : {dataset[i]['instruction'][:100]}...")
    print(f"      # Dokumen   : {len(dataset[i]['documents'])}")
    print(f"      CoT         : {dataset[i]['thought_process'][:80]}...")
    print(f"      Jawaban     : {dataset[i]['completion'][:80]}...")
print("-" * 40)

# ============================================================================
# STEP 4: Format Data ke Template Chat Llama 3.1 (RAFT Format)
# ============================================================================
EOS_TOKEN = tokenizer.eos_token

SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Indonesia. Jawab pertanyaan berdasarkan "
    "dokumen-dokumen yang diberikan. Tidak semua dokumen relevan dengan "
    "pertanyaan, jadi pilihlah informasi dari dokumen yang paling sesuai."
)

# ============================================================================
# STRATEGI ANTI-MONOTON: Variasi format response saat training
# ============================================================================
import random

random.seed(3407)

TRANSITION_PHRASES = [
    "\n\n",
    "\n\nBerdasarkan analisis di atas, ",
    "\n\nDari uraian tersebut, ",
    "\n\nKesimpulannya, ",
    "\n\nDengan demikian, ",
    "\n\nJadi, ",
    "\n\nSebagai kesimpulan, ",
    "\n\nMaka dari itu, ",
]

ANALYSIS_OPENERS = [
    "",
    "Mari analisis dokumen-dokumen yang tersedia.\n",
    "Berikut analisis relevansi dokumen.\n",
    "Saya akan evaluasi setiap dokumen.\n",
]


def formatting_prompts_func(examples):
    instructions     = examples["instruction"]
    documents_list   = examples["documents"]
    thought_processes = examples["thought_process"]
    completions      = examples["completion"]
    texts = []

    for i, (instr, docs, cot, comp) in enumerate(
        zip(instructions, documents_list, thought_processes, completions)
    ):
        # Format dokumen sebagai numbered list
        docs_text = ""
        for idx, doc in enumerate(docs, 1):
            docs_text += f"\n\nDokumen {idx}:\n{doc}"

        user_message = f"{instr}{docs_text}"

        # Variasi deterministik per sampel
        rng = random.Random(i * 3407 + 42)
        transition      = rng.choice(TRANSITION_PHRASES)
        analysis_opener = rng.choice(ANALYSIS_OPENERS)

        assistant_response = f"{analysis_opener}{cot}{transition}{comp}"

        text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_response}{EOS_TOKEN}"
        )
        texts.append(text)

    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

print("\n[STEP 4] Contoh data setelah formatting (800 karakter pertama):")
print(dataset[0]["text"][:800])

# ============================================================================
# STEP 5: Konfigurasi & Jalankan Training
# ============================================================================
# FIX: Gunakan SFTConfig (bukan TrainingArguments) untuk TRL >= 0.9.0
#      SFTConfig menyatukan semua argumen SFT + training dalam satu objek.
#
# FIX: BATCH_SIZE dinaikkan 2 -> 16 (manfaatkan 191 GB VRAM B200).
#      Effective batch size = 16 * 2 = 32 (lebih representatif per update).
#
# FIX: Tambah save_steps & save_total_limit untuk checkpoint berkala.
#      Jika training crash di tengah jalan, bisa resume dari checkpoint.
#
# FIX: Tambah logging_dir untuk TensorBoard (opsional tapi direkomendasikan).
# ============================================================================
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 15
WARMUP_STEPS      = 5
BATCH_SIZE        = 16   # FIX: naik dari 2 -> 16 (B200 191 GB VRAM)
GRAD_ACCUM_STEPS  = 2    # FIX: turun dari 4 -> 2, effective batch tetap 32
WEIGHT_DECAY      = 0.02

steps_per_epoch = max(1, len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS))
MAX_STEPS       = steps_per_epoch * NUM_EPOCHS

print(f"\n[STEP 5] Konfigurasi Training:")
print(f"  Dataset size       : {len(dataset)} sampel")
print(f"  Learning rate      : {LEARNING_RATE}")
print(f"  Num epochs         : {NUM_EPOCHS}")
print(f"  Batch size         : {BATCH_SIZE}")
print(f"  Grad accum steps   : {GRAD_ACCUM_STEPS}")
print(f"  Effective batch    : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Steps per epoch    : {steps_per_epoch}")
print(f"  Max steps          : {MAX_STEPS}")
print(f"  BF16 supported     : {is_bfloat16_supported()}")
print(f"  Precision          : {'BF16' if is_bfloat16_supported() else 'FP16'}")

# FIX: SFTConfig menggantikan TrainingArguments + parameter SFT lama
sft_config = SFTConfig(
    # -- Parameter SFT (dulu di SFTTrainer) --
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,

    # -- Parameter Training --
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs_lora",
    logging_dir="outputs_lora/tensorboard_logs",  # FIX: TensorBoard logging
    report_to="none",

    # FIX: Checkpoint berkala agar aman jika training crash
    save_strategy="steps",
    save_steps=steps_per_epoch,       # Simpan tiap 1 epoch
    save_total_limit=3,               # Simpan maksimal 3 checkpoint terakhir
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,  # FIX: semua argumen masuk lewat SFTConfig
)

print("\n>>> Memulai training...")
trainer_stats = trainer.train()
print(">>> Fine-tuning selesai!")

# Cetak statistik training
# FIX: Gunakan key yang benar dari trainer_stats.metrics
metrics = trainer_stats.metrics
print(f"\n{'='*60}")
print("STATISTIK TRAINING")
print(f"{'='*60}")
print(f"  Train loss      : {metrics.get('train_loss', 'N/A')}")
print(f"  Train runtime   : {metrics.get('train_runtime', 0)/60:.1f} menit")
print(f"  Samples/sec     : {metrics.get('train_samples_per_second', 'N/A')}")
print(f"  Steps/sec       : {metrics.get('train_steps_per_second', 'N/A')}")
print(f"{'='*60}")

# ============================================================================
# STEP 6: Simpan Adaptor LoRA & Merge ke Base Model
# ============================================================================
LORA_ADAPTER_DIR = "lora_adapter_raft_perdes"
MERGED_MODEL_DIR = "model_merged_raft_perdes"

print("\n[STEP 6] Menyimpan adaptor LoRA...")
model.save_pretrained(LORA_ADAPTER_DIR)
tokenizer.save_pretrained(LORA_ADAPTER_DIR)
print(f"  Adaptor LoRA disimpan di: {LORA_ADAPTER_DIR}/")

print("\n[STEP 6b] Merging adaptor ke base model (BF16)...")
# FIX: save_method="merged_16bit" tetap benar untuk BF16
model.save_pretrained_merged(MERGED_MODEL_DIR, tokenizer, save_method="merged_16bit")
print(f"  Model merged disimpan di: {MERGED_MODEL_DIR}/")

# ============================================================================
# STEP 7: Uji Inferensi Model Fine-Tuned (RAFT Format)
# ============================================================================
from unsloth import FastLanguageModel as FLM

print("\n[STEP 7] Menguji model fine-tuned dengan format RAFT...")
FLM.for_inference(model)

test_cases = [
    {
        "question": "Apa yang dimaksud dengan keputusan kepala desa menurut Peraturan Desa Biru No. 07 Tahun 2015?",
        "documents": [
            "pasal 12\n\n(3) jumlah tim sebagaimana dimaksud pada ayat (1), paling sedikit 7 (tujuh) dan paling banyak 11 (sebelas) orang.",
            "pasal 32\n\n(5) dalam hal pembahasan dalam musyawarah desa sebagaimana dimaksud pada ayat (4) tidak\nmenyepakati teknis pelaksanaan program sektor dan/atau program daerah, kepala desa dapat\nmengajukan keberatan atas bagian dari teknis pelaksanaan yang tidak disepakati, disertai dasar\npertimbangan keberatan dimaksud.",
            "pasal 1\n\n4. keputusan kepala desa adalah keputusan yang dibuat oleh kepala desa\n\nsebagai tindak lanjut peraturan desa atau ketentuan lain yang bersifat",
        ],
    },
    {
        "question": "Siapa saja yang mengikuti musyawarah perencanaan pembangunan desa?",
        "documents": [
            "pasal 8\n\n2. untuk melaksanakan tugas sebagaimana dimaksud dalam ayat 1 pasal 8 ini ketua tim\n\nkibbla mempunyai fungsi : 1. memimpin dan mengendalikan seluruh kegiatan tim kibbla.",
            "pasal 12\n\n4. fungsi khusus kordinator bidang;\n1. kordinator bidang data dan informasi;",
            "pasal 25\n\n(2) musyawarah perencanaan pembangunan desa sebagaimana dimaksud pada ayat (1) diikuti oleh\npemerintah desa, badan permusyawaratan desa, dan unsur masyarakat.",
        ],
    },
]

for i, test_case in enumerate(test_cases):
    docs_text = ""
    for idx, doc in enumerate(test_case["documents"], 1):
        docs_text += f"\n\nDokumen {idx}:\n{doc}"

    user_message = f"{test_case['question']}{docs_text}"

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        min_p=0.05,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\n--- Test {i+1} ---")
    print(f"Pertanyaan : {test_case['question']}")
    print(f"# Dokumen  : {len(test_case['documents'])}")
    print(f"Jawaban    :\n{response}")

print("\n" + "=" * 60)
print("FINE-TUNING RAFT SELESAI!")
print(f"  LoRA adapter : {LORA_ADAPTER_DIR}/")
print(f"  Merged model : {MERGED_MODEL_DIR}/")
print(f"{'='*60}")
print()
print("PERBANDINGAN MODEL:")
print(f"  1. Base model + RAG    : Meta-Llama-3.1-8B-Instruct (tanpa fine-tune)")
print(f"  2. Model RAFT (baru)   : {MERGED_MODEL_DIR}")
print(f"  3. Model Q&A sederhana : model_merged_perdes (dari finetune_lora_new.py)")
print("=" * 60)