"""
=============================================================================
Fine-Tuning Llama 3.1 8B Instruct dengan LoRA untuk RAG Peraturan Desa
=============================================================================

Tujuan:
  Fine-tune model Llama 3.1 8B Instruct menggunakan LoRA (Low-Rank Adaptation)
  agar model mampu menjawab pertanyaan berdasarkan konteks peraturan desa yang
  diberikan (RAG-aware). Model yang telah di-fine-tune kemudian dibandingkan
  dengan RAG vanilla (base model + RAG tanpa fine-tune).

Pendekatan:
  - QLoRA: Kuantisasi 4-bit + LoRA untuk efisiensi memori
  - Unsloth: Library untuk mempercepat fine-tuning 2x lebih cepat
  - Format data: instruction (konteks RAG) + input (pertanyaan) -> output (jawaban)

Referensi Jurnal:
  [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
      & Chen, W. (2022). "LoRA: Low-Rank Adaptation of Large Language Models."
      In ICLR 2022. https://arxiv.org/abs/2106.09685

  [2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
      "QLoRA: Efficient Finetuning of Quantized Language Models."
      In NeurIPS 2023. https://arxiv.org/abs/2305.14314

  [3] Lewis, P., Perez, E., Piktus, A., et al. (2020). "Retrieval-Augmented
      Generation for Knowledge-Intensive NLP Tasks." In NeurIPS 2020.
      https://arxiv.org/abs/2005.11401

  [4] Gao, Y., Xiong, Y., Gao, X., et al. (2024). "Retrieval-Augmented
      Generation for Large Language Models: A Survey." arXiv:2312.10997.
      https://arxiv.org/abs/2312.10997

  [5] Zhang, S., Dong, L., Li, X., et al. (2023). "Instruction Tuning for
      Large Language Models: A Survey." arXiv:2308.10792.
      https://arxiv.org/abs/2308.10792

  [6] Shi, F., Chen, X., Misra, K., et al. (2024). "Large Language Models Can
      Be Easily Distracted by Irrelevant Context." In ICML 2023.
      https://arxiv.org/abs/2302.00093

  [7] Chen, Z., Steiner, A., & Kumar, A. (2024). "RAG vs Fine-tuning:
      Pipelines, Tradeoffs, and a Case Study on Agriculture."
      arXiv:2401.08406. https://arxiv.org/abs/2401.08406

  [8] Zhang, T., Shao, F., Garg, S., et al. (2024). "RAFT: Adapting Language
      Model to Domain Specific RAG." In Transactions on ML Research.
      https://arxiv.org/abs/2403.10131

=============================================================================
"""

# ============================================================================
# STEP 0: Instalasi Dependencies
# ============================================================================
# Jalankan command berikut sebelum menjalankan script:
#
#   pip install torch transformers accelerate peft bitsandbytes datasets trl unsloth pandas
#
# Pastikan CUDA driver sudah terinstal dan kompatibel dengan PyTorch.
# ============================================================================

# ============================================================================
# STEP 1: Cek Lingkungan (GPU & Library)
# ============================================================================
import torch
import transformers

print("=" * 60)
print("CEK LINGKUNGAN")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU count       : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}          : {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory   : {props.total_memory / 1024**3:.2f} GB")

print(f"Transformers ver: {transformers.__version__}")
print("=" * 60)

assert torch.cuda.is_available(), "CUDA tidak tersedia! Pastikan GPU terdeteksi."

# ============================================================================
# STEP 2: Load Model dengan Unsloth + QLoRA (4-bit)
# ============================================================================
"""
Referensi:
  - Hu et al. (2022) [1]: LoRA menggunakan low-rank decomposition untuk
    mengurangi parameter yang perlu dilatih dari ~8B menjadi ~42 juta (0.5%).
  - Dettmers et al. (2023) [2]: QLoRA mengombinasikan kuantisasi 4-bit NF4
    dengan LoRA untuk fine-tuning model 65B pada GPU 48GB.
"""
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 2048  # RAFT samples: ~600-800 tokens instruction + input + output
DTYPE = None            # Auto-detect: Float16/BFloat16
LOAD_IN_4BIT = True     # QLoRA: kuantisasi 4-bit NF4

# BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # remote HuggingFace
BASE_MODEL_NAME = "../model/Meta-Llama-3.1-8B-Instruct"  # local path (lebih cepat, tidak perlu download)

print("\n[STEP 2] Loading model dengan QLoRA (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    device_map="auto",
    # token tidak diperlukan jika load dari path lokal
)

# Konfigurasi LoRA
# r=16: rank LoRA (Hu et al. menyarankan 8-64, 16 adalah sweet spot)
# target_modules: semua linear layer (Q, K, V, O, Gate, Up, Down)
# lora_alpha=16: scaling factor (biasanya = r atau 2*r)
# lora_dropout=0: dropout 0 untuk performa optimal (Hu et al.)
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Hemat VRAM
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
print(f"{'='*60}")

# ============================================================================
# STEP 3: Siapkan Dataset Fine-Tuning (RAFT Format)
# ============================================================================
"""
Format Data RAFT (Retrieval Augmented Fine-Tuning):
  Referensi: Zhang et al. (2024) [8] - "RAFT: Adapting Language Model to
  Domain Specific RAG"

  Setiap sampel berisi:
  - instruction: System prompt + dokumen konteks (1 GOLD + N DISTRACTOR)
    - GOLD_DOCUMENT = sumber jawaban yang benar (simulasi top-1 retrieval)
    - DISTRACTOR    = dokumen tidak relevan (simulasi retrieval noise)
    - Posisi gold diacak (tidak selalu di posisi 1)
  - input: Pertanyaan user
  - output: Jawaban yang di-grounding HANYA dari GOLD_DOCUMENT
  - metadata: Tracking info (gold_doc_id, gold_position, dll)

  Tujuan RAFT:
  - Model belajar mengidentifikasi dokumen relevan dari sekumpulan dokumen
  - Model belajar mengabaikan distractor (noise dari retriever)
  - Model menghasilkan jawaban yang faithful ke gold document
"""
import json
from datasets import Dataset

DATA_PATH = "../data/training_samples_raft_normal.json"

print("\n[STEP 3] Memuat dataset RAFT...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

data = []
for item in raw_data:
    data.append({
        "instruction": item["instruction"],  # system prompt + all docs
        "input": item["input"],              # question
        "output": item["output"],            # answer from gold doc
    })

dataset = Dataset.from_list(data)
print(f"Jumlah sampel training: {len(dataset)}")

# Statistik distractor
n_distractors = [item["metadata"]["n_distractors"] for item in raw_data]
gold_positions = [item["metadata"]["gold_position"] for item in raw_data]
villages = set(item["metadata"]["village_name"] for item in raw_data)
print(f"  Distractors per sample: {n_distractors[0]} (konsisten)")
print(f"  Gold positions: {sorted(set(gold_positions))}")
print(f"  Villages: {villages}")

# Tampilkan contoh data
print("\n--- CONTOH DATA TRAINING (RAFT) ---")
print(f"Instruction (system + docs):\n{dataset[0]['instruction'][:300]}...")
print(f"\nInput (pertanyaan):\n{dataset[0]['input']}")
print(f"\nOutput (jawaban dari GOLD):\n{dataset[0]['output'][:200]}...")
print("-" * 40)

# ============================================================================
# STEP 4: Format Data ke Template Chat Llama 3.1 (RAFT)
# ============================================================================
"""
RAFT data sudah memiliki struktur yang jelas:
  instruction = system prompt + [DOKUMEN 1..N] (gold + distractors)
  input       = pertanyaan user
  output      = jawaban dari gold document

Template chat Llama 3.1:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  {system_prompt_part}<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  {documents + pertanyaan}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  {jawaban}<|eot_id|>

Kita memisahkan system prompt dari dokumen+question agar model belajar:
  1. System role: "Anda adalah asisten hukum..."
  2. User role: dokumen konteks + pertanyaan
  3. Assistant role: jawaban yang grounded ke gold document
"""
EOS_TOKEN = tokenizer.eos_token


def extract_system_and_docs(instruction_text: str) -> tuple:
    """
    Pisahkan system prompt dari dokumen konteks.

    Input:
      "Anda adalah asisten hukum...\\n\\n...=== DOKUMEN KONTEKS ===\\n..."

    Output:
      system_prompt = "Anda adalah asisten hukum..."
      docs_section  = "=== DOKUMEN KONTEKS ===\\n..."
    """
    # Cari batas antara system prompt dan dokumen
    marker = "=== DOKUMEN KONTEKS ==="
    idx = instruction_text.find(marker)

    if idx >= 0:
        system_prompt = instruction_text[:idx].strip()
        docs_section = instruction_text[idx:].strip()
    else:
        # Fallback: gunakan seluruh instruction sebagai user content
        system_prompt = ""
        docs_section = instruction_text.strip()

    return system_prompt, docs_section


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instr, inp, out in zip(instructions, inputs, outputs):
        system_prompt, docs_section = extract_system_and_docs(instr)

        if system_prompt:
            # Format dengan system + user + assistant
            text = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{docs_section}\n\n{inp}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{out}{EOS_TOKEN}"
            )
        else:
            # Fallback: semua di user turn
            text = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{instr}\n\n{inp}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{out}{EOS_TOKEN}"
            )
        texts.append(text)

    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

print("\n[STEP 4] Contoh data setelah formatting (RAFT):")
print(dataset[0]["text"][:600])

# ============================================================================
# STEP 5: Konfigurasi & Jalankan Training
# ============================================================================
"""
Hyperparameter yang digunakan (berdasarkan rekomendasi jurnal):

  - learning_rate = 2e-4: Standar untuk LoRA (Hu et al. [1])
  - batch_size = 2, grad_accum = 4: Effective batch size = 8
  - max_steps: Disesuaikan dengan ukuran dataset
  - epochs: 6-8 epochs untuk dataset kecil (Chen et al. [7])
  - optimizer: adamw_8bit untuk efisiensi memori (Dettmers et al. [2])
  - lr_scheduler: Linear decay (standar untuk LoRA)
  - warmup_steps: 5 steps untuk stabilitas awal
"""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

LEARNING_RATE = 2e-4
NUM_EPOCHS = 8
WARMUP_STEPS = 5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4

# Hitung max_steps berdasarkan jumlah epoch
steps_per_epoch = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)) + 1
MAX_STEPS = steps_per_epoch * NUM_EPOCHS

print(f"\n[STEP 5] Konfigurasi Training:")
print(f"  Learning rate     : {LEARNING_RATE}")
print(f"  Num epochs        : {NUM_EPOCHS}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Grad accum steps  : {GRAD_ACCUM_STEPS}")
print(f"  Effective batch   : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Max steps         : {MAX_STEPS}")
print(f"  BF16 supported    : {is_bfloat16_supported()}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_lora_rag",
        report_to="none",
        save_strategy="no",  # Hindari pickling error pada SFTConfig; simpan manual di STEP 6
    ),
)

print("\n>>> Memulai training...")
trainer_stats = trainer.train()
print(">>> Fine-tuning selesai!")

# Cetak statistik training
print(f"\n{'='*60}")
print("STATISTIK TRAINING")
print(f"{'='*60}")
print(f"  Total steps     : {trainer_stats.metrics.get('train_steps_total', 'N/A')}")
print(f"  Train loss      : {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
print(f"  Train runtime   : {trainer_stats.metrics.get('train_runtime', 0)/60:.1f} menit")
print(f"{'='*60}")

# ============================================================================
# STEP 6: Simpan Adaptor LoRA & Merge ke Base Model
# ============================================================================
LORA_ADAPTER_DIR = "lora_adapter_rag_perdes"
MERGED_MODEL_DIR = "model_merged_rag_perdes"

print("\n[STEP 6] Menyimpan adaptor LoRA...")
model.save_pretrained(LORA_ADAPTER_DIR)
tokenizer.save_pretrained(LORA_ADAPTER_DIR)
print(f"  Adaptor LoRA disimpan di: {LORA_ADAPTER_DIR}")

print("\n[STEP 6b] Merging adaptor ke base model (16-bit)...")
model.save_pretrained_merged(MERGED_MODEL_DIR, tokenizer, save_method="merged_16bit")
print(f"  Model merged disimpan di: {MERGED_MODEL_DIR}")

# ============================================================================
# STEP 7: Uji Inferensi Model Fine-Tuned
# ============================================================================
from unsloth import FastLanguageModel as FLM

print("\n[STEP 7] Menguji model fine-tuned...")
FLM.for_inference(model)

# Contoh pertanyaan dengan konteks RAFT (GOLD + DISTRACTOR)
SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Kabupaten Bandung.\n\n"
    "Di bawah ini terdapat beberapa dokumen peraturan. Satu dokumen adalah "
    "GOLD_DOCUMENT (sumber jawaban yang benar), sisanya adalah DISTRACTOR "
    "(dokumen tidak relevan yang sengaja disertakan). Gunakan HANYA informasi "
    "dari GOLD_DOCUMENT untuk menjawab pertanyaan."
)

test_cases = [
    {
        "instruction": (
            f"{SYSTEM_PROMPT}\n\n"
            "=== DOKUMEN KONTEKS ===\n\n"
            "[DOKUMEN 1] [GOLD_DOCUMENT]\n"
            "Judul  : Peraturan Desa Biru No. 07 Tahun 2015 - Kibbla\n"
            "Bagian : pasal 1\n"
            "Desa   : Biru\n\n"
            "pasal 1\n\n"
            "15. bayi adalah anak usia 0 bulan sampai dengan 11 bulan 28 hari\n\n"
            "---\n\n"
            "[DOKUMEN 2] [DISTRACTOR]\n"
            "Judul  : Peraturan Desa Loa, No. 114 Tahun 2014\n"
            "Bagian : pasal 6\n"
            "Desa   : Loa\n\n"
            "pasal 6\n\n"
            "1. air bersih berskala desa;\n\n"
            "=== AKHIR DOKUMEN KONTEKS ==="
        ),
        "input": "Apa yang diatur dalam Peraturan Desa Biru No. 07 Tahun 2015 mengenai pasal 1?",
    },
    {
        "instruction": (
            f"{SYSTEM_PROMPT}\n\n"
            "=== DOKUMEN KONTEKS ===\n\n"
            "[DOKUMEN 1] [DISTRACTOR]\n"
            "Judul  : Peraturan Desa Biru No. 07 Tahun 2015 - Kibbla\n"
            "Bagian : pasal 2\n"
            "Desa   : Biru\n\n"
            "pasal 2\n\n"
            "kesehatan ibu, bayi baru lahir, bayi dan anak balita berasaskan nilai ilmiah.\n\n"
            "---\n\n"
            "[DOKUMEN 2] [GOLD_DOCUMENT]\n"
            "Judul  : Peraturan Desa Loa, No. 114 Tahun 2014\n"
            "Bagian : pasal 1\n"
            "Desa   : Loa\n\n"
            "pasal 1\n\n"
            "24. lembaga adat desa adalah lembaga yang menyelenggarakan fungsi adat istiadat dan "
            "menjadi bagian dari susunan asli desa yang tumbuh dan berkembang atas prakarsa masyarakat desa.\n\n"
            "=== AKHIR DOKUMEN KONTEKS ==="
        ),
        "input": "Menurut Peraturan Desa Loa No. 114 Tahun 2014, apa yang dimaksud lembaga adat desa?",
    },
]

for i, tc in enumerate(test_cases):
    sys_prompt, docs = extract_system_and_docs(tc["instruction"])
    user_content = f"{docs}\n\n{tc['input']}"

    if sys_prompt:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": f"{tc['instruction']}\n\n{tc['input']}"}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"\n--- Test {i+1} ---")
    print(f"Pertanyaan  : {tc['input']}")
    print(f"Jawaban     : {response}")

print("\n" + "=" * 60)
print("FINE-TUNING SELESAI!")
print(f"  LoRA adapter : {LORA_ADAPTER_DIR}")
print(f"  Merged model : {MERGED_MODEL_DIR}")
print("=" * 60)
