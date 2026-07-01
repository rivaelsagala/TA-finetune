"""
=============================================================================
Fine-Tuning Llama 3.1 8B Instruct dengan LoRA + Unsloth
RAFT (Retrieval-Augmented Fine-Tuning) untuk Peraturan Desa
=============================================================================

Perubahan dari kode sebelumnya:
  [FIX]  bf16=True          : Typo fatal diperbaiki (sebelumnya False)
  [FIX]  Unsloth            : Ganti AutoModelForCausalLM -> FastLanguageModel
                               untuk 2x speedup otomatis
  [FIX]  Format dataset     : Gunakan messages + chat_template (bukan
                               dataset_text_field="messages" yang konflik)
  [FIX]  Hapus assistant_only_loss : Parameter tidak ada di SFTConfig saat ini
  [FIX]  Hapus attn_implementation : Unsloth handle optimasi attention sendiri
  [FIX]  optim adamw_8bit   : Lebih efisien dari adamw_torch_fused di B200
  [ADD]  Merge model        : Save full merged model 16-bit di akhir training
  [ADD]  Inference test     : Validasi model setelah training
  [ADD]  EarlyStoppingCallback: Tetap dipakai tapi dengan patience lebih besar
  [ADD]  Logging VRAM       : Monitor penggunaan memori GPU

  [FIX]  EOS_TOKEN          : Ganti messages format -> plain text f-string.
                               TRL baru validasi tokenizer.chat_template saat
                               processing_class= dipakai, dan ketemu placeholder
                               "<EOS_TOKEN>" (Unsloth inject) yang tidak ada di
                               vocabulary Llama 3.1. Solusi: build string manual
                               pakai tokenizer.eos_token ("<|eot_id|>") langsung,
                               kasih ke SFTTrainer sebagai dataset_text_field="text".
                               SFTTrainer tidak parse chat_template -> tidak error.

Arsitektur:
  - Model     : Llama 3.1 8B Instruct (full BF16, tanpa quantization)
  - LoRA      : r=32, alpha=64, semua projection layers
  - Optimizer : adamw_8bit + cosine scheduler
  - Dataset   : RAFT format (instruction + documents -> completion)
  - Validation: 10% split + EarlyStopping (patience=3)

Referensi:
  [1] Hu et al. (2022). LoRA. https://arxiv.org/abs/2106.09685
  [2] Zhang et al. (2024). RAFT. https://arxiv.org/abs/2403.10131
=============================================================================
"""

import json
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
MODEL_ID         = "../model/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH     = "../data/dataset/raft_dataset_final.jsonl"
OUTPUT_DIR       = "../model/llama-3.1-8b-raft-lora"
LORA_ADAPTER_DIR = f"{OUTPUT_DIR}/lora_adapter"
MERGED_MODEL_DIR = f"{OUTPUT_DIR}/merged_bf16"

# LoRA params
LORA_R           = 32
LORA_ALPHA       = 64
# LORA_DROPOUT = 0 wajib untuk Unsloth agar bisa fast-patch semua layer
# Jika > 0, Unsloth skip optimasi di LoRA layer -> performa turun

# Training params (optimal untuk B200 191 GB VRAM)
BATCH_SIZE             = 16
GRADIENT_ACCUMULATION  = 2    # Effective batch = 32
EPOCHS                 = 3
LEARNING_RATE          = 1e-4
MAX_SEQ_LENGTH         = 4096  # RAFT dokumen panjang
WARMUP_RATIO           = 0.03
MAX_GRAD_NORM          = 0.3
EARLY_STOP_PATIENCE    = 3     # Lebih besar dari 2 agar tidak stop terlalu cepat

# System prompt — konsisten antara training dan inference
SYSTEM_PROMPT = (
    "Anda adalah asisten AI ahli hukum desa yang sangat teliti.\n"
    "Gunakan HANYA informasi yang terdapat pada dokumen.\n"
    "Jangan menggunakan pengetahuan di luar dokumen.\n"
    "Jika dokumen tidak cukup, katakan informasi tidak ditemukan."
)


# ---------------------------------------------------------------------------
# STEP 1: FORMAT DATASET KE PLAIN TEXT
#
# FIX EOS_TOKEN: Kode sebelumnya pakai messages format + processing_class=tokenizer.
#   TRL baru membaca tokenizer.chat_template saat processing_class dipakai,
#   dan ketemu placeholder "<EOS_TOKEN>" yang di-inject Unsloth ke template.
#   Token itu tidak ada di vocabulary Llama 3.1 -> ValueError.
#
#   Solusi: Build string teks Llama 3.1 chat format secara manual pakai f-string.
#   EOS token diambil dari tokenizer.eos_token ("<|eot_id|>") bukan dari
#   placeholder string. Dataset dikembalikan sebagai plain text field "text".
#   SFTTrainer tidak perlu parse chat_template -> tidak ketemu "<EOS_TOKEN>".
# ---------------------------------------------------------------------------
def format_raft_prompt(sample: dict, eos_token: str) -> dict:
    """
    Build string teks Llama 3.1 chat format secara manual.

    eos_token diambil dari tokenizer.eos_token ("<|eot_id|>") yang
    memang ada di vocabulary — bukan placeholder "<EOS_TOKEN>" dari Unsloth.
    """
    docs_text = "\n\n".join([
        f"<document id={i+1}>\n{doc}\n</document>"
        for i, doc in enumerate(sample["documents"])
    ])

    user_content = (
        f"Question:\n{sample['instruction']}\n\n"
        f"Documents:\n{docs_text}"
    )

    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{sample['completion']}{eos_token}"
    )
    return {"text": text}


def prepare_dataset(file_path: str, eos_token: str) -> Dataset:
    """Load JSONL dan format ke plain text Dataset."""
    raw = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    formatted = [format_raft_prompt(s, eos_token) for s in raw]
    return Dataset.from_list(formatted)


# ---------------------------------------------------------------------------
# STEP 2: MAIN
# ---------------------------------------------------------------------------
def main():
    # -- CEK LINGKUNGAN --
    print("=" * 60)
    print("CEK LINGKUNGAN")
    print("=" * 60)
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print(f"BF16 support : {is_bfloat16_supported()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}        : {torch.cuda.get_device_name(i)}")
            print(f"  VRAM       : {props.total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    assert torch.cuda.is_available(), "CUDA tidak tersedia!"
    assert is_bfloat16_supported(), "BF16 tidak didukung GPU ini!"

    # -- LOAD MODEL dengan Unsloth --
    # Ganti AutoModelForCausalLM -> FastLanguageModel
    # Unsloth otomatis optimasi attention, tidak perlu attn_implementation
    # LOAD_IN_4BIT=False karena B200 punya 191 GB VRAM, tidak butuh QLoRA
    print(f"\n[STEP 1] Memuat model: {MODEL_ID}")
    print("         Menggunakan Unsloth FastLanguageModel (Full BF16, tanpa quantization)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,      # Full precision, B200 cukup VRAM
        device_map="auto",
    )

    # Pad token: Llama 3.1 punya <|finetune_right_pad_id|> sebagai dedicated pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    # FIX EOS_TOKEN: Unsloth inject placeholder "<EOS_TOKEN>" ke chat_template.
    # TRL baru (>= 0.9) validasi setiap token di chat_template terhadap vocabulary
    # saat processing_class=tokenizer dipakai. "<EOS_TOKEN>" tidak ada di vocabulary
    # Llama 3.1 -> ValueError. Solusi: override chat_template dengan official
    # Llama 3.1 template yang pakai <|eot_id|> (ada di vocabulary) sebagai EOS.
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    print(f"  EOS token  : {tokenizer.eos_token}")
    print(f"  PAD token  : {tokenizer.pad_token}")
    print(f"  BOS token  : {tokenizer.bos_token}")

    # -- KONFIGURASI LoRA --
    print("\n[STEP 2] Mengonfigurasi LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,       # 0 agar Unsloth bisa full fast-patch semua layer
                              # dropout > 0 membuat Unsloth skip optimasi di LoRA layer
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
        # Hapus task_type="CAUSAL_LM" — Unsloth sudah set ini secara
        # internal, jika di-pass manual akan TypeError: multiple values
    )

    # Print trainable params
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"  Total params     : {total:,}")
    print(f"  Trainable (LoRA) : {trainable:,} ({100*trainable/total:.2f}%)")

    # -- SIAPKAN DATASET --
    print(f"\n[STEP 3] Memuat dataset: {DATASET_PATH}")

    # FIX EOS_TOKEN: pass tokenizer.eos_token ke prepare_dataset.
    # Dataset langsung berisi plain text field "text" — tidak perlu map/formatting_func lagi.
    dataset = prepare_dataset(DATASET_PATH, tokenizer.eos_token)
    print(f"  Total sampel     : {len(dataset)}")

    # Tampilkan contoh
    print("\n--- CONTOH DATA SETELAH FORMATTING (800 karakter pertama) ---")
    print(dataset[0]["text"][:800])
    print("...[DIPOTONG]...")
    print("-" * 60)

    # Train/eval split
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"\n  Train : {len(dataset_split['train'])} sampel")
    print(f"  Eval  : {len(dataset_split['test'])} sampel")

    # -- KONFIGURASI TRAINING --
    steps_per_epoch = max(1, len(dataset_split["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION))
    total_steps     = steps_per_epoch * EPOCHS

    print(f"\n[STEP 4] Konfigurasi Training:")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Grad accum       : {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch  : {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Steps/epoch      : {steps_per_epoch}")
    print(f"  Total steps      : {total_steps}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Learning rate    : {LEARNING_RATE}")
    print(f"  Precision        : BF16")

    # Hitung warmup_steps dari ratio secara manual
    # warmup_ratio deprecated >= Transformers v5.2, ganti ke warmup_steps
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    print(f"  Warmup steps     : {warmup_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # -- Parameter SFT --
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,

        # -- Training hyperparams --
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        # -- Precision --
        bf16=True,
        fp16=False,

        # -- Logging & Saving --
        logging_steps=5,
        # logging_dir deprecated >= v5.2
        # Untuk TensorBoard jalankan: export TENSORBOARD_LOGGING_DIR=<path>
        save_strategy="epoch",
        save_total_limit=3,

        # -- Evaluasi & Early Stopping --
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # -- Reproducibility --
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # chat_template sudah di-patch, aman dipakai
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOP_PATIENCE
            )
        ],
    )

    # -- MULAI TRAINING --
    # Log VRAM sebelum training
    if torch.cuda.is_available():
        used       = torch.cuda.memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  VRAM sebelum training: {used:.1f} GB / {total_vram:.1f} GB")

    print("\n>>> Memulai training...")
    trainer_stats = trainer.train()
    print(">>> Training selesai!")

    # Log VRAM setelah training
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM setelah training: {used:.1f} GB / {total_vram:.1f} GB")

    # Cetak statistik
    metrics = trainer_stats.metrics
    print(f"\n{'='*60}")
    print("STATISTIK TRAINING")
    print(f"{'='*60}")
    print(f"  Train loss      : {metrics.get('train_loss', 'N/A')}")
    print(f"  Train runtime   : {metrics.get('train_runtime', 0)/60:.1f} menit")
    print(f"  Samples/sec     : {metrics.get('train_samples_per_second', 'N/A')}")
    print(f"  Steps/sec       : {metrics.get('train_steps_per_second', 'N/A')}")
    print(f"{'='*60}")

    # -- SIMPAN MODEL --
    print(f"\n[STEP 5] Menyimpan LoRA adapter -> {LORA_ADAPTER_DIR}")
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"  ✅ LoRA adapter disimpan")

    # Merge LoRA ke base model (full BF16) — untuk inference tanpa PEFT overhead
    print(f"\n[STEP 5b] Merging LoRA ke base model (BF16) -> {MERGED_MODEL_DIR}")
    model.save_pretrained_merged(MERGED_MODEL_DIR, tokenizer, save_method="merged_16bit")
    print(f"  ✅ Merged model disimpan")

    # -- INFERENCE TEST --
    print("\n[STEP 6] Uji Inferensi Model Fine-Tuned...")
    FastLanguageModel.for_inference(model)

    test_cases = [
        {
            "question": "Apa yang dimaksud dengan keputusan kepala desa?",
            "documents": [
                "pasal 12\n\n(3) jumlah tim paling sedikit 7 dan paling banyak 11 orang.",
                "pasal 32\n\n(5) kepala desa dapat mengajukan keberatan atas teknis pelaksanaan.",
                "pasal 1\n\n4. keputusan kepala desa adalah keputusan yang dibuat oleh kepala desa sebagai tindak lanjut peraturan desa.",
            ],
        },
        {
            "question": "Siapa saja yang mengikuti musyawarah perencanaan pembangunan desa?",
            "documents": [
                "pasal 8\n\n2. ketua tim kibbla memimpin dan mengendalikan seluruh kegiatan tim.",
                "pasal 12\n\n4. fungsi kordinator bidang data dan informasi.",
                "pasal 25\n\n(2) musyawarah diikuti oleh pemerintah desa, badan permusyawaratan desa, dan unsur masyarakat.",
            ],
        },
    ]

    for i, tc in enumerate(test_cases):
        docs_text = "\n\n".join([
            f"<document id={j+1}>\n{doc}\n</document>"
            for j, doc in enumerate(tc["documents"])
        ])
        user_content = f"Question:\n{tc['question']}\n\nDocuments:\n{docs_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
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
        print(f"Pertanyaan : {tc['question']}")
        print(f"Jawaban    :\n{response}")

    # -- RINGKASAN AKHIR --
    print("\n" + "=" * 60)
    print("FINE-TUNING RAFT SELESAI!")
    print(f"  LoRA adapter  : {LORA_ADAPTER_DIR}/")
    print(f"  Merged model  : {MERGED_MODEL_DIR}/")
    print("=" * 60)
    print("\nCara load model untuk inference:")
    print(f"""
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="{MERGED_MODEL_DIR}",
      max_seq_length={MAX_SEQ_LENGTH},
      dtype=torch.bfloat16,
      load_in_4bit=False,
  )
  FastLanguageModel.for_inference(model)
    """)


if __name__ == "__main__":
    main()