import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH = "../data/dataset/raft_dataset_final.jsonl"
OUTPUT_DIR = "../models/llama-3.1-8b-raft-lora"

# LoRA & Training Params (Cocok untuk GPU besar seperti B200 / A100)
LORA_R = 32
LORA_ALPHA = 64
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 4096 # Penting: karena dokumen RAFT panjang

# ---------------------------------------------------------------------------
# 1. LOAD & FORMAT DATASET KE LLAMA-3 CHAT TEMPLATE
# ---------------------------------------------------------------------------
def format_raft_prompt(sample):
    """
    Mengubah format RAFT menjadi format percakapan Llama 3.1
    Format RAFT: completion-only target, grounded pada dokumen yang diberikan.
    """
    sys_prompt = (
        "Anda adalah asisten AI ahli hukum desa yang sangat teliti.\n"
        "Gunakan HANYA informasi yang terdapat pada dokumen.\n"
        "Jangan menggunakan pengetahuan di luar dokumen.\n"
        "Jika dokumen tidak cukup, katakan informasi tidak ditemukan."
    )
    
    # Gabungkan 5 dokumen menjadi 1 string dengan delimiter yang kuat
    docs_text = "\n\n".join([f"<document id={i+1}>\n{doc}\n</document>" for i, doc in enumerate(sample["documents"])])
    
    user_prompt = f"Question:\n{sample['instruction']}\n\nDocuments:\n{docs_text}"
    
    # Sesuai best practice: Jangan melatih <thought> jika Llama 3.1 tidak diprogram untuk CoT eksplisit.
    # apply_chat_template otomatis akan menambahkan <|eot_id|> di akhir teks.
    assistant_response = sample['completion']
    
    # Llama 3 Chat Template
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]
    return messages

def prepare_dataset(file_path, tokenizer):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                messages = format_raft_prompt(sample)
                # Terapkan chat template bawaan tokenizer (otomatis nambah <|start_header_id|> dsb)
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                data_list.append({"text": text})
                
    return Dataset.from_list(data_list)

# ---------------------------------------------------------------------------
# 2. SETUP MODEL & TOKENIZER
# ---------------------------------------------------------------------------
def main():
    print(f"[*] Memuat Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[*] Menyiapkan Dataset...")
    dataset = prepare_dataset(DATASET_PATH, tokenizer)
    print(f"Total sampel latih: {len(dataset)}")
    
    print("\nContoh Prompt yang akan dilatih:")
    print("="*50)
    print(dataset[0]['text'][:1000] + "\n...[DIPOTONG]...\n")
    print("="*50)

    # Menggunakan full BFloat16 tanpa 4-bit quantization untuk memaksimalkan speed di B200
    print("[*] Memuat Model Llama 3.1 8B dengan Flash Attention 2 (Full BFloat16)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" # Sangat cepat di B200!
    )

    print("[*] Mengonfigurasi LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------------------
    # 3. TRAINING
    # ---------------------------------------------------------------------------
    print("[*] Menyiapkan Validation Split & Data Collator...")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Memastikan model hanya menghitung loss pada jawaban Assistant (bukan pada input dokumen)
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    print("[*] Memulai Training SFT...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch", # Evaluasi di setiap epoch
        load_best_model_at_end=True, # Tambahan untuk Best Model
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,                     # Reproducibility
        optim="adamw_torch_fused",   # Fused optimizer lebih cepat
        bf16=True,                   # B200 sangat mendukung BFloat16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # Aktifkan gradient checkpointing untuk menghemat memori
    model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        packing=False,               # Eksplisit set False atau True tergantung dataset
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Early Stopping
    )

    trainer.train()
    
    print("[*] Training Selesai! Menyimpan model LoRA...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"[+] Model berhasil disimpan di {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
