import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ========================
# 1. Load dan preprocessing dataset
# ========================
def load_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    formatted = []
    for item in data:
        # Gabungkan dokumen dengan penanda
        docs = item["documents"]
        doc_text = ""
        for i, d in enumerate(docs, 1):
            doc_text += f"[Dokumen {i}]: {d}\n"
        
        # Prompt user
        user_msg = f"Pertanyaan: {item['instruction']}\n\nBerikut adalah dokumen yang tersedia:\n{doc_text.strip()}"
        
        # Target asisten = reasoning + jawaban
        assistant_msg = f"{item['thought_process']}\n\nJawaban: {item['completion']}"
        
        formatted.append({"user": user_msg, "assistant": assistant_msg})
    
    return Dataset.from_list(formatted)

dataset = load_dataset("../dataset/raft_dataset_v3_production.jsonl")  # ganti path sesuai
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "eval": split_dataset["test"]
})
print(dataset)

# ========================
# 2. Model dan tokenizer (quantized 4-bit untuk efisiensi)
# ========================
model_name = "../model/Meta-Llama-3.1-8B-Instruct"  # atau path lokal
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# ========================
# 3. Format chat sesuai Llama 3.1 Instruct
# ========================
def formatting_func(example):
    """
    Mengubah contoh ke format chat Llama 3.1.
    """
    messages = [
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    # Gunakan chat_template bawaan tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# DataCollator untuk hanya menghitung loss pada bagian assistant
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# ========================
# 4. LoRA Configuration
# ========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)  # ~0.2% dari total parameter


# ========================
# 5. Training Arguments + SFTTrainer
# ========================
output_dir = "../model/llama3.1-8b-raft-lora"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    formatting_func=formatting_func,
    data_collator=collator,
    max_seq_length=2048,
    neftune_noise_alpha=5,  # opsional, meningkatkan stabilitas
)

# Mulai training
trainer.train()


# ========================
# 6. Simpan adapter LoRA & tokenizer
# ========================
trainer.save_model()
tokenizer.save_pretrained(output_dir)