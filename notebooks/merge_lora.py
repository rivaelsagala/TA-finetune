import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_ADAPTER_DIR = "../models/llama-3.1-8b-raft-lora/final"
MERGED_OUTPUT_DIR = "../models/llama-3.1-8b-raft-merged"

def main():
    print(f"[*] 1. Memuat Base Model asli: {BASE_MODEL_ID}")
    # Load base model (disarankan di CPU saat merging agar tidak terjadi error memori/OOM)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    print(f"[*] 2. Memuat LoRA Adapter hasil training Anda dari: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    print("[*] 3. Menyatukan (Merging) Base Model dan LoRA secara permanen...")
    # Perintah sakti untuk menggabungkan weight LoRA ke Base Model
    merged_model = model.merge_and_unload()
    
    print(f"[*] 4. Menyimpan model utuh yang baru ke: {MERGED_OUTPUT_DIR}")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    
    print("[*] 5. Menyimpan Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("\n[+] TAHAP 1 SELESAI! Model berhasil disatukan.")
    print(f"[!] Sekarang Anda bisa mengonversi folder '{MERGED_OUTPUT_DIR}' menjadi .gguf menggunakan llama.cpp")

if __name__ == "__main__":
    main()
