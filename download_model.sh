#!/usr/bin/env bash
# =============================================================================
# Download Model Fine-Tuned dari Server B200 ke Lokal
# =============================================================================
#
# CARA PAKAI (di mesin LOKAL kamu):
#
#   1. Pastikan SSH key sudah ter-setup ke server B200
#   2. Jalankan:
#
#      bash download_model.sh <user>@<server_ip>
#
#   Contoh:
#      bash download_model.sh kel07@10.0.0.5
#
# OPSI TRANSFER:
#   - Opsi A (rsync - RECOMMENDED): resume kalau terputus
#   - Opsi B (scp): sederhana, tidak bisa resume
#   - Opsi C (tar + scp): compress dulu biar lebih cepat
#
# UKURAN MODEL:
#   - Merged model: ~15 GB (butuh GPU 16GB+ VRAM di lokal)
#   - LoRA adapter only: ~177 MB (butuh download base model dari HF)
# =============================================================================

set -euo pipefail

SERVER=${1:-}
REMOTE_DIR="/workspace/notebooks"
LOCAL_MODEL_DIR="./model_merged_rag_perdes"
LOCAL_ADAPTER_DIR="./lora_adapter_rag_perdes"

if [ -z "$SERVER" ]; then
    echo "❌ Usage: bash download_model.sh <user>@<server_ip>"
    echo ""
    echo "Contoh:"
    echo "  bash download_model.sh kel07@10.0.0.5"
    echo ""
    echo "Opsi:"
    echo "  --adapter-only   Hanya download LoRA adapter (~177MB, butuh base model)"
    echo "  --full           Download merged model (~15GB, siap pakai)"
    exit 1
fi

ADAPTER_ONLY=false
if [ "${2:-}" = "--adapter-only" ]; then
    ADAPTER_ONLY=true
fi

echo "============================================================"
echo "  Download Model Fine-Tuned dari Server B200"
echo "============================================================"
echo ""

if $ADAPTER_ONLY; then
    # =========================================================
    # OPSI: Hanya LoRA Adapter (~177MB)
    # =========================================================
    echo "📦 Mode: LoRA Adapter only (~177MB)"
    echo "   Kamu perlu base model dari HuggingFace:"
    echo "   meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo ""

    mkdir -p "$LOCAL_ADAPTER_DIR"

    echo "📥 Downloading adapter dari $SERVER..."
    rsync -avzP \
        "$SERVER:$REMOTE_DIR/lora_adapter_rag_perdes/" \
        "$LOCAL_ADAPTER_DIR/"

    echo ""
    echo "✅ Adapter tersimpan di: $LOCAL_ADAPTER_DIR"
    echo ""
    echo "Cara pakai di lokal:"
    echo "  1. Install: pip install transformers torch peft"
    echo "  2. Python:"
    echo "     from peft import PeftModel"
    echo "     from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "     base = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')"
    echo "     model = PeftModel.from_pretrained(base, './lora_adapter_rag_perdes')"
    echo "     tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')"

else
    # =========================================================
    # OPSI: Merged Model (~15GB) - RECOMMENDED
    # =========================================================
    echo "📦 Mode: Merged Model (~15GB)"
    echo "   Model siap pakai, tidak perlu base model terpisah."
    echo ""

    mkdir -p "$LOCAL_MODEL_DIR"

    echo "📥 Downloading merged model dari $SERVER..."
    echo "   (Ini bisa lama tergantung koneksi internet, ~15GB)"
    echo ""

    # Gunakan rsync dengan progress bar dan resume capability
    rsync -avzP --exclude='*.bin' \
        "$SERVER:$REMOTE_DIR/model_merged_rag_perdes/" \
        "$LOCAL_MODEL_DIR/"

    echo ""
    echo "✅ Model tersimpan di: $LOCAL_MODEL_DIR"
    echo ""
    echo "Cara pakai di lokal (via BE Flask API):"
    echo "  1. Copy model ke folder yang sama dengan BE/"
    echo "  2. cd BE && python run.py"
    echo "  3. curl -X POST http://localhost:6000/api/chat-rag \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"message\": \"Apa isi pasal 1?\", \"konteks\": \"...\"}'"
fi

echo ""
echo "============================================================"
echo "  Selesai!"
echo "============================================================"
