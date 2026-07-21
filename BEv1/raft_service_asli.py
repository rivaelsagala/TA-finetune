import re
import torch
from unsloth import FastLanguageModel

MODEL_DIR = "/workspace/model/raft_unsloth_v8/lora_adapter"  
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False

model = None
tokenizer = None

RAFT_SYSTEM_PROMPT = """Anda adalah asisten AI ahli dalam menjawab pertanyaan berdasarkan dokumen hukum dan peraturan desa.

Tugas Anda:
1. Periksa semua dokumen referensi.
2. Cocokkan secara ketat nama desa, nomor/tahun peraturan, pasal, ayat, dan isi ketentuan.
3. Pilih dokumen yang benar-benar mendukung jawaban dan abaikan distraktor.
4. Jangan menggabungkan ketentuan dari desa, nomor peraturan, pasal, atau ayat yang berbeda.
5. Kutip potongan bukti yang benar-benar terdapat pada dokumen terpilih.
6. Jangan menambah fakta di luar dokumen atau pengetahuan domain yang telah dipelajari.
7. Jika informasi memang tidak tersedia, nyatakan dengan jelas bahwa pertanyaan tidak dapat dijawab dari informasi yang tersedia.

Format jawaban HARUS seperti ini:

KONTEKS_DIPILIH: [id dokumen]
KONTEKS_DITOLAK: [id dokumen]


<thought>
alasan memilih atau menolak dokumen
</thought>

JAWABAN:
<jawaban akhir>

"""


# Prompt inference harus berhenti tepat setelah header assistant.
INFER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Pertanyaan: {instruction}

Dokumen Referensi:
{documents}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def load_model():
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    FastLanguageModel.for_inference(model)

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = 4096

    return model, tokenizer


def format_documents(docs):
    formatted = []
    for i, doc in enumerate(docs, start=1):
        formatted.append(f"[{i}] {doc}")
    return "\n\n".join(formatted)


def generate_answer(question, documents, system_prompt=None):
    global model, tokenizer

    if model is None or tokenizer is None:
        load_model()

    if not system_prompt:
        system_prompt = RAFT_SYSTEM_PROMPT

    documents_text = format_documents(documents)

    prompt = INFER_PROMPT.format(
        system_prompt=system_prompt,
        instruction=question,
        documents=documents_text,
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

    result_text = decoded[len(prompt_text):].strip()
    result_text = result_text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

    konteks_dipilih = ""
    konteks_ditolak = ""
    thought_process = ""
    jawaban = result_text

    # Extract KONTEKS_DIPILIH
    match_dipilih = re.search(r"KONTEKS_DIPILIH:\s*(.*)", result_text)
    if match_dipilih:
        konteks_dipilih = match_dipilih.group(1).strip()

    # Extract KONTEKS_DITOLAK
    match_ditolak = re.search(r"KONTEKS_DITOLAK:\s*(.*)", result_text)
    if match_ditolak:
        konteks_ditolak = match_ditolak.group(1).strip()

    # Extract <thought>
    match_thought = re.search(r"<thought>(.*?)</thought>", result_text, re.DOTALL)
    if match_thought:
        thought_process = match_thought.group(1).strip()

    # Extract JAWABAN
    if "JAWABAN:" in result_text:
        jawaban = result_text.split("JAWABAN:")[-1].strip()

    return {
        "konteks_dipilih": konteks_dipilih,
        "konteks_ditolak": konteks_ditolak,
        "thought_process": thought_process,
        "jawaban": jawaban
    }