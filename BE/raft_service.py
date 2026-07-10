import torch
from unsloth import FastLanguageModel

MODEL_DIR = "/workspace/model/model_merged_raft_v4"  
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False

model = None
tokenizer = None

RAFT_SYSTEM_PROMPT = """Anda adalah asisten AI ahli dalam menjawab pertanyaan berdasarkan dokumen hukum dan peraturan desa.

Tugas Anda:
1. Periksa semua dokumen referensi yang diberikan.
2. Pilih hanya dokumen yang benar-benar menjawab pertanyaan.
3. Abaikan dokumen yang tidak relevan, salah pasal/ayat, atau hanya mirip topiknya.
4. Jika tidak ada dokumen yang valid, katakan bahwa informasi tidak ditemukan pada dokumen yang diberikan.
5. Jawaban akhir hanya boleh berdasarkan dokumen yang dipilih.

Format jawaban HARUS seperti ini:
KONTEKS_DIPILIH: [id dokumen]
KONTEKS_DITOLAK: [id dokumen]
JAWABAN:
<jawaban akhir>
"""

INFER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

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
        model.generation_config.max_length = None

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

    if "JAWABAN:" in result_text:
        result_text = result_text.split("JAWABAN:")[-1].strip()

    return result_text