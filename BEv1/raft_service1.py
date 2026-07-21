import torch
from unsloth import FastLanguageModel

MODEL_DIR = "/workspace/model/sft_instruction_completion/lora_adapter"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 512
DTYPE = None
LOAD_IN_4BIT = False

model = None
tokenizer = None

SYSTEM_PROMPT = (
    "Anda adalah asisten AI yang menjawab pertanyaan tentang pemerintahan "
    "dan peraturan desa. Berikan jawaban langsung, akurat, jelas, dan tidak "
    "mengarang informasi."
)


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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_answer(question, system_prompt=None):
    """
    Menghasilkan jawaban dari model berdasarkan pertanyaan.

    Format prompt sama persis dengan saat fine-tuning:
        system prompt + pertanyaan user → model.generate() → jawaban

    Tidak menggunakan dokumen, retrieval, atau konteks eksternal.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        load_model()

    question = str(question).strip()
    if not question:
        raise ValueError("Pertanyaan tidak boleh kosong.")

    active_system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": active_system_prompt},
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode hanya token yang di-generate (bukan input prompt)
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer