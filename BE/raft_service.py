import re
import random
import torch
from unsloth import FastLanguageModel

MODEL_DIR = "/workspace/model/sft_instruction_context_thought_completion/lora_adapter5"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 1024
DTYPE = None
LOAD_IN_4BIT = False

model = None
tokenizer = None


SYSTEM_PROMPT = (
    "Anda adalah asisten AI yang menjawab pertanyaan tentang pemerintahan "
    "dan peraturan desa. Gunakan context yang diberikan sebagai referensi utama. "
    "Jika context mengandung informasi yang relevan dengan pertanyaan, gunakan "
    "informasi tersebut sebagai konfirmasi dan dasar jawaban. "
    "Jika context tidak relevan atau tidak memiliki informasi yang cukup, "
    "jawab berdasarkan pengetahuan Anda sendiri - dan jika benar-benar tidak tahu, katakan tidak tahu. "
    "Evaluasi semua dokumen; dokumen pertama bukan selalu yang paling relevan. "
    "Tulis analisis singkat dan relevan pada bagian '### Thought', lalu tulis "
    "jawaban final yang jelas pada bagian '### Answer'. Jangan mengarang fakta "
    "yang tidak didukung context maupun pengetahuan Anda."
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

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def get_document_text(document):
    if document is None:
        return ""
    if isinstance(document, str):
        return document.strip()
    if isinstance(document, dict):
        content = (
            document.get("content")
            or document.get("text")
            or document.get("document")
            or ""
        )
        return str(content).strip()
    return str(document).strip()


def format_context(context, document_order="normal", random_seed=None):

    if context is None:
        return "Tidak ada context tambahan."

    if isinstance(context, str):
        text = context.strip()
        return text or "Tidak ada context tambahan."

    if isinstance(context, list):
        doc_texts = []

        for item in context:
            text = get_document_text(item)
            if text:
                doc_texts.append(text)

        if not doc_texts:
            return "Tidak ada context tambahan."

        if document_order == "reverse":
            doc_texts = list(reversed(doc_texts))

        elif document_order == "shuffle":
            rng = random.Random(random_seed)
            rng.shuffle(doc_texts)

        elif document_order != "normal":
            raise ValueError(
                "document_order harus berupa "
                "'normal', 'reverse', atau 'shuffle'."
            )

        parts = [
            f"[Dokumen {i}]\n{text}"
            for i, text in enumerate(doc_texts, start=1)
        ]

        return "\n\n".join(parts)

    text = get_document_text(context)
    return text or "Tidak ada context tambahan."


def build_user_content(instruction, context):
    return (
        "### Instruction\n"
        f"{instruction.strip()}\n\n"
        "### Context\n"
        f"{context.strip()}"
    )


def build_assistant_content(thought, completion):
    return (
        f"{thought.strip()}\n\n"
        f"{completion.strip()}"
    )


def parse_model_response(text):
    text = str(text).strip()
    thought_marker = "### Thought"
    answer_marker = "### Answer"

    thought = ""
    answer = text

    if answer_marker in text:
        before_answer, answer = text.split(answer_marker, 1)
        answer = answer.lstrip(" :\n").strip()

        if thought_marker in before_answer:
            thought = before_answer.split(thought_marker, 1)[1]
            thought = thought.lstrip(" :\n").strip()
    elif thought_marker in text:
        thought = text.split(thought_marker, 1)[1].lstrip(" :\n").strip()
        answer = ""

    return {
        "thought": thought,
        "answer": answer,
        "raw_output": text,
    }


def generate_response(
    question,
    context=None,
    documents=None,
    document_order="normal",
):
    global model, tokenizer

    if model is None or tokenizer is None:
        load_model()

    question = str(question).strip()

    if not question:
        raise ValueError("Pertanyaan tidak boleh kosong.")

    raw_context = context if context is not None else documents

    context_text = format_context(
        raw_context,
        document_order=document_order,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_content(
                question,
                context_text,
            ),
        },
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

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return parse_model_response(generated_text)


def generate_answer(question, context=None, documents=None):

    result = generate_response(question=question, context=context, documents=documents)
    return result["answer"] or result["raw_output"]