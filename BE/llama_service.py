from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "model_merged_legal")
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("⏳ Loading model...")    
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        _tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("✅ Model loaded.")
    return _model, _tokenizer


def generate_answer(pertanyaan: str, system_prompt: str = None, max_new_tokens: int = 512) -> str:
    model, tokenizer = load_model()

    if system_prompt is None:
        system_prompt = "Kamu adalah asisten hukum yang membantu menjawab pertanyaan dengan bahasa yang mudah dipahami."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pertanyaan}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response
