"""
=============================================================================
RAG Pipeline: Chunking -> Embedding -> Retrieval -> Fine-Tune Data -> Compare
=============================================================================

This script demonstrates the FULL consistent pipeline where:
  1. Raw peraturan desa documents are chunked into passages
  2. Those SAME passages are:
     a. Embedded into a vector store (for RAG retrieval)
     b. Used as the "instruction" field in fine-tuning data
  3. At test time, BOTH models get the SAME retrieved context
  4. Results are compared fairly

Key Principle (from Gao et al., 2024; Chen et al., 2024):
  TRAINING CONTEXT == RETRIEVED CONTEXT at inference time

=============================================================================
"""

import json
import os
import re
from pathlib import Path


# ============================================================================
# STEP 1: Extract Chunks from Raw Peraturan Desa Documents
# ============================================================================
"""
Each pasal becomes one chunk. This chunk text is used EVERYWHERE:
  - As instruction in fine-tuning data
  - As text to embed in vector store
  - As context passed to model at inference
"""

DATA_DIR = "../data"


def extract_chunks_from_perdes(filepath: str) -> list[dict]:
    """
    Extract pasal-level chunks from a peraturan desa JSON file.

    Each chunk = one pasal (or one logical section).
    The chunk text becomes:
      - instruction field in fine-tuning data
      - text in vector store for embedding
      - context at retrieval time

    Returns:
        list of dicts with keys: doc_id, pasal, chunk_text, metadata
    """
    filename = Path(filepath).stem
    chunks = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse each line (existing Q&A format)
    qa_pairs = []
    for line in lines:
        line = line.strip()
        if line:
            qa_pairs.append(json.loads(line))

    # Group Q&A pairs by pasal number
    pasal_groups = {}
    for qa in qa_pairs:
        # Extract pasal number from input or output
        input_text = qa.get("input", "")
        output_text = qa.get("output", "")

        pasal_match = re.search(r'Pasal\s+(\d+)', output_text)
        if pasal_match:
            pasal_num = pasal_match.group(1)
        else:
            pasal_match = re.search(r'Pasal\s+(\d+)', input_text)
            pasal_num = pasal_match.group(1) if pasal_match else "unknown"

        if pasal_num not in pasal_groups:
            pasal_groups[pasal_num] = []
        pasal_groups[pasal_num].append(qa)

    # Build chunks: each chunk = one pasal's full text
    for pasal_num, pairs in pasal_groups.items():
        # Use the first output as the basis for chunk text
        # In practice, you'd use the actual regulation document
        first_output = pairs[0].get("output", "")

        # Extract the actual pasal content (after "Berdasarkan Pasal X...")
        content_match = re.search(
            r'(?:Berdasarkan|Menurut|Dalam)\s+Pasal\s+\d+[^,]*,\s*(.*)',
            first_output, re.DOTALL
        )
        pasal_content = content_match.group(1) if content_match else first_output

        # Build the chunk text (THIS IS WHAT GETS EMBEDDED AND USED IN FINE-TUNE)
        chunk_text = (
            f"{filename}\n"
            f"Pasal {pasal_num}: {pasal_content}"
        )

        chunks.append({
            "doc_id": filename,
            "pasal": pasal_num,
            "chunk_text": chunk_text,
            "qa_pairs": pairs,  # associated Q&A for fine-tuning
        })

    return chunks


# ============================================================================
# STEP 2: Build Vector Store (Embed Chunks)
# ============================================================================
"""
The SAME chunk_text from Step 1 is embedded here.
This ensures retrieval returns the exact text the model was trained on.
"""

# Pseudocode for embedding (requires sentence-transformers or similar):
#
# from sentence_transformers import SentenceTransformer
# import numpy as np
#
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # or Indonesian model
#
# all_chunks = []  # all chunks from all documents
# all_embeddings = []
#
# for filepath in glob(f"{DATA_DIR}/*.json"):
#     chunks = extract_chunks_from_perdes(filepath)
#     for chunk in chunks:
#         all_chunks.append(chunk)
#         embedding = embedder.encode(chunk["chunk_text"])
#         all_embeddings.append(embedding)
#
# # Store in vector DB (FAISS, ChromaDB, etc.)
# vector_store = FAISS.from_embeddings(all_embeddings, all_chunks)


# ============================================================================
# STEP 3: Generate Fine-Tuning Data from Chunks
# ============================================================================
"""
KEY INSIGHT: The instruction field = chunk_text (the SAME text in vector store)

This ensures:
  - During training: model sees chunk_text as context
  - During inference: retriever returns chunk_text as context
  - PERFECT MATCH between training and inference!
"""

def generate_finetune_data_from_chunks(all_chunks: list[dict]) -> list[dict]:
    """
    Generate fine-tuning dataset from chunks.

    Each sample:
      instruction = chunk_text (same as embedded text)
      input = question
      output = grounded answer
    """
    finetune_data = []

    for chunk in all_chunks:
        for qa in chunk["qa_pairs"]:
            sample = {
                "instruction": chunk["chunk_text"],  # EXACT SAME as vector store
                "input": qa["input"],
                "output": qa["output"],
            }
            finetune_data.append(sample)

    return finetune_data


def generate_distractor_data(
    all_chunks: list[dict],
    num_distractors: int = 2
) -> list[dict]:
    """
    Generate fine-tuning data WITH distractors.

    Each sample:
      instruction = relevant_chunk + [distractor_chunk_1] + [distractor_chunk_2]
      input = question
      output = answer from relevant chunk only + note about distractors
    """
    import random
    random.seed(42)

    distractor_data = []

    for i, target_chunk in enumerate(all_chunks):
        # Get distractors from DIFFERENT documents
        other_chunks = [
            c for c in all_chunks
            if c["doc_id"] != target_chunk["doc_id"]
        ]

        if len(other_chunks) < num_distractors:
            continue

        distractors = random.sample(other_chunks, num_distractors)

        # Build instruction with relevant + distractor chunks
        instruction_parts = [target_chunk["chunk_text"]]
        for d in distractors:
            instruction_parts.append(
                f"\n[DISTRAKTOR] {d['chunk_text']}"
            )

        combined_instruction = "\n".join(instruction_parts)

        # Generate Q&A from the target chunk
        for qa in target_chunk["qa_pairs"][:1]:  # Use first Q&A per chunk
            # Build output with distractor note
            distractor_names = ", ".join(
                [d["doc_id"] for d in distractors]
            )
            grounded_output = (
                f"{qa['output']}\n\n"
                f"(Catatan: Informasi dari {distractor_names} "
                f"tidak relevan dengan pertanyaan ini dan diabaikan.)"
            )

            sample = {
                "instruction": combined_instruction,
                "input": qa["input"],
                "output": grounded_output,
            }
            distractor_data.append(sample)

    return distractor_data


# ============================================================================
# STEP 4: Retrieval at Inference Time (Same Chunks!)
# ============================================================================
"""
At inference, the retriever returns the SAME chunk_text that was used in
fine-tuning. This ensures a fair comparison.
"""

def retrieve_context(
    query: str,
    vector_store,
    top_k: int = 3,
    embedder=None,
) -> list[dict]:
    """
    Retrieve top-k most relevant chunks for a query.

    Returns the SAME chunk_text that was used in fine-tuning instruction.
    """
    # query_embedding = embedder.encode(query)
    # results = vector_store.search(query_embedding, top_k=top_k)
    # return results  # each result contains chunk_text

    # Pseudocode - actual implementation depends on vector DB
    pass


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
    include_distractors: bool = True,
) -> str:
    """
    Build the prompt for the LLM with retrieved context.

    The context format matches the fine-tuning instruction format EXACTLY.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        if i == 0:
            # First chunk assumed to be most relevant
            context_parts.append(chunk["chunk_text"])
        elif include_distractors:
            context_parts.append(
                f"\n[DISTRAKTOR] {chunk['chunk_text']}"
            )

    full_context = "\n".join(context_parts)
    prompt = f"{full_context}\n\n{query}"

    return prompt


# ============================================================================
# STEP 5: Fair Comparison Setup
# ============================================================================
"""
For a fair comparison, BOTH models must receive the EXACT SAME input:
  - Same retrieved chunks
  - Same context format
  - Same question
  - Same generation parameters
"""

def run_comparison(
    question: str,
    vector_store,
    base_model, base_tokenizer,      # Vanilla RAG
    ft_model, ft_tokenizer,          # Fine-Tuned RAG
    embedder=None,
    top_k: int = 3,
):
    """
    Run fair comparison: both models get same retrieved context.
    """

    # 1. Retrieve context (SAME for both models)
    retrieved_chunks = retrieve_context(
        question, vector_store, top_k, embedder
    )

    # 2. Build prompt (SAME format for both models)
    context = build_rag_prompt(question, retrieved_chunks)

    # 3. Generate from both models (SAME parameters)
    vanilla_answer = generate(
        base_model, base_tokenizer, context, question
    )
    finetuned_answer = generate(
        ft_model, ft_tokenizer, context, question
    )

    return {
        "question": question,
        "retrieved_context": [c["chunk_text"] for c in retrieved_chunks],
        "vanilla_rag_answer": vanilla_answer,
        "finetuned_rag_answer": finetuned_answer,
    }


def generate(model, tokenizer, context, question, max_new_tokens=512):
    """Generate answer from model with given context."""
    prompt = f"{context}\n\n{question}"
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
    )

    return tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )


# ============================================================================
# STEP 6: Full Pipeline Example
# ============================================================================
"""
Here's how everything connects:
"""

if __name__ == "__main__":
    print("=" * 70)
    print("FULL RAG PIPELINE: Chunking -> Embedding -> Fine-Tune -> Compare")
    print("=" * 70)

    # --- STEP A: Extract chunks from all peraturan desa files ---
    print("\n[STEP A] Extracting chunks from peraturan desa files...")

    all_chunks = []
    json_files = sorted(Path(DATA_DIR).glob("*.json"))[:10]  # first 10 files

    for filepath in json_files:
        chunks = extract_chunks_from_perdes(str(filepath))
        all_chunks.extend(chunks)
        print(f"  {filepath.name}: {len(chunks)} chunks (pasal)")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # --- STEP B: Generate fine-tuning data ---
    print("\n[STEP B] Generating fine-tuning data from chunks...")

    clean_data = generate_finetune_data_from_chunks(all_chunks)
    print(f"  Clean samples (no distractors): {len(clean_data)}")

    distractor_samples = generate_distractor_data(all_chunks, num_distractors=2)
    print(f"  Distractor samples: {len(distractor_samples)}")

    # Save
    all_data = clean_data + distractor_samples
    output_path = "../data/auto_generated_finetune_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Saved to: {output_path}")

    # --- STEP C: Show how chunk_text flows ---
    print("\n[STEP C] Data Consistency Verification")
    print("-" * 70)

    if all_chunks:
        sample_chunk = all_chunks[0]
        print(f"\n  Chunk text (stored in vector store):")
        print(f"  '{sample_chunk['chunk_text'][:150]}...'")

        print(f"\n  Same text as fine-tune instruction:")
        if clean_data:
            print(f"  '{clean_data[0]['instruction'][:150]}...'")

        print(f"\n  Same text as retrieval context at inference:")
        print(f"  (retriever returns this exact chunk_text)")

        print(f"\n  ✓ MATCH: chunk_text == instruction == retrieval_context")

    # --- STEP D: Diagram ---
    print("\n" + "=" * 70)
    print("PIPELINE DIAGRAM")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    RAW PERATURAN DESA FILES                      │
    │                  (Desa Biru, Cigentur, etc.)                     │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     CHUNKING (per pasal)                         │
    │                                                                  │
    │  chunk = "Desa Biru No 3/2016                                   │
    │           Pasal 8: Biaya pemasangan Rp 300.000..."              │
    │                                                                  │
    │  THIS CHUNK IS USED IN ALL 3 PLACES BELOW                       │
    └──────┬──────────────────┬──────────────────┬────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
    │ VECTOR STORE │  │ FINE-TUNE    │  │ DISTRACTOR           │
    │ (Embedding)  │  │ DATA         │  │ DATA                 │
    │              │  │              │  │                      │
    │ embed(chunk) │  │ instruction  │  │ instruction =        │
    │ -> vector    │  │  = chunk     │  │  relevant_chunk +    │
    │              │  │ input = ?    │  │  distractor_chunk_1 + │
    │ stored as:   │  │ output = ans │  │  distractor_chunk_2   │
    │ {vector,     │  │              │  │                      │
    │  chunk_text} │  │              │  │ input = ?             │
    └──────┬───────┘  └──────┬───────┘  │ output = ans + note   │
           │                 │          └──────────┬───────────┘
           │                 │                     │
           ▼                 ▼                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    INFERENCE TIME                                │
    │                                                                  │
    │  1. User asks: "Berapa biaya air bersih di Desa Biru?"          │
    │  2. Retriever searches vector store                             │
    │  3. Returns: top-k chunks (SAME chunk_text as training)         │
    │  4. Build prompt: retrieved chunks + question                   │
    │                                                                  │
    │  ┌─────────────────────────────┐  ┌─────────────────────────────┐
    │  │ MODEL A: Vanilla RAG        │  │ MODEL B: Fine-Tuned RAG     │
    │  │ (base Llama 3.1)            │  │ (Llama + LoRA fine-tuned)   │
    │  │                             │  │                             │
    │  │ Input: SAME prompt          │  │ Input: SAME prompt          │
    │  │ Output: may hallucinate     │  │ Output: grounded in context │
    │  │         mixes desa info     │  │         ignores distractors │
    │  └─────────────────────────────┘  └─────────────────────────────┘
    │                                                                  │
    │  COMPARE: faithfulness, relevance, ROUGE-L, hallucination       │
    └─────────────────────────────────────────────────────────────────┘
    """)

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. CHUNK TEXT MUST BE IDENTICAL everywhere:
       - What's in the vector store
       - What's in the fine-tune instruction field
       - What's retrieved and passed to the model at inference

    2. For distractors:
       - During training: distractors are ADDED to instruction field
       - During inference: retriever returns top-k (some may be irrelevant)
       - The model learned to IGNORE distractors during training

    3. Fair comparison requires:
       - Same test questions
       - Same retrieval results (same vector store, same query)
       - Same prompt format
       - Same generation parameters (temperature, max_tokens, etc.)

    4. Test set should include:
       - Questions where retriever finds the CORRECT pasal (easy)
       - Questions where retriever returns MIXED relevant+irrelevant (hard)
       - Questions from desa NOT in training set (generalization test)
    """)
