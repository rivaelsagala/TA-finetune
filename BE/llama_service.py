"""
LlamaService - Thread-safe singleton for RAFT model inference.

Handles model loading, text generation (plain chat and RAG/RAFT mode),
Chain-of-Thought parsing, and answer enrichment.
"""

import threading
import traceback
import torch
import re
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# System prompt - MUST match training exactly
RAFT_SYSTEM_PROMPT = (
    "Anda adalah asisten hukum yang membantu menjawab pertanyaan tentang "
    "Peraturan Desa (Perdes) di Indonesia. Jawab pertanyaan berdasarkan "
    "dokumen-dokumen yang diberikan. Tidak semua dokumen relevan dengan "
    "pertanyaan, jadi pilihlah informasi dari dokumen yang paling sesuai."
)


class LlamaService:
    """
    Thread-safe singleton service for Llama model inference.
    Supports both plain chat and RAFT (Retrieval-Augmented Fine-Tuning) modes.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize service with empty model state."""
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_type = None
        self._instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton instance (thread-safe).
        
        Returns:
            LlamaService: The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str):
        """
        Load model and tokenizer from the specified path.

        Args:
            model_path: Path to the model directory.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info(f"Loading model from: {model_path}")

            # Load model with automatic device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Store model metadata
            self.model_path = model_path
            self.model_type = "raft" if "raft" in model_path.lower() else "base"

            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(f"GPU memory allocated: {allocated:.2f} GB")
                logger.info(f"GPU memory reserved: {reserved:.2f} GB")

            logger.info(f"Model loaded successfully. Type: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            # Reset state on failure
            self.model = None
            self.tokenizer = None
            self.model_path = None
            self.model_type = None
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_answer(
        self,
        pertanyaan: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> dict:
        """
        Generate answer in plain chat mode (no documents).

        Args:
            pertanyaan: The user's question.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.

        Returns:
            dict: Contains raw_response and model_type, or error info.
        """
        try:
            # Build conversation messages
            messages = [
                {"role": "system", "content": RAFT_SYSTEM_PROMPT},
                {"role": "user", "content": pertanyaan}
            ]

            # Tokenize using chat template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Track prompt length for decoding only new tokens
            prompt_length = input_ids.shape[-1]

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    top_k=50,
                    repetition_penalty=1.15,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode only the generated tokens (skip prompt)
            generated_tokens = output_ids[0][prompt_length:]
            decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return {
                "raw_response": decoded_text.strip(),
                "model_type": self.model_type
            }

        except Exception as e:
            logger.error(f"Error in generate_answer: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return {
                "raw_response": "",
                "model_type": self.model_type,
                "error": f"{type(e).__name__}: {e}" or repr(e)
            }

    def generate_answer_rag(
        self,
        pertanyaan: str,
        dokumen: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> dict:
        """
        Generate answer in RAG/RAFT mode with document context.

        Args:
            pertanyaan: The user's question.
            dokumen: List of document strings.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.

        Returns:
            dict: Contains analisis, jawaban, raw_response, model_type, num_documents.
        """
        try:
            # Format documents for input
            formatted_docs = self._format_documents(dokumen)

            # Build user message with question and documents
            user_message = f"{pertanyaan}{formatted_docs}"

            # Build conversation messages
            messages = [
                {"role": "system", "content": RAFT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]

            # Tokenize using chat template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Track prompt length for decoding only new tokens
            prompt_length = input_ids.shape[-1]

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    top_k=50,
                    repetition_penalty=1.15,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode only the generated tokens (skip prompt)
            generated_tokens = output_ids[0][prompt_length:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Parse Chain-of-Thought and answer
            analisis, jawaban = self._split_cot_answer(raw_text)

            # Enrich/clean the answer
            jawaban = self._enrich_jawaban(jawaban)

            return {
                "analisis": analisis,
                "jawaban": jawaban,
                "raw_response": raw_text.strip(),
                "model_type": "raft",
                "num_documents": len(dokumen)
            }

        except Exception as e:
            logger.error(f"Error in generate_answer_rag: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return {
                "analisis": "",
                "jawaban": "",
                "raw_response": "",
                "model_type": "raft",
                "num_documents": len(dokumen),
                "error": f"{type(e).__name__}: {e}" or repr(e)
            }

    def _format_documents(self, dokumen: list) -> str:
        """
        Format documents as numbered list matching training format.

        Args:
            dokumen: List of document strings.

        Returns:
            str: Formatted document string.
        """
        docs_text = ""
        for idx, doc in enumerate(dokumen, 1):
            docs_text += f"\n\nDokumen {idx}:\n{doc}"
        return docs_text

    def _split_cot_answer(self, raw_text: str) -> tuple:
        """
        Parse <CoT>...</CoT> block from raw output.

        Args:
            raw_text: Raw model output text.

        Returns:
            tuple: (analisis, jawaban)
        """
        # Try to find CoT block
        cot_pattern = r'<CoT>(.*?)</CoT>\s*(.*)'
        match = re.search(cot_pattern, raw_text, re.DOTALL)

        if match:
            analisis = match.group(1).strip()
            jawaban = match.group(2).strip()
            return (analisis, jawaban)

        # Fallback: split on double newline
        if '\n\n' in raw_text:
            parts = raw_text.split('\n\n', 1)
            analisis = parts[0].strip()
            jawaban = parts[1].strip() if len(parts) > 1 else ""
            return (analisis, jawaban)

        # No split possible
        return ("", raw_text.strip())

    def _enrich_jawaban(self, jawaban: str) -> str:
        """
        Clean and enrich the answer by removing document references.

        Args:
            jawaban: Raw answer text.

        Returns:
            str: Cleaned answer text.
        """
        # Return as-is if too short
        if len(jawaban) < 10:
            return jawaban

        # Remove document references (e.g., "Dokumen 1", "lihat Dokumen 2")
        jawaban = re.sub(
            r'(?:lihat |mengacu pada |berdasarkan )?Dokumen \d+',
            '',
            jawaban,
            flags=re.IGNORECASE
        )

        # Clean up multiple spaces
        jawaban = re.sub(r'  +', ' ', jawaban)

        # Clean up multiple newlines
        jawaban = re.sub(r'\n{3,}', '\n\n', jawaban)

        # Strip leading/trailing whitespace
        jawaban = jawaban.strip()

        return jawaban

    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.

        Returns:
            dict: Model information including GPU memory usage.
        """
        if self.model is None:
            return {
                "model_path": None,
                "model_type": None,
                "loaded": False,
                "gpu_memory_allocated_gb": 0,
                "gpu_memory_reserved_gb": 0,
                "gpu_device_count": 0
            }

        # Get GPU memory info
        gpu_allocated = 0
        gpu_reserved = 0
        gpu_count = 0

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_count = torch.cuda.device_count()

        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "loaded": True,
            "gpu_memory_allocated_gb": round(gpu_allocated, 2),
            "gpu_memory_reserved_gb": round(gpu_reserved, 2),
            "gpu_device_count": gpu_count
        }
