"""
from typing import ClassVar, Optional
from llama_cpp import Llama
from src.core.config.settings import settings


class LLMEngine:
    _instance: ClassVar[Optional["LLMEngine"]] = None
    model: Llama  # Explicitly declare the attribute

    def __new__(cls) -> "LLMEngine":
        if cls._instance is None:
            # Create the instance and initialize attributes
            cls._instance = super().__new__(cls)
            cls._instance.model = Llama(
                model_path=settings.LLM_MODEL_PATH,
                n_gpu_layers=20,
                # n_ctx=2048,
                n_ctx=4096,
                n_batch=512,
                verbose=False,
            )
        return cls._instance


def get_llm_engine() -> LLMEngine:
    return LLMEngine()
"""

# src/llm/engine.py
from typing import ClassVar, Optional

from llama_cpp import Llama

from src.core.config.settings import settings


class LLMEngine:
    _instance: ClassVar[Optional['LLMEngine']] = None
    model: Llama

    def __new__(cls) -> 'LLMEngine':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = Llama(
                model_path=settings.LLM_MODEL_PATH,
                n_gpu_layers=20,
                n_ctx=4096,  # Keep your original context window
                n_batch=512,
                verbose=False,
            )
        return cls._instance

    def generate_response(self, prompt: str) -> str:
        """Generate response using initialized model"""
        response = self.model.create_chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3,  # Add your desired params
            max_tokens=256,
        )
        return response['choices'][0]['message']['content']


def get_llm_engine() -> LLMEngine:
    return LLMEngine()


# Add direct access function for RAG pipeline
def generate_response(prompt: str) -> str:
    return get_llm_engine().generate_response(prompt)
