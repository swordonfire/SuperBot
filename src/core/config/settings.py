import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    PROJECT_NAME: str = 'SuperBot'
    # LLM_MODEL_PATH: str = "data/models/DeepseekQwen/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    # LLM_MODEL_PATH: str = "data/models/llama3_1-8B/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    LLM_MODEL_PATH: Optional[str] = os.getenv('LLM_MODEL_PATH')
    EMBEDDING_MODEL: Optional[str] = os.getenv('EMBEDDING_MODEL')
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv('TELEGRAM_BOT_TOKEN')

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    def validate_paths(self):
        """Ensure model file exists"""
        model_path = Path(self.LLM_MODEL_PATH)
        logging.info(f'Checking model path: {model_path.absolute()}')
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found at {model_path}')
        logging.info('Model path validation successful!')


# Initialize and validate settings immediately
settings = Settings()
settings.validate_paths()
