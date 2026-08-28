"""
Typed Configuration Management for FastAPI Layer using Pydantic Settings / OS Environment.
"""

import os
from typing import List
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class APISettings(BaseModel):
    """
    Typed API Settings configuration class.
    """
    model_name: str = os.getenv("INFERENCE_MODEL", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")).strip('"\'')
    inference_url: str = os.getenv("INFERENCE_URL", "http://localhost:8000").strip('"\'')
    inference_timeout: int = int(os.getenv("INFERENCE_TIMEOUT", "30"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip('"\'')
    top_k: int = int(os.getenv("TOP_K", "30"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "30"))
    allowed_origins_raw: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").strip('"\'')
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip('"\'')
    backend_mode: str = os.getenv("BACKEND_MODE", "remote").strip('"\'')  # 'remote', 'huggingface', 'mock'

    @property
    def allowed_origins(self) -> List[str]:
        if not self.allowed_origins_raw or self.allowed_origins_raw.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins_raw.split(",") if origin.strip()]


def get_settings() -> APISettings:
    return APISettings()
