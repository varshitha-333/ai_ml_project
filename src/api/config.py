"""
Typed Configuration Management for FastAPI Layer using Pydantic Settings / OS Environment.
"""

import os
from typing import List
from pydantic import BaseModel

from pathlib import Path

env_file = Path(__file__).resolve().parent.parent.parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file, override=True)
except ImportError:
    pass


class APISettings(BaseModel):
    """
    Typed API Settings configuration class.
    """
    model_name: str = os.getenv("MODEL_NAME", os.getenv("INFERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")).strip('"\'')
    inference_url: str = os.getenv("INFERENCE_URL", "https://salvaging-ardently-late.ngrok-free.dev").strip('"\'')
    inference_timeout: int = int(os.getenv("INFERENCE_TIMEOUT", "60"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip('"\'')
    top_k: int = int(os.getenv("TOP_K", "10"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    allowed_origins_raw: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").strip('"\'')
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip('"\'')
    backend_mode: str = os.getenv("BACKEND_MODE", "remote").strip('"\'')  # 'remote', 'huggingface', 'mock'
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    rerank_initial_pool_size: int = int(os.getenv("RERANK_INITIAL_POOL_SIZE", "30"))

    @property
    def allowed_origins(self) -> List[str]:
        if not self.allowed_origins_raw or self.allowed_origins_raw.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins_raw.split(",") if origin.strip()]


def get_settings() -> APISettings:
    return APISettings()
