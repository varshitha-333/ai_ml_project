"""
Pipeline Service Singleton Manager for Preloading Static Index & Model Backends.
"""

from typing import Optional, Dict, Any
from src.evaluator_pipeline import FacetEvaluatorPipeline
from src.scoring.inference_client import InferenceClient
from src.scoring.inference_backend import (
    MockInferenceBackend,
    HuggingFaceInferenceBackend,
    RemoteInferenceClientBackend
)
from src.api.config import get_settings


class PipelineService:
    """
    Service wrapper for the FacetEvaluatorPipeline singleton.
    Loads static resources (facet catalog, dense vectors, BM25 index) ONCE on startup.
    """

    _instance: Optional["PipelineService"] = None

    def __init__(self):
        self.settings = get_settings()
        self.client = InferenceClient(
            inference_url=self.settings.inference_url,
            model_name=self.settings.model_name,
            timeout=self.settings.inference_timeout
        )
        self.backend = self._initialize_backend()
        self.pipeline = FacetEvaluatorPipeline(
            backend=self.backend,
            top_k=self.settings.top_k,
            batch_size=self.settings.batch_size
        )
        self.is_initialized = False

    def _initialize_backend(self):
        mode = self.settings.backend_mode.lower()
        if mode == "remote":
            return RemoteInferenceClientBackend(client=self.client)
        elif mode == "huggingface":
            return HuggingFaceInferenceBackend(
                model_id=self.settings.model_name,
                load_in_4bit=True
            )
        else:
            return MockInferenceBackend()

    def initialize(self):
        if not self.is_initialized:
            print("Initializing Facet Evaluator API Service & preloading in-memory index...")
            self.pipeline.initialize()
            self.is_initialized = True
            print("Facet Evaluator API Service Initialized Successfully!")

    @classmethod
    def get_instance(cls) -> "PipelineService":
        if cls._instance is None:
            cls._instance = PipelineService()
        return cls._instance


def get_pipeline_service() -> PipelineService:
    service = PipelineService.get_instance()
    if not service.is_initialized:
        service.initialize()
    return service
