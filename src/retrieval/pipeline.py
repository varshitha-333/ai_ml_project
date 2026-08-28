"""
High-Level Retrieval Pipeline Interface.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.retrieval.search import HybridFacetRetriever


class RetrievalPipeline:
    """
    Pipeline manager for candidate facet retrieval.
    """

    def __init__(self, data_path: Optional[Path] = None, top_k: int = 30):
        self.top_k = top_k
        self.retriever = HybridFacetRetriever(default_top_k=top_k)
        self.is_initialized = False

        if data_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            data_path = project_root / "data" / "processed" / "enriched_facets.json"

        self.data_path = data_path

    def initialize(self) -> "RetrievalPipeline":
        """
        Loads enriched facet catalog and fits retrieval index.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Enriched dataset not found at {self.data_path}. Run preprocessing first.")

        with open(self.data_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        self.documents = documents
        self.retriever.fit(documents)
        self.is_initialized = True
        return self

    def retrieve(self, conversation_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top candidate observable facets for a conversation text.
        """
        if not self.is_initialized:
            self.initialize()
        return self.retriever.retrieve_candidates(conversation_text, top_k=top_k)
