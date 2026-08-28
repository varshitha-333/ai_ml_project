"""
Dense Vector Indexer & Vector Search with 3-tier Fallback & Disk Caching.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import re
from pathlib import Path


class PurePythonTFVectorizer:
    """
    3rd-tier zero-dependency Pure Python Term Frequency Cosine Similarity vectorizer.
    Ensures pipeline execution even when sentence-transformers and scikit-learn are missing.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        self.doc_norms: List[float] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        vocab_set = set()
        tokenized_docs = [self._tokenize(d) for d in docs]
        for tokens in tokenized_docs:
            vocab_set.update(tokens)

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocabulary)
        matrix = np.zeros((len(docs), max(vocab_size, 1)), dtype=np.float32)

        for doc_idx, tokens in enumerate(tokenized_docs):
            for t in tokens:
                if t in self.vocabulary:
                    matrix[doc_idx, self.vocabulary[t]] += 1.0
            
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        return matrix

    def transform_single(self, text: str) -> np.ndarray:
        vocab_size = len(self.vocabulary)
        vec = np.zeros((1, max(vocab_size, 1)), dtype=np.float32)
        tokens = self._tokenize(text)
        for t in tokens:
            if t in self.vocabulary:
                vec[0, self.vocabulary[t]] += 1.0
        norm = np.linalg.norm(vec[0])
        if norm > 0:
            vec[0] /= norm
        return vec


class DenseVectorIndexer:
    """
    Dense vector embedder using SentenceTransformers (primary), Scikit-Learn TF-IDF (secondary),
    or PurePythonTFVectorizer (tertiary fallback) with disk caching for high performance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.pure_vectorizer = None
        self.backend_type = None
        self.embeddings: np.ndarray = None
        self.facet_docs: List[Dict[str, Any]] = []
        
        if cache_dir is None:
            cache_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed")
        self.cache_path = Path(cache_dir) / "facet_embeddings_cache.npz"

    def _prepare_text_representation(self, facet: Dict[str, Any]) -> str:
        """
        Enriched Representation: normalized_facet + raw_facet + facet_type + scoring_definition + keywords
        """
        name = facet.get("normalized_facet", "")
        raw_name = facet.get("raw_facet", "")
        ftype = facet.get("facet_type", "conversational_trait")
        desc = facet.get("scoring_definition", "")
        reason = facet.get("abstention_reason", "")
        keywords_str = " ".join(facet.get("keywords", []))
        return f"Facet Name: {name} | Raw Facet: {raw_name} | Category: {ftype} | Definition: {desc} | Keywords: {keywords_str} {reason}"

    def fit(self, facet_docs: List[Dict[str, Any]]) -> "DenseVectorIndexer":
        self.facet_docs = facet_docs
        texts = [self._prepare_text_representation(f) for f in facet_docs]

        # 1. Check disk cache first for fast initialization
        if self.cache_path.exists():
            try:
                cached_data = np.load(self.cache_path, allow_pickle=True)
                if len(cached_data["embeddings"]) == len(texts):
                    self.embeddings = cached_data["embeddings"]
                    self.backend_type = str(cached_data["backend_type"])
                    print(f"Loaded {len(self.embeddings)} precomputed dense embeddings from disk cache.")
                    return self
            except Exception:
                pass

        # 2. Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Initializing SentenceTransformer dense vector model: '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            self.backend_type = "sentence_transformers"
        except Exception:
            # 3. Fallback to Scikit-Learn TF-IDF
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                print("SentenceTransformers not available. Using Scikit-Learn TF-IDF vectorizer fallback...")
                self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.embeddings = matrix.toarray()
                self.backend_type = "sklearn_tfidf"
            except Exception:
                # 4. Fallback to Pure Python TF Vectorizer
                print("ML libraries not available. Using 3rd-tier PurePythonTFVectorizer fallback...")
                self.pure_vectorizer = PurePythonTFVectorizer()
                self.embeddings = self.pure_vectorizer.fit_transform(texts)
                self.backend_type = "pure_python_tf"

        # Save to disk cache
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path, embeddings=self.embeddings, backend_type=self.backend_type)
        except Exception:
            pass

        return self

    def search(self, query: str, top_k: int = 30) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Encode query
        if self.backend_type == "sentence_transformers" and self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        elif self.backend_type == "sklearn_tfidf" and self.tfidf_vectorizer is not None:
            q_emb = self.tfidf_vectorizer.transform([query]).toarray()[0]
        elif self.backend_type == "pure_python_tf" and self.pure_vectorizer is not None:
            q_emb = self.pure_vectorizer.transform_single(query)[0]
        else:
            return []

        # Compute cosine similarity
        norm_q = np.linalg.norm(q_emb)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)

        if norm_q == 0:
            similarities = np.zeros(len(self.embeddings))
        else:
            denom = norm_docs * norm_q
            denom[denom == 0] = 1e-9
            similarities = np.dot(self.embeddings, q_emb) / denom

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.facet_docs[idx], float(similarities[idx])))

        return results
