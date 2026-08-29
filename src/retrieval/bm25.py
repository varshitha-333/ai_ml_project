"""
BM25 Lexical Indexer and Search Engine.

Implements Okapi BM25 ranking over preprocessed facet representations
(normalized_facet + facet_type + scoring_definition).
"""

import math
import re
from typing import List, Dict, Any, Tuple


class BM25Indexer:
    """
    Pure Python & rank_bm25 compatible Okapi BM25 Indexer.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.num_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Robust tokenizer splitting camelCase, hyphens, n-grams, and trailing symbols for multi-word phrase matching.
        """
        if not text:
            return []
        # Split camelCase: "HonestyHumility" -> "Honesty Humility"
        expanded_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        clean_text = expanded_text.lower().replace("-", " ")
        raw_tokens = re.findall(r'\b[a-z0-9]+\b', clean_text)

        # Generate Bigram n-grams for multi-word phrase matching
        bigrams = [f"{raw_tokens[i]}_{raw_tokens[i+1]}" for i in range(len(raw_tokens)-1)]

        orig_tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return list(dict.fromkeys(raw_tokens + orig_tokens + bigrams))

    def fit(self, documents: List[Dict[str, Any]]) -> "BM25Indexer":
        """
        Builds the BM25 index over a list of facet records.
        """
        self.corpus_documents = documents
        self.num_docs = len(documents)
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_freqs = {}

        total_len = 0
        for doc in documents:
            # Combine normalized name, facet type, and scoring definition for rich representation
            text = f"{doc.get('normalized_facet', '')} {doc.get('facet_type', '')} {doc.get('scoring_definition', '')}"
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            # Document frequency
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_len / max(self.num_docs, 1)

        # Precompute IDF values
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # Standard Okapi BM25 IDF formula
            idf_val = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[token] = max(idf_val, 0.01)

        return self

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calculates BM25 relevance scores for a query string across indexed documents.

        Returns:
            List[Tuple[Dict[str, Any], float]]: Ranked list of (document, bm25_score)
        """
        if not self.corpus_documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(doc, 0.0) for doc in self.corpus_documents[:top_k]]

        scores = [0.0] * self.num_docs

        for q_token in query_tokens:
            if q_token not in self.idf:
                continue
            q_idf = self.idf[q_token]

            for doc_idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(q_token)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1.0)))
                scores[doc_idx] += q_idf * (numerator / max(denominator, 1e-6))

        # Rank indices by score descending
        ranked_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)
        results = [(self.corpus_documents[i], scores[i]) for i in ranked_indices[:top_k]]
        return results
