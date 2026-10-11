import math
from typing import List, Dict, Any, Tuple
from collections import Counter
from ir_engine.indexer import InvertedIndex

class VectorSpaceModel:
    """
    Vector Space Model (VSM) using TF-IDF weighting and Cosine Similarity.
    
    TF(t, d) = 1 + log(tf) if tf > 0 else 0 (Sublinear/log frequency scaling)
    IDF(t) = log((N + 1) / (df + 1)) + 1
    Weight(t, d) = TF(t, d) * IDF(t)
    
    Cosine Similarity: cos(q, d) = (q . d) / (||q|| * ||d||)
    """

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.doc_magnitudes: Dict[str, float] = {}
        self._compute_document_vectors()

    def _compute_document_vectors(self) -> None:
        """Precomputes TF-IDF vectors and Euclidean norms for all indexed documents"""
        self.doc_vectors.clear()
        self.doc_magnitudes.clear()
        N = self.index.num_docs

        for doc_id, scheme in self.index.documents.items():
            vec: Dict[str, float] = {}
            tokens = self.index.doc_tokens.get(doc_id, [])
            term_counts = Counter(tokens)
            sum_sq = 0.0

            for term, count in term_counts.items():
                df = self.index.get_doc_freq(term)
                # Sublinear TF scaling
                tf = 1.0 + math.log10(count) if count > 0 else 0.0
                # Smoothed IDF
                idf = math.log10((N + 1.0) / (df + 1.0)) + 1.0
                weight = tf * idf
                vec[term] = weight
                sum_sq += weight * weight

            self.doc_vectors[doc_id] = vec
            self.doc_magnitudes[doc_id] = math.sqrt(sum_sq) if sum_sq > 0 else 1.0

    def compute_query_vector(self, query_tokens: List[str]) -> Tuple[Dict[str, float], float]:
        """Computes TF-IDF vector and magnitude for the query"""
        query_counts = Counter(query_tokens)
        q_vec: Dict[str, float] = {}
        sum_sq = 0.0
        N = self.index.num_docs

        for term, count in query_counts.items():
            df = self.index.get_doc_freq(term)
            tf = 1.0 + math.log10(count) if count > 0 else 0.0
            idf = math.log10((N + 1.0) / (df + 1.0)) + 1.0 if df > 0 else 1.0
            weight = tf * idf
            q_vec[term] = weight
            sum_sq += weight * weight

        magnitude = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
        return q_vec, magnitude

    def search(self, query_tokens: List[str], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity between query and all candidate documents.
        Returns ranked documents with scores and computation details.
        """
        if not query_tokens:
            return []

        q_vec, q_mag = self.compute_query_vector(query_tokens)
        candidate_scores: Dict[str, float] = {}
        matching_terms_map: Dict[str, List[str]] = {}

        # Accumulate dot product only over candidate documents containing at least one query term
        for term, q_weight in q_vec.items():
            postings = self.index.get_postings(term)
            for doc_id, post_info in postings.items():
                d_weight = self.doc_vectors.get(doc_id, {}).get(term, 0.0)
                dot_contrib = q_weight * d_weight
                candidate_scores[doc_id] = candidate_scores.get(doc_id, 0.0) + dot_contrib

                if doc_id not in matching_terms_map:
                    matching_terms_map[doc_id] = []
                matching_terms_map[doc_id].append(term)

        results: List[Dict[str, Any]] = []
        for doc_id, dot_product in candidate_scores.items():
            d_mag = self.doc_magnitudes.get(doc_id, 1.0)
            cosine_sim = dot_product / (q_mag * d_mag) if (q_mag * d_mag) > 0 else 0.0

            # Scale to percentage (0 - 100%)
            relevance_pct = min(100.0, max(0.0, cosine_sim * 100.0))

            results.append({
                "doc_id": doc_id,
                "scheme": self.index.documents[doc_id],
                "score": round(cosine_sim, 4),
                "relevance_pct": round(relevance_pct, 1),
                "matching_terms": matching_terms_map.get(doc_id, []),
                "model": "Vector Space Model (TF-IDF Cosine)"
            })

        # Rank documents descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
