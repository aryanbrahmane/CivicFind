import math
from typing import List, Dict, Any, Tuple
from ir_engine.indexer import InvertedIndex

class BM25Model:
    """
    Okapi BM25 Probabilistic Information Retrieval Model.
    
    BM25(q, d) = sum_{t in q} IDF(t) * [ f(t, d) * (k1 + 1) ] / [ f(t, d) + k1 * (1 - b + b * (|d| / L_avg)) ]
    
    Parameters:
    k1: controls term frequency saturation (default 1.5)
    b: controls document length normalization (default 0.75)
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.idf_cache: Dict[str, float] = {}
        self._compute_idfs()

    def _compute_idfs(self) -> None:
        """Precomputes Robertson-Spärck Jones IDF for all vocabulary terms"""
        self.idf_cache.clear()
        N = self.index.num_docs

        for term in self.index.index.keys():
            df = self.index.get_doc_freq(term)
            # RSJ IDF formula with smoothing to avoid negative weights
            val = (N - df + 0.5) / (df + 0.5)
            idf = math.log(max(val, 0.0) + 1.0)
            self.idf_cache[term] = idf

    def get_idf(self, term: str) -> float:
        """Returns BM25 IDF for a term"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        N = self.index.num_docs
        df = self.index.get_doc_freq(term)
        val = (N - df + 0.5) / (df + 0.5) if df > 0 else N + 0.5
        return math.log(max(val, 0.0) + 1.0)

    def search(self, query_tokens: List[str], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Ranks documents according to Okapi BM25 score.
        """
        if not query_tokens:
            return []

        doc_scores: Dict[str, float] = {}
        matching_terms_map: Dict[str, List[str]] = {}
        L_avg = self.index.avg_doc_length or 1.0
        k1 = self.k1
        b = self.b

        for term in query_tokens:
            postings = self.index.get_postings(term)
            if not postings:
                continue

            idf = self.get_idf(term)
            for doc_id, post_data in postings.items():
                f_td = post_data["tf"]
                doc_len = self.index.doc_lengths.get(doc_id, L_avg)

                # Length normalization component
                len_norm = 1.0 - b + b * (doc_len / L_avg)
                tf_component = (f_td * (k1 + 1.0)) / (f_td + k1 * len_norm)

                term_score = idf * tf_component
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

                if doc_id not in matching_terms_map:
                    matching_terms_map[doc_id] = []
                if term not in matching_terms_map[doc_id]:
                    matching_terms_map[doc_id].append(term)

        if not doc_scores:
            return []

        # Find max score for relative percentage normalization
        max_score = max(doc_scores.values()) if doc_scores else 1.0

        results: List[Dict[str, Any]] = []
        for doc_id, score in doc_scores.items():
            # Percentage relevance relative to theoretical top score
            relevance_pct = min(100.0, max(10.0, (score / max_score) * 100.0))

            results.append({
                "doc_id": doc_id,
                "scheme": self.index.documents[doc_id],
                "score": round(score, 4),
                "relevance_pct": round(relevance_pct, 1),
                "matching_terms": matching_terms_map.get(doc_id, []),
                "model": "Okapi BM25 Probabilistic Model"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
