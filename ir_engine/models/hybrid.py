from typing import List, Dict, Any
from ir_engine.models.bm25 import BM25Model
from ir_engine.models.semantic import SemanticLSIModel
from ir_engine.indexer import InvertedIndex

class HybridSearchModel:
    """
    Hybrid Retrieval Engine combining Lexical Okapi BM25 and Semantic LSI.
    
    Formula:
    Score_Hybrid = alpha * Normalized_BM25 + (1 - alpha) * Normalized_Semantic
    
    Combines high-precision exact keyword matching with semantic topical generalization.
    """

    def __init__(self, bm25_model: BM25Model, semantic_model: SemanticLSIModel, alpha: float = 0.65):
        self.bm25 = bm25_model
        self.semantic = semantic_model
        self.alpha = alpha

    def search(self, query_text: str, query_tokens: List[str], top_k: int = 50) -> List[Dict[str, Any]]:
        bm25_res = self.bm25.search(query_tokens, top_k=top_k * 2)
        semantic_res = self.semantic.search(query_text, top_k=top_k * 2)

        bm25_dict = {r["doc_id"]: r for r in bm25_res}
        sem_dict = {r["doc_id"]: r for r in semantic_res}

        # Normalize BM25 scores between 0 and 1
        max_bm25 = max([r["score"] for r in bm25_res], default=1.0) or 1.0
        # Semantic scores are already cosine [-1, 1], clip to [0, 1]
        
        all_doc_ids = set(bm25_dict.keys()).union(set(sem_dict.keys()))
        combined_results: List[Dict[str, Any]] = []

        for doc_id in all_doc_ids:
            bm25_item = bm25_dict.get(doc_id)
            sem_item = sem_dict.get(doc_id)

            bm25_score_norm = (bm25_item["score"] / max_bm25) if bm25_item else 0.0
            sem_score_norm = max(0.0, sem_item["score"]) if sem_item else 0.0

            hybrid_score = (self.alpha * bm25_score_norm) + ((1.0 - self.alpha) * sem_score_norm)

            scheme = (bm25_item or sem_item)["scheme"]
            matching_terms = list(set((bm25_item["matching_terms"] if bm25_item else []) + 
                                      (sem_item["matching_terms"] if sem_item else [])))

            combined_results.append({
                "doc_id": doc_id,
                "scheme": scheme,
                "score": round(hybrid_score, 4),
                "relevance_pct": round(min(100.0, max(0.0, hybrid_score * 100.0)), 1),
                "matching_terms": matching_terms,
                "model": "Hybrid IR (BM25 Lexical + LSI Semantic)",
                "bm25_score": round(bm25_item["score"], 3) if bm25_item else 0.0,
                "semantic_score": round(sem_item["score"], 3) if sem_item else 0.0
            })

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]
