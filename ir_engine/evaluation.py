import math
import json
import os
from typing import List, Dict, Any, Tuple, Set

class IREvaluator:
    """
    Comprehensive Information Retrieval Evaluation Suite.
    
    Computes standard IR effectiveness metrics:
    - Precision, Recall, F1-Score
    - Precision@K (P@1, P@3, P@5, P@10)
    - Recall@K
    - Average Precision (AP)
    - Mean Average Precision (MAP)
    - Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
    - Discounted Cumulative Gain (DCG@K)
    - Normalized Discounted Cumulative Gain (NDCG@K)
    """

    def __init__(self, ground_truth_file: str):
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            self.test_queries: List[Dict[str, Any]] = json.load(f)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_set: Set[str], k: int) -> float:
        """P@K: Proportion of top-K retrieved documents that are relevant"""
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_retrieved / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_set: Set[str], k: int) -> float:
        """R@K: Proportion of relevant documents retrieved in top-K"""
        if not relevant_set:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_retrieved / float(len(relevant_set))

    @staticmethod
    def f1_at_k(precision: float, recall: float) -> float:
        """Harmonic mean of precision and recall"""
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """
        AP: Area under precision-recall curve.
        AP = (1 / |Relevant|) * sum_{k=1}^N (P@k * rel(k))
        """
        if not relevant_set:
            return 0.0

        cum_precision = 0.0
        hits = 0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                cum_precision += precision_at_i

        return cum_precision / float(len(relevant_set)) if relevant_set else 0.0

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """RR: Reciprocal of the rank of the first relevant retrieved document"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def dcg_at_k(retrieved_ids: List[str], graded_rel_map: Dict[str, int], k: int) -> float:
        """
        DCG@K with exponential gain:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = graded_rel_map.get(doc_id, 0)
            numerator = (2 ** rel) - 1.0
            denominator = math.log2(i + 2.0)  # i=0 -> log2(2) = 1
            dcg += numerator / denominator
        return dcg

    @classmethod
    def ndcg_at_k(cls, retrieved_ids: List[str], graded_rel_map: Dict[str, int], k: int) -> float:
        """
        NDCG@K = DCG@K / IDCG@K
        where IDCG@K is the maximum possible DCG for the ideal ranking.
        """
        actual_dcg = cls.dcg_at_k(retrieved_ids, graded_rel_map, k)

        # Ideal ranking: sort all available relevant documents by graded score descending
        ideal_grades = sorted(graded_rel_map.values(), reverse=True)
        idcg = 0.0
        for i, grade in enumerate(ideal_grades[:k]):
            numerator = (2 ** grade) - 1.0
            denominator = math.log2(i + 2.0)
            idcg += numerator / denominator

        if idcg == 0.0:
            return 0.0
        return min(1.0, actual_dcg / idcg)

    def evaluate_model(self, model_name: str, search_func: Any) -> Dict[str, Any]:
        """
        Evaluates a search model against all benchmark test queries.
        `search_func` takes a query string and returns a list of result objects with 'doc_id'.
        """
        query_evaluations: List[Dict[str, Any]] = []

        total_p1 = 0.0
        total_p3 = 0.0
        total_p5 = 0.0
        total_p10 = 0.0
        total_r5 = 0.0
        total_r10 = 0.0
        total_ap = 0.0
        total_rr = 0.0
        total_ndcg5 = 0.0
        total_ndcg10 = 0.0

        n_queries = len(self.test_queries)

        for tq in self.test_queries:
            query_id = tq["query_id"]
            query_text = tq["query"]
            relevant_set = set(tq["relevant_docs"])
            graded_map = tq.get("graded_relevance", {doc_id: 1 for doc_id in relevant_set})

            # Execute search
            results = search_func(query_text)
            retrieved_ids = [r["doc_id"] for r in results]

            p1 = self.precision_at_k(retrieved_ids, relevant_set, 1)
            p3 = self.precision_at_k(retrieved_ids, relevant_set, 3)
            p5 = self.precision_at_k(retrieved_ids, relevant_set, 5)
            p10 = self.precision_at_k(retrieved_ids, relevant_set, 10)
            r5 = self.recall_at_k(retrieved_ids, relevant_set, 5)
            r10 = self.recall_at_k(retrieved_ids, relevant_set, 10)
            f1_5 = self.f1_at_k(p5, r5)
            f1_10 = self.f1_at_k(p10, r10)
            ap = self.average_precision(retrieved_ids, relevant_set)
            rr = self.reciprocal_rank(retrieved_ids, relevant_set)
            ndcg5 = self.ndcg_at_k(retrieved_ids, graded_map, 5)
            ndcg10 = self.ndcg_at_k(retrieved_ids, graded_map, 10)

            total_p1 += p1
            total_p3 += p3
            total_p5 += p5
            total_p10 += p10
            total_r5 += r5
            total_r10 += r10
            total_ap += ap
            total_rr += rr
            total_ndcg5 += ndcg5
            total_ndcg10 += ndcg10

            query_evaluations.append({
                "query_id": query_id,
                "query": query_text,
                "category": tq.get("category", ""),
                "num_relevant": len(relevant_set),
                "num_retrieved": len(retrieved_ids),
                "p@1": round(p1, 4),
                "p@3": round(p3, 4),
                "p@5": round(p5, 4),
                "p@10": round(p10, 4),
                "recall@5": round(r5, 4),
                "recall@10": round(r10, 4),
                "f1@5": round(f1_5, 4),
                "f1@10": round(f1_10, 4),
                "ap": round(ap, 4),
                "rr": round(rr, 4),
                "ndcg@5": round(ndcg5, 4),
                "ndcg@10": round(ndcg10, 4)
            })

        macro_metrics = {
            "model_name": model_name,
            "queries_evaluated": n_queries,
            "mean_p@1": round(total_p1 / n_queries, 4),
            "mean_p@3": round(total_p3 / n_queries, 4),
            "mean_p@5": round(total_p5 / n_queries, 4),
            "mean_p@10": round(total_p10 / n_queries, 4),
            "mean_recall@5": round(total_r5 / n_queries, 4),
            "mean_recall@10": round(total_r10 / n_queries, 4),
            "mean_f1@5": round(self.f1_at_k(total_p5 / n_queries, total_r5 / n_queries), 4),
            "mean_f1@10": round(self.f1_at_k(total_p10 / n_queries, total_r10 / n_queries), 4),
            "map": round(total_ap / n_queries, 4),
            "mrr": round(total_rr / n_queries, 4),
            "mean_ndcg@5": round(total_ndcg5 / n_queries, 4),
            "mean_ndcg@10": round(total_ndcg10 / n_queries, 4)
        }

        return {
            "summary": macro_metrics,
            "query_details": query_evaluations
        }
