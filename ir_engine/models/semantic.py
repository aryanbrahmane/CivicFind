import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from ir_engine.indexer import InvertedIndex

class SemanticLSIModel:
    """
    Latent Semantic Indexing (LSI) / Semantic Information Retrieval Model.
    
    Applies Singular Value Decomposition (SVD) on the Term-Document TF-IDF matrix
    to project terms and documents into a low-dimensional dense semantic concept space.
    
    This captures latent topical relationships and semantic synonyms, enabling
    the system to retrieve documents even when user terminology differs from the document wording.
    """

    def __init__(self, index: InvertedIndex, n_components: int = 25):
        self.index = index
        self.n_components = n_components
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            tokenizer=lambda text: self.index.preprocessor.preprocess(text),
            token_pattern=None,
            lowercase=True
        )
        self.svd: TruncatedSVD = None
        self.doc_ids: List[str] = []
        self.doc_dense_matrix: np.ndarray = None
        self.doc_dense_norms: np.ndarray = None
        self._fit_semantic_space()

    def _fit_semantic_space(self) -> None:
        """Constructs TF-IDF matrix and fits Truncated SVD semantic space"""
        self.doc_ids = list(self.index.documents.keys())
        corpus = [
            self.index._prepare_document_text(self.index.documents[doc_id])
            for doc_id in self.doc_ids
        ]

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        n_features = tfidf_matrix.shape[1]
        n_samples = tfidf_matrix.shape[0]

        # Ensure n_components is valid for dataset dimensions
        k = min(self.n_components, n_samples - 1, n_features - 1)
        if k < 2:
            k = 2

        self.svd = TruncatedSVD(n_components=k, random_state=42)
        self.doc_dense_matrix = self.svd.fit_transform(tfidf_matrix)

        # Precompute L2 norms for cosine similarity
        norms = np.linalg.norm(self.doc_dense_matrix, axis=1)
        norms[norms == 0] = 1.0  # avoid division by zero
        self.doc_dense_norms = norms

    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Projects query into latent semantic concept space and calculates
        cosine similarity against all dense document vectors.
        """
        if not query_text.strip():
            return []

        # Vectorize query
        q_tfidf = self.vectorizer.transform([query_text])
        # Project into latent semantic space
        q_dense = self.svd.transform(q_tfidf)[0]

        q_norm = np.linalg.norm(q_dense)
        if q_norm == 0:
            q_norm = 1.0

        # Cosine similarity in semantic concept space
        dot_products = np.dot(self.doc_dense_matrix, q_dense)
        cosine_sims = dot_products / (self.doc_dense_norms * q_norm)

        # Retrieve top-k
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        query_tokens = self.index.preprocessor.preprocess(query_text)

        for idx in top_indices:
            score = float(cosine_sims[idx])
            doc_id = self.doc_ids[idx]
            scheme = self.index.documents[doc_id]

            # Matching terms for highlight
            doc_tokens = self.index.doc_tokens.get(doc_id, [])
            term_overlap = [t for t in query_tokens if t in doc_tokens]

            relevance_pct = min(100.0, max(0.0, score * 100.0))

            results.append({
                "doc_id": doc_id,
                "scheme": scheme,
                "score": round(score, 4),
                "relevance_pct": round(relevance_pct, 1),
                "matching_terms": term_overlap,
                "model": f"Latent Semantic Indexing (Dense SVD k={self.svd.n_components})"
            })

        return results
