import math
import re
from typing import Dict, List, Any, Set, Tuple, Optional
from ir_engine.preprocessor import TextPreprocessor

class InvertedIndex:
    """
    Positional Inverted Index with comprehensive statistics and term dictionary.
    Supports:
    - Positional postings lists: term -> {doc_id: {'tf': int, 'positions': [int]}}
    - Document Frequency df(t) and Collection Frequency cf(t)
    - Document lengths and average document length (L_avg)
    - Wildcard / Pattern matching (e.g., 'farm*', '*ship')
    - Levenshtein edit distance for spelling correction / Did You Mean
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        # Postings structure: term -> {doc_id: {"tf": int, "positions": List[int]}}
        self.index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Document store: doc_id -> raw scheme dict
        self.documents: Dict[str, Dict[str, Any]] = {}
        # Token count per document
        self.doc_lengths: Dict[str, int] = {}
        # Preprocessed token lists per document
        self.doc_tokens: Dict[str, List[str]] = {}
        # Total collection statistics
        self.total_tokens: int = 0
        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0

    def _prepare_document_text(self, scheme: Dict[str, Any]) -> str:
        """
        Combines and weights document fields into a consolidated text representation.
        Field weighting is applied by repeating prominent fields like Name and Keywords.
        """
        name = scheme.get("name", "")
        keywords = " ".join(scheme.get("keywords", []))
        category = scheme.get("category", "")
        target = scheme.get("target_beneficiaries", "")
        department = scheme.get("department", "")
        level = scheme.get("level", "")
        description = scheme.get("description", "")
        eligibility = scheme.get("eligibility", "")
        benefits = scheme.get("benefits", "")
        documents = scheme.get("required_documents", "")

        # Give higher semantic emphasis to Title and Keywords
        combined_text = (
            f"{name} {name} {keywords} {keywords} "
            f"{category} {level} {department} "
            f"{target} {description} {eligibility} {benefits} {documents}"
        )
        return combined_text

    def build_index(self, schemes: List[Dict[str, Any]]) -> None:
        """Indexes the entire collection of government schemes"""
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()
        self.doc_tokens.clear()
        self.total_tokens = 0

        for scheme in schemes:
            doc_id = scheme["id"]
            self.documents[doc_id] = scheme
            doc_text = self._prepare_document_text(scheme)
            tokens = self.preprocessor.preprocess(doc_text)
            self.doc_tokens[doc_id] = tokens
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            self.total_tokens += doc_len

            # Index tokens with their positions
            for pos, term in enumerate(tokens):
                if term not in self.index:
                    self.index[term] = {}
                if doc_id not in self.index[term]:
                    self.index[term][doc_id] = {"tf": 0, "positions": []}
                self.index[term][doc_id]["tf"] += 1
                self.index[term][doc_id]["positions"].append(pos)

        self.num_docs = len(self.documents)
        self.avg_doc_length = (self.total_tokens / self.num_docs) if self.num_docs > 0 else 0.0

    def get_doc_freq(self, term: str) -> int:
        """Returns document frequency df(t) for a given term"""
        return len(self.index.get(term, {}))

    def get_collection_freq(self, term: str) -> int:
        """Returns collection frequency cf(t) (total occurrences across all docs)"""
        postings = self.index.get(term, {})
        return sum(item["tf"] for item in postings.values())

    def get_postings(self, term: str) -> Dict[str, Dict[str, Any]]:
        """Returns the postings map for a term: doc_id -> {tf, positions}"""
        return self.index.get(term, {})

    def get_term_stats(self, term: str) -> Dict[str, Any]:
        """Returns detailed statistics for a term for UI inspection"""
        postings = self.get_postings(term)
        df = len(postings)
        cf = sum(item["tf"] for item in postings.values())
        idf = math.log((self.num_docs + 1) / (df + 1)) + 1 if df > 0 else 0.0

        sample_postings = []
        for doc_id, data in sorted(postings.items(), key=lambda x: x[1]["tf"], reverse=True)[:15]:
            scheme = self.documents[doc_id]
            sample_postings.append({
                "doc_id": doc_id,
                "scheme_name": scheme.get("name", ""),
                "category": scheme.get("category", ""),
                "tf": data["tf"],
                "positions": data["positions"][:10]  # first 10 positions
            })

        return {
            "term": term,
            "document_frequency": df,
            "collection_frequency": cf,
            "idf": round(idf, 4),
            "total_documents": self.num_docs,
            "postings_count": len(postings),
            "postings": sample_postings
        }

    def get_index_summary(self) -> Dict[str, Any]:
        """Returns comprehensive summary of the index for the IR dashboard"""
        vocab_size = len(self.index)
        top_terms = sorted(
            [{"term": t, "df": len(p), "cf": sum(x["tf"] for x in p.values())} for t, p in self.index.items()],
            key=lambda x: x["df"],
            reverse=True
        )[:20]

        return {
            "num_documents": self.num_docs,
            "vocabulary_size": vocab_size,
            "total_tokens": self.total_tokens,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "top_terms": top_terms
        }

    # ---------------- Pattern Matching & Wildcard Searches ----------------
    def wildcard_search(self, pattern: str) -> List[str]:
        """
        Pattern matching support for wildcard queries:
        Supports trailing wildcard (e.g. 'farm*'), leading wildcard ('*ship'),
        or inner wildcard ('sch*lar').
        """
        # Convert wildcard pattern into regex
        regex_pattern = "^" + pattern.replace("*", ".*").replace("?", ".") + "$"
        compiled_regex = re.compile(regex_pattern, re.IGNORECASE)

        matching_terms = [term for term in self.index.keys() if compiled_regex.match(term)]
        return sorted(matching_terms)

    # ---------------- Levenshtein Distance & Spell Suggestion ----------------
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Computes minimum edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        return dp[m][n]

    def suggest_spelling(self, term: str, max_distance: int = 2) -> Optional[str]:
        """
        Finds the closest term in index vocabulary using edit distance.
        Returns suggestion if candidate found, else None.
        """
        if term in self.index:
            return None  # Term is already valid

        candidates = []
        for vocab_term in self.index.keys():
            if abs(len(vocab_term) - len(term)) <= max_distance:
                dist = self.levenshtein_distance(term, vocab_term)
                if dist <= max_distance:
                    df = len(self.index[vocab_term])
                    candidates.append((vocab_term, dist, df))

        if not candidates:
            return None

        # Sort by lowest distance first, then highest document frequency
        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[0][0]
