import math
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
from ir_engine.indexer import InvertedIndex
from ir_engine.preprocessor import TextPreprocessor

# Government and public-service domain thesaurus mapping
DOMAIN_THESAURUS = {
    "student": ["scholarship", "higher", "education", "college", "tuition", "fee", "hostel"],
    "students": ["scholarship", "higher", "education", "college", "tuition", "fee", "hostel"],
    "scholarship": ["financial", "assistance", "tuition", "fee", "waiver", "stipend", "merit"],
    "farmer": ["kisan", "agriculture", "crop", "subsidy", "irrigation", "pump", "shettale"],
    "farmers": ["kisan", "agriculture", "crop", "subsidy", "irrigation", "pump", "shetkari"],
    "agriculture": ["farming", "crop", "kisan", "irrigation", "tractor", "fertilizer", "soil"],
    "women": ["mahila", "ladki", "girl", "mother", "maternity", "shg", "bachat"],
    "woman": ["mahila", "ladki", "girl", "mother", "maternity", "shg", "bachat"],
    "girl": ["kanya", "sukanya", "ladki", "female", "women", "daughter"],
    "hospital": ["healthcare", "medical", "cashless", "surgery", "treatment", "doctor"],
    "health": ["hospital", "medical", "treatment", "ayushman", "insurance", "medicine"],
    "loan": ["credit", "mudra", "interest", "subvention", "collateral", "capital", "subsidy"],
    "business": ["enterprise", "msme", "startup", "entrepreneur", "shopkeeper", "trade"],
    "unemployed": ["youth", "internship", "skill", "training", "stipend", "apprenticeship"],
    "job": ["employment", "apprenticeship", "internship", "career", "skill", "stipend"],
    "youth": ["internship", "apprentice", "stipend", "skill", "training", "employment"],
    "house": ["housing", "awas", "gharkul", "pmay", "pucca", "construction"],
    "housing": ["awas", "gharkul", "pmay", "pucca", "shelter", "construction"],
    "pension": ["senior", "elderly", "destitute", "niradhar", "shravanbal", "monthly"],
    "elderly": ["senior", "citizen", "pension", "old", "age", "vayoshri"],
    "disabled": ["divyang", "disability", "assistive", "wheelchair", "udid", "handicapped"],
    "disability": ["divyang", "assistive", "wheelchair", "handicap", "tricycle", "udid"],
    "solar": ["surya", "electricity", "rooftop", "pump", "kusum", "power", "subsidy"],
    "free": ["grant", "waiver", "subsidy", "assistance", "reimbursement", "zero"]
}

class QueryReformulator:
    """
    Query Reformulation Engine:
    1. Query Expansion (Domain Thesaurus & Synonym Injection)
    2. Rocchio Relevance Feedback (Vector space query vector updating)
    3. Spelling Correction & 'Did You Mean?' suggestions
    """

    def __init__(self, index: InvertedIndex, preprocessor: Optional[TextPreprocessor] = None):
        self.index = index
        self.preprocessor = preprocessor or TextPreprocessor()

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expands user query using domain thesaurus.
        Returns original tokens, added terms, combined expanded query, and expansion explanations.
        """
        raw_tokens = self.preprocessor.tokenize(query.lower())
        expanded_terms: List[str] = []
        explanation: List[str] = []

        seen_terms = set(raw_tokens)

        for token in raw_tokens:
            stemmed_or_lemmatized = self.preprocessor.preprocess(token)
            base_term = stemmed_or_lemmatized[0] if stemmed_or_lemmatized else token

            # Check direct or base token in thesaurus
            synonyms = DOMAIN_THESAURUS.get(token) or DOMAIN_THESAURUS.get(base_term)
            if synonyms:
                added_for_term = []
                for syn in synonyms:
                    if syn not in seen_terms:
                        expanded_terms.append(syn)
                        seen_terms.add(syn)
                        added_for_term.append(syn)
                if added_for_term:
                    explanation.append(f"'{token}' -> [{', '.join(added_for_term[:3])}]")

        combined_text = query + " " + " ".join(expanded_terms)
        combined_tokens = self.preprocessor.preprocess(combined_text)

        return {
            "original_query": query,
            "expanded_terms": expanded_terms,
            "combined_query_text": combined_text,
            "combined_tokens": combined_tokens,
            "explanation": " | ".join(explanation) if explanation else "No expansion needed"
        }

    def suggest_query_spelling(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Checks each word in query against inverted index vocabulary.
        If unrecognized terms found with close Levenshtein matches, suggests corrected query.
        """
        words = query.strip().split()
        corrected_words = []
        has_correction = False

        for word in words:
            # Strip punctuation for check
            clean = "".join(c for c in word.lower() if c.isalnum())
            if not clean:
                corrected_words.append(word)
                continue

            tokens = self.preprocessor.preprocess(clean)
            if tokens and tokens[0] in self.index.index:
                corrected_words.append(word)
            else:
                suggestion = self.index.suggest_spelling(clean)
                if suggestion and suggestion != clean:
                    corrected_words.append(suggestion)
                    has_correction = True
                else:
                    corrected_words.append(word)

        if has_correction:
            suggested_query = " ".join(corrected_words)
            return {
                "original": query,
                "suggested": suggested_query
            }
        return None

    def apply_rocchio_feedback(
        self,
        original_query_tokens: List[str],
        relevant_doc_ids: List[str],
        non_relevant_doc_ids: List[str],
        vsm_model: Any,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.25,
        top_expanded_terms: int = 5
    ) -> Dict[str, Any]:
        """
        Rocchio Relevance Feedback Algorithm:
        q_new = alpha * q_0 + (beta / |D_r|) * sum(d_r) - (gamma / |D_nr|) * sum(d_nr)
        
        Returns updated query terms and weight modifications.
        """
        q_0, _ = vsm_model.compute_query_vector(original_query_tokens)
        q_new: Dict[str, float] = {t: alpha * w for t, w in q_0.items()}

        # Accumulate relevant document vectors
        if relevant_doc_ids:
            num_rel = len(relevant_doc_ids)
            for doc_id in relevant_doc_ids:
                d_vec = vsm_model.doc_vectors.get(doc_id, {})
                for term, weight in d_vec.items():
                    contrib = (beta / num_rel) * weight
                    q_new[term] = q_new.get(term, 0.0) + contrib

        # Subtract non-relevant document vectors
        if non_relevant_doc_ids:
            num_nrel = len(non_relevant_doc_ids)
            for doc_id in non_relevant_doc_ids:
                d_vec = vsm_model.doc_vectors.get(doc_id, {})
                for term, weight in d_vec.items():
                    sub = (gamma / num_nrel) * weight
                    q_new[term] = max(0.0, q_new.get(term, 0.0) - sub)

        # Remove zero or negative weights
        q_new = {t: w for t, w in q_new.items() if w > 0.001}

        # Select highest weighted terms to form reformulated query tokens
        sorted_terms = sorted(q_new.items(), key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _ in sorted_terms[:max(len(original_query_tokens) + 4, 10)]]

        added_terms = [t for t in top_terms if t not in original_query_tokens][:top_expanded_terms]

        return {
            "reformulated_tokens": top_terms,
            "added_terms": added_terms,
            "term_weights": {t: round(w, 4) for t, w in sorted_terms[:15]},
            "rocchio_parameters": {"alpha": alpha, "beta": beta, "gamma": gamma}
        }
