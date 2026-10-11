import re
from typing import List, Dict, Any, Set, Optional
from ir_engine.indexer import InvertedIndex
from ir_engine.preprocessor import TextPreprocessor

class BooleanModel:
    """
    Classical Boolean Retrieval Model.
    Supports:
    - Boolean operators: AND, OR, NOT (case-insensitive)
    - Parentheses grouping: e.g. 'farmer AND (subsidy OR loan)'
    - Exact postings list intersection (AND), union (OR), and difference (NOT)
    """

    def __init__(self, index: InvertedIndex, preprocessor: Optional[TextPreprocessor] = None):
        self.index = index
        self.preprocessor = preprocessor or TextPreprocessor()

    def _get_term_doc_set(self, term: str) -> Set[str]:
        """Preprocesses term and retrieves set of matching doc_ids from inverted index"""
        tokens = self.preprocessor.preprocess(term)
        if not tokens:
            return set()
        clean_term = tokens[0]
        postings = self.index.get_postings(clean_term)
        return set(postings.keys())

    def _tokenize_boolean_query(self, query_str: str) -> List[str]:
        """Tokenizes boolean query into operators, parentheses, and terms"""
        # Insert spaces around parentheses
        q = query_str.replace("(", " ( ").replace(")", " ) ")
        raw_tokens = q.split()
        return raw_tokens

    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """Converts infix boolean tokens to postfix (Reverse Polish Notation) using Shunting-Yard"""
        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        output: List[str] = []
        op_stack: List[str] = []

        # Insert implicit AND if two terms appear consecutively without operator
        normalized_tokens: List[str] = []
        for i, token in enumerate(tokens):
            normalized_tokens.append(token)
            if i < len(tokens) - 1:
                curr_upper = token.upper()
                next_upper = tokens[i + 1].upper()
                if curr_upper not in ("AND", "OR", "NOT", "(") and next_upper not in ("AND", "OR", ")"):
                    normalized_tokens.append("AND")

        for token in normalized_tokens:
            token_upper = token.upper()
            if token_upper in precedence:
                while op_stack and op_stack[-1] != "(" and precedence.get(op_stack[-1], 0) >= precedence[token_upper]:
                    output.append(op_stack.pop())
                op_stack.append(token_upper)
            elif token == "(":
                op_stack.append("(")
            elif token == ")":
                while op_stack and op_stack[-1] != "(":
                    output.append(op_stack.pop())
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
            else:
                output.append(token)

        while op_stack:
            output.append(op_stack.pop())

        return output

    def evaluate_boolean(self, query_str: str) -> Set[str]:
        """Evaluates boolean query string and returns set of matching doc_ids"""
        tokens = self._tokenize_boolean_query(query_str)
        if not tokens:
            return set()

        postfix = self._to_postfix(tokens)
        stack: List[Set[str]] = []
        all_docs = set(self.index.documents.keys())

        for token in postfix:
            token_upper = token.upper()
            if token_upper == "NOT":
                if stack:
                    s = stack.pop()
                    stack.append(all_docs - s)
                else:
                    stack.append(all_docs)
            elif token_upper == "AND":
                if len(stack) >= 2:
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left.intersection(right))
                elif stack:
                    # Single operand fallback
                    pass
            elif token_upper == "OR":
                if len(stack) >= 2:
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left.union(right))
                elif stack:
                    pass
            else:
                # Leaf term
                stack.append(self._get_term_doc_set(token))

        return stack[-1] if stack else set()

    def search(self, query_str: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Executes Boolean retrieval.
        Scores matching documents by number of distinct query terms matched.
        """
        matched_doc_ids = self.evaluate_boolean(query_str)
        if not matched_doc_ids:
            return []

        # Find query terms for highlighting
        clean_terms = self.preprocessor.preprocess(query_str)

        results: List[Dict[str, Any]] = []
        for doc_id in matched_doc_ids:
            scheme = self.index.documents[doc_id]
            doc_tokens = self.index.doc_tokens.get(doc_id, [])
            term_overlap = [t for t in clean_terms if t in doc_tokens]

            results.append({
                "doc_id": doc_id,
                "scheme": scheme,
                "score": 1.0,
                "relevance_pct": 100.0,
                "matching_terms": term_overlap,
                "model": "Classical Boolean Retrieval Model"
            })

        # Sort alphabetically or by term overlap count
        results.sort(key=lambda x: len(x["matching_terms"]), reverse=True)
        return results[:top_k]
