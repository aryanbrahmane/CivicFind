import re
import string
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure required NLTK resources
try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except Exception:
    NLTK_STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
        "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
        "for", "with", "about", "against", "between", "into", "through", "during", "before", 
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
        "will", "just", "don", "should", "now"
    }

# Additional conversational & domain stopwords that provide little retrieval discrimination
DOMAIN_STOPWORDS = {
    "please", "find", "search", "looking", "want", "need", "get", "avail", 
    "give", "tell", "show", "available", "apply", "details", "information", "info"
}

CUSTOM_STOPWORDS = NLTK_STOPWORDS.union(DOMAIN_STOPWORDS)

class TextPreprocessor:
    """
    Complete Text Preprocessing Pipeline for Information Retrieval:
    1. Case Folding / Lowercasing
    2. Tokenization (alphanumeric extraction with hyphen handling)
    3. Stop-word Removal (standard + domain-specific)
    4. Stemming (Porter Stemmer)
    5. Lemmatization (WordNet Lemmatizer with POS tagging)
    """

    def __init__(self, use_stemming: bool = True, use_lemmatization: bool = True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stemmer = PorterStemmer()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    @staticmethod
    def get_wordnet_pos(treebank_tag: str):
        """Map POS tag to first character lemmatize() accepts"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize raw string into list of clean alphanumeric words,
        stripping punctuation while keeping alphanumeric tokens.
        """
        if not text:
            return []
        # Replace non-alphanumeric chars (except hyphens inside words) with space
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", text)
        # Split on whitespace and hyphens
        raw_tokens = re.split(r"[\s\-]+", cleaned)
        tokens = [t.strip() for t in raw_tokens if t.strip() and len(t.strip()) > 1]
        return tokens

    def preprocess(self, text: str, return_trace: bool = False) -> Any:
        """
        Executes full preprocessing pipeline.
        If return_trace=True, returns dictionary of transformation stages for UI visualization.
        """
        if not text:
            if return_trace:
                return {
                    "original": "",
                    "lowercased": "",
                    "tokens": [],
                    "no_stopwords": [],
                    "stemmed": [],
                    "lemmatized": [],
                    "final_tokens": []
                }
            return []

        # 1. Lowercasing
        lowercased = text.lower()

        # 2. Tokenization
        raw_tokens = self.tokenize(lowercased)

        # 3. Stop-word filtering
        no_stopwords = [t for t in raw_tokens if t not in CUSTOM_STOPWORDS]

        # 4. Lemmatization
        lemmatized = []
        if self.lemmatizer:
            try:
                pos_tags = nltk.pos_tag(no_stopwords)
                for word, tag in pos_tags:
                    pos = self.get_wordnet_pos(tag)
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos=pos))
            except Exception:
                lemmatized = [self.lemmatizer.lemmatize(t) for t in no_stopwords]
        else:
            lemmatized = list(no_stopwords)

        # 5. Stemming (optional / canonical reduction)
        stemmed = [self.stemmer.stem(t) for t in lemmatized]

        # Final tokens used in index/query (we default to lemmatized for human readability,
        # or stemmer if requested)
        final_tokens = lemmatized if self.use_lemmatization else stemmed

        if return_trace:
            return {
                "original": text,
                "lowercased": lowercased,
                "tokens": raw_tokens,
                "no_stopwords": no_stopwords,
                "lemmatized": lemmatized,
                "stemmed": stemmed,
                "final_tokens": final_tokens
            }

        return final_tokens

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocesses user query using identical steps as document indexing"""
        return self.preprocess(query, return_trace=False)
