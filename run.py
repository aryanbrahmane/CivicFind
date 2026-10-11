import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import nltk
    for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
        try:
            nltk.data.find(f'corpora/{pkg}' if pkg in ['stopwords', 'wordnet'] else f'tokenizers/{pkg}')
        except LookupError:
            print(f"Downloading required NLTK resource: {pkg}...")
            nltk.download(pkg, quiet=True)
except Exception as e:
    print(f"NLTK check notice: {e}")

from app import app

if __name__ == "__main__":
    print("=" * 65)
    print("  CIVICFIND: Intelligent Information Retrieval System")
    print("  Government Schemes & Public Services Discovery")
    print("=" * 65)
    print("  Server running at: http://127.0.0.1:5000")
    print("  Access the CivicFind web interface in your browser!")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False)
