import os
import json
import time
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template, send_from_directory

from ir_engine.preprocessor import TextPreprocessor
from ir_engine.indexer import InvertedIndex
from ir_engine.query_reformulation import QueryReformulator
from ir_engine.evaluation import IREvaluator
from ir_engine.models.vsm import VectorSpaceModel
from ir_engine.models.bm25 import BM25Model
from ir_engine.models.boolean import BooleanModel
from ir_engine.models.semantic import SemanticLSIModel
from ir_engine.models.hybrid import HybridSearchModel

app = Flask(__name__, static_folder="static", template_folder="templates")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMES_FILE = os.path.join(BASE_DIR, "data", "schemes.json")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")

# Global IR Engine Singletons
print("Loading CivicFind datasets and initializing IR Engine...")
with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
    schemes_data: List[Dict[str, Any]] = json.load(f)

preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=True)
index = InvertedIndex(preprocessor=preprocessor)
index.build_index(schemes_data)

vsm_model = VectorSpaceModel(index)
bm25_model = BM25Model(index, k1=1.5, b=0.75)
boolean_model = BooleanModel(index, preprocessor)
semantic_model = SemanticLSIModel(index, n_components=25)
hybrid_model = HybridSearchModel(bm25_model, semantic_model, alpha=0.65)
reformulator = QueryReformulator(index, preprocessor)
evaluator = IREvaluator(GROUND_TRUTH_FILE)

print(f"CivicFind IR Engine ready! Indexed {index.num_docs} documents, vocabulary {len(index.index)} terms.")

# ----------------- ROUTES -----------------

@app.route("/")
def index_page():
    """Renders the main CivicFind application interface"""
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search_schemes():
    """
    Search endpoint supporting multiple IR models, query expansion,
    filtering by category and level, and transparent retrieval diagnostics.
    """
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    model_name = data.get("model", "bm25").lower()  # bm25, vsm, boolean, semantic, hybrid
    use_expansion = bool(data.get("use_expansion", False))
    top_k = int(data.get("top_k", 25))
    filters = data.get("filters", {})

    if not query:
        return jsonify({"results": [], "query_info": {"error": "Query cannot be empty"}})

    # 1. Check for spelling corrections / 'Did You Mean?'
    spelling_info = reformulator.suggest_query_spelling(query)

    # 2. Query expansion
    expansion_info = {}
    search_query_text = query
    if use_expansion:
        expansion_info = reformulator.expand_query(query)
        search_query_text = expansion_info["combined_query_text"]

    query_tokens = preprocessor.preprocess(search_query_text)

    # 3. Model Retrieval
    raw_results: List[Dict[str, Any]] = []
    model_label = "Okapi BM25"

    if model_name == "vsm":
        model_label = "Vector Space Model (TF-IDF Cosine)"
        raw_results = vsm_model.search(query_tokens, top_k=top_k * 2)
    elif model_name == "boolean":
        model_label = "Classical Boolean Retrieval"
        raw_results = boolean_model.search(search_query_text, top_k=top_k * 2)
    elif model_name == "semantic":
        model_label = "Latent Semantic Indexing (LSI Dense Cosine)"
        raw_results = semantic_model.search(search_query_text, top_k=top_k * 2)
    elif model_name == "hybrid":
        model_label = "Hybrid IR (BM25 + LSI Semantic)"
        raw_results = hybrid_model.search(search_query_text, query_tokens, top_k=top_k * 2)
    else:  # default bm25
        model_label = "Okapi BM25 Probabilistic Model"
        raw_results = bm25_model.search(query_tokens, top_k=top_k * 2)

    # 4. Apply Filters (Category, Level, Target Beneficiaries)
    cat_filter = filters.get("category")
    level_filter = filters.get("level")
    target_filter = filters.get("target")

    filtered_results = []
    for r in raw_results:
        sch = r["scheme"]
        if cat_filter and cat_filter != "All" and sch.get("category") != cat_filter:
            continue
        if level_filter and level_filter != "All" and level_filter.lower() not in sch.get("level", "").lower():
            continue
        if target_filter and target_filter.lower() not in sch.get("target_beneficiaries", "").lower():
            continue
        filtered_results.append(r)

    final_results = filtered_results[:top_k]
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Preprocessing Trace for IR Inspector
    prep_trace = preprocessor.preprocess(query, return_trace=True)

    return jsonify({
        "results": final_results,
        "query_info": {
            "query": query,
            "effective_query_text": search_query_text,
            "model_used": model_label,
            "model_code": model_name,
            "tokens": query_tokens,
            "preprocessing_trace": prep_trace,
            "expansion": expansion_info,
            "spelling_suggestion": spelling_info,
            "total_matches": len(filtered_results),
            "execution_time_ms": duration_ms
        }
    })

@app.route("/api/relevance-feedback", methods=["POST"])
def relevance_feedback():
    """
    Rocchio Relevance Feedback endpoint:
    Adjusts query vector based on relevant and non-relevant documents marked by user,
    and returns reformulated rankings and updated query terms.
    """
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    original_query = data.get("original_query", "").strip()
    relevant_ids = data.get("relevant_doc_ids", [])
    non_relevant_ids = data.get("non_relevant_doc_ids", [])
    top_k = int(data.get("top_k", 20))

    if not original_query or (not relevant_ids and not non_relevant_ids):
        return jsonify({"error": "Original query and at least one feedback document required"}), 400

    orig_tokens = preprocessor.preprocess(original_query)
    feedback_info = reformulator.apply_rocchio_feedback(
        original_query_tokens=orig_tokens,
        relevant_doc_ids=relevant_ids,
        non_relevant_doc_ids=non_relevant_ids,
        vsm_model=vsm_model
    )

    reformulated_tokens = feedback_info["reformulated_tokens"]
    # Re-rank using VSM with reformulated query
    re_ranked = vsm_model.search(reformulated_tokens, top_k=top_k)

    duration_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "results": re_ranked,
        "feedback_info": feedback_info,
        "execution_time_ms": duration_ms
    })

@app.route("/api/schemes", methods=["GET"])
def get_schemes():
    """Returns schemes list with optional category filter and pagination"""
    category = request.args.get("category")
    level = request.args.get("level")
    search_q = request.args.get("search", "").lower()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    filtered = schemes_data
    if category and category != "All":
        filtered = [s for s in filtered if s.get("category") == category]
    if level and level != "All":
        filtered = [s for s in filtered if level.lower() in s.get("level", "").lower()]
    if search_q:
        filtered = [s for s in filtered if search_q in s.get("name", "").lower() or search_q in s.get("description", "").lower()]

    total = len(filtered)
    start_idx = (page - 1) * per_page
    paginated = filtered[start_idx:start_idx + per_page]

    # Unique categories and levels for UI dropdowns
    categories = sorted(list(set(s.get("category", "") for s in schemes_data if s.get("category"))))
    levels = ["All", "Maharashtra State", "Central", "All-India"]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "schemes": paginated,
        "categories": categories,
        "levels": levels
    })

@app.route("/api/schemes/<doc_id>", methods=["GET"])
def get_scheme_detail(doc_id):
    """Returns single scheme full record"""
    scheme = index.documents.get(doc_id)
    if not scheme:
        return jsonify({"error": "Scheme not found"}), 404
    return jsonify(scheme)

@app.route("/api/index/stats", methods=["GET"])
def get_index_stats():
    """Returns index collection statistics for the IR dashboard"""
    return jsonify(index.get_index_summary())

@app.route("/api/index/term/<term>", methods=["GET"])
def get_term_postings(term):
    """Returns postings list and document frequencies for any inspected term"""
    clean_tokens = preprocessor.preprocess(term)
    lookup_term = clean_tokens[0] if clean_tokens else term.lower()
    stats = index.get_term_stats(lookup_term)
    return jsonify(stats)

@app.route("/api/evaluate", methods=["GET"])
def run_evaluation():
    """
    Evaluates and benchmarks all IR models against the ground-truth test suite:
    VSM, BM25, Boolean, Semantic LSI, Hybrid.
    Returns comparative table and Chart.js datasets.
    """
    # 1. BM25
    bm25_eval = evaluator.evaluate_model(
        "Okapi BM25",
        lambda q: bm25_model.search(preprocessor.preprocess(q), top_k=15)
    )

    # 2. VSM (TF-IDF Cosine)
    vsm_eval = evaluator.evaluate_model(
        "Vector Space Model (VSM)",
        lambda q: vsm_model.search(preprocessor.preprocess(q), top_k=15)
    )

    # 3. Boolean
    boolean_eval = evaluator.evaluate_model(
        "Boolean Retrieval",
        lambda q: boolean_model.search(q, top_k=15)
    )

    # 4. Semantic LSI
    semantic_eval = evaluator.evaluate_model(
        "Latent Semantic Indexing (LSI)",
        lambda q: semantic_model.search(q, top_k=15)
    )

    # 5. Hybrid
    hybrid_eval = evaluator.evaluate_model(
        "Hybrid Search (BM25 + LSI)",
        lambda q: hybrid_model.search(q, preprocessor.preprocess(q), top_k=15)
    )

    models_comparison = [
        bm25_eval["summary"],
        vsm_eval["summary"],
        semantic_eval["summary"],
        hybrid_eval["summary"],
        boolean_eval["summary"]
    ]

    return jsonify({
        "models_summary": models_comparison,
        "bm25_details": bm25_eval["query_details"],
        "hybrid_details": hybrid_eval["query_details"],
        "total_test_queries": len(evaluator.test_queries)
    })

@app.route("/api/preprocess-trace", methods=["POST"])
def preprocess_trace_endpoint():
    """Live interactive text preprocessing step trace"""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    trace = preprocessor.preprocess(text, return_trace=True)
    return jsonify(trace)

@app.route("/api/suggest", methods=["GET"])
def suggest_terms():
    """Prefix suggestion for instant search autocomplete"""
    prefix = request.args.get("q", "").strip().lower()
    if len(prefix) < 2:
        return jsonify([])

    matches = [t for t in index.index.keys() if t.startswith(prefix)][:10]
    return jsonify(matches)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
