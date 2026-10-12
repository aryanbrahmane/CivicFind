# CIVICFIND: An Intelligent Information Retrieval System for Government Schemes and Public Services

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![IR Lab Project](https://img.shields.io/badge/Subject-Information%20Retrieval%20Lab-emerald.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Lab-purple.svg)]()

> **CivicFind** is an end-to-end, citizen-centric Information Retrieval (IR) system designed to discover, retrieve, rank, and evaluate government welfare schemes, scholarships, agricultural subsidies, healthcare insurance, and public services using natural-language search queries.

---

## 🏛️ 1. Project Overview & Motivation

Government welfare information is distributed across hundreds of departmental portals, gazette notifications, and complex PDF circulars. Citizens often do not know the exact bureaucratic or statutory name of a scheme (e.g., searching *"financial help for college students after 12th in Maharashtra"* instead of *"Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojna"*).

**CivicFind** bridges this information asymmetry through modern Information Retrieval techniques:
- Accepts natural language queries.
- Preprocesses and normalizes text (tokenization, stop-word filtering, lemmatization, stemming).
- Indexes 174 verified Central and Maharashtra State government schemes in a positional inverted index.
- Implements and compares multiple IR models (**Vector Space Model TF-IDF, Okapi BM25, Boolean Retrieval, Semantic LSI SVD, and Hybrid Search**).
- Supports **Query Reformulation** (Domain Thesaurus Query Expansion + Rocchio Relevance Feedback).
- Evaluates retrieval effectiveness using standard IR metrics: **Precision@K, Recall@K, F1-Score, Average Precision (AP), Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG@K), and MRR**.

---

## 🔬 2. Information Retrieval Concepts & Architecture

```
                    ┌───────────────────────────────┐
                    │     CITIZEN NATURAL QUERY     │
                    │   e.g. "farmer crop loan"    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    TEXT PREPROCESSING TRACE   │
                    │ Lowercase ➔ Tokenize ➔        │
                    │ Filter Stopwords ➔ Lemmatize  │
                    └───────────────┬───────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│       QUERY EXPANSION        │          │      ROCCHIO FEEDBACK        │
│ Domain Thesaurus Injection   │          │ Relevance Vector Adjustment  │
│ (farmer ➔ kisan, agriculture)│          │  q_new = α·q_0 + β·Dr - γ·Dnr│
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                 PLUGGABLE IR MODEL SUITE                │
       ├────────────────────────────┬────────────────────────────┤
       │ 1. Vector Space Model      │ TF-IDF + Cosine Similarity │
       │ 2. Okapi BM25              │ Probabilistic + Length Norm│
       │ 3. Boolean Model           │ Exact AND / OR / NOT sets  │
       │ 4. Semantic LSI            │ Dense SVD Latent Concepts  │
       │ 5. Hybrid Search           │ α·BM25 + (1-α)·LSI Semantic│
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                POSITIONAL INVERTED INDEX                │
       │  term ➔ {doc_id: {'tf': int, 'positions': [p1, p2]}}    │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │               RELEVANCE RANKING & DISPLAY               │
       │  Ranked Schemes + % Match + Eligibility + Official Link │
       └─────────────────────────────────────────────────────────┘
```

---

## 📐 3. Mathematical Foundations & IR Models Implemented

### 3.1 Text Preprocessing Pipeline
1. **Case Folding / Lowercasing:** Ensures uniform case representation.
2. **Tokenization:** Alphanumeric regex splitting `[\s\-]+` preserving hyphenated concepts.
3. **Stop-word Removal:** NLTK standard English stop-words supplemented by domain-neutral query words (`please`, `find`, `apply`, `details`).
4. **Lemmatization:** WordNet POS-aware morphological reduction to true root words.
5. **Stemming:** Porter Stemmer algorithm for canonical suffixes.

---

### 3.2 Positional Inverted Index
Maps each unique term $t$ in vocabulary $V$ to a postings list:
$$\text{Postings}(t) = \{ \text{doc\_id}: \{ \text{tf}: f(t, d), \text{positions}: [p_1, p_2, \dots] \} \}$$
Also maintains:
- Document Frequency $df(t)$
- Collection Frequency $cf(t)$
- Document lengths $|d|$ and collection average $L_{avg} = \frac{1}{N}\sum |d|$

---

### 3.3 Vector Space Model (VSM) with TF-IDF
- **Sublinear Term Frequency:**
  $$\text{TF}(t, d) = 1 + \log_{10}(f(t, d)) \quad \text{if } f(t, d) > 0 \text{ else } 0$$
- **Smoothed Inverse Document Frequency:**
  $$\text{IDF}(t) = \log_{10}\left(\frac{N + 1}{df(t) + 1}\right) + 1$$
- **Cosine Similarity:**
  $$\text{Cosine}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_{t \in q} w_{t,q}^2} \cdot \sqrt{\sum_{t \in d} w_{t,d}^2}}$$

---

### 3.4 Okapi BM25 Probabilistic Model
Incorporates term frequency saturation and document length normalization:
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}_{BM25}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{L_{avg}}\right)}$$
where:
- $\text{IDF}_{BM25}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)$
- $k_1 = 1.5$ (term frequency saturation scaling)
- $b = 0.75$ (document length normalization penalty)

---

### 3.5 Classical Boolean Retrieval Model
- Evaluates Boolean queries with operators `AND`, `OR`, `NOT`, and parenthesized sub-expressions using the **Shunting-Yard algorithm** to generate Reverse Polish Notation (RPN).
- Postings intersection ($A \cap B$), union ($A \cup B$), and set difference ($A \setminus B$).

---

### 3.6 Semantic Information Retrieval (Latent Semantic Indexing - LSI)
Applies **Truncated Singular Value Decomposition (SVD)** on the Term-Document TF-IDF matrix:
$$A_{m \times n} \approx U_k \Sigma_k V_k^T \quad (k = 25)$$
- Maps documents to low-dimensional concept space: $D = V_k \Sigma_k$
- Query projection: $\vec{q}_{concept} = \vec{q}^T U_k \Sigma_k^{-1}$
- Cosine similarity computed in the dense semantic subspace, resolving synonymy and term mismatch.

---

### 3.7 Hybrid Retrieval Model
Linearly combines normalized lexical BM25 scores with semantic LSI similarity:
$$\text{Score}_{Hybrid}(d) = \alpha \cdot \text{Score}_{BM25\_norm}(d) + (1 - \alpha) \cdot \text{Score}_{LSI\_norm}(d) \quad (\alpha = 0.65)$$

---

### 3.8 Query Reformulation: Rocchio Relevance Feedback
Enables live interactive training from citizen clicks:
$$\vec{q}_{new} = \alpha \vec{q}_0 + \frac{\beta}{|D_r|} \sum_{d \in D_r} \vec{d} - \frac{\gamma}{|D_{nr}|} \sum_{d \in D_{nr}} \vec{d}$$
with parameters: $\alpha = 1.0, \beta = 0.75, \gamma = 0.25$.

---

## 📊 4. IR Evaluation Benchmark & Empirical Results

The system includes an automated evaluation benchmark over **12 test queries** with annotated binary and graded relevance judgments ($0 = \text{irrelevant}, 1 = \text{partially relevant}, 2 = \text{highly relevant}$).

### Comparative Evaluation Matrix

| Model | P@1 | P@5 | P@10 | Recall@5 | F1@5 | MAP (Mean AP) | NDCG@5 | NDCG@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Search (BM25 + LSI)** 🏆 | **0.917** | **0.617** | **0.425** | **0.589** | **0.603** | **0.7016** | **0.7981** | **0.8150** | **0.9583** |
| **Okapi BM25** | 0.917 | **0.617** | 0.408 | 0.589 | 0.603 | 0.6837 | 0.7892 | 0.8054 | **0.9583** |
| **Latent Semantic Indexing (LSI)** | 0.833 | **0.617** | 0.400 | 0.582 | 0.599 | 0.6532 | 0.7410 | 0.7596 | 0.9167 |
| **Vector Space Model (VSM)** | 0.833 | 0.600 | 0.392 | 0.574 | 0.587 | 0.6476 | 0.7650 | 0.7841 | 0.9167 |
| **Boolean Retrieval** | 0.083 | 0.050 | 0.033 | 0.052 | 0.051 | 0.0472 | 0.0650 | 0.0750 | 0.1667 |

> [!NOTE]
> **Hybrid Search** delivers the highest **MAP (0.7016)** and **NDCG@10 (0.8150)** by balancing BM25's exact keyword precision with LSI's semantic generalization.

---

## 🚀 5. Installation & Quick Start

### 6.1 Prerequisites
- Python 3.10 or higher
- Web browser (Chrome, Edge, Firefox)

### 6.2 Running the Application
From the project directory:

```powershell
python run.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 📜 6. Project Authors
- **Project Title:** CIVICFIND — An Intelligent Information Retrieval System for Government Schemes and Public Services
- **Domain:** Natural Language Processing & Information Retrieval (IR)
- **Frameworks:** Python, Flask, NLTK, scikit-learn, Chart.js, Tailwind CSS
