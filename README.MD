### Adaptive Retrieval Router — Evaluation Notebook

This notebook documents the analysis of a supervised ML–based retrieval routing strategy for StackOverflow question answering.

The router selects the optimal retrieval strategy per query:

BM25 – lexical matching

Dense – semantic MiniLM embeddings + FAISS

Hybrid – combined lexical + dense

The goal is to see if routing strategy improves MRR / nDCG while reducing retrieval latency.

##### Project Objective

The router model predicts one of:
```
bm25 | dense | hybrid
```
based on 2 groups of features:

Pre-retrieval features (pre retrieval):

- Query length, ratios length
- code ratio
- stopword ratio
- Digit / punctuation ratios

Post-retrieval preview features (cheap but requires BM25/dense top-k=2):

- BM25 top-k preview scores

- Dense top-k cosine similarities


#### ML models

Two models were trained using the ML system template (split → preprocess → tune → evaluate -> register):

- Logistic Regression 

- XGBoost

Both models are tuned using stratified 5-fold cross validation. 

Logistic regression is selected based on the validation performance
full config here: https://github.com/ai-portfolio-zmin/lm-system-template/blob/main/config/route.yml


#### Notebook Structure

This notebook performs:

1. Load evaluation data:
   - Test queries

   - Ground-truth relevance dictionary

2. For each query:

   - Run BM25, Dense, and Hybrid retrieval

     - Call vector-search service endpoints

     - Record retrieved doc IDs and latency

     - Convert retrieved docs into a relevance array, e.g. [1, 0, 0, 0, 0]

   - Compute Oracle:

     - Select method that ranks the ground-truth answer highest

     - Break ties by latency

   - Run Router:

     - Call router endpoint to get predicted method

3. Evaluation

   - Compute MRR and nDCG for:

     - BM25

     - Dense

     - Hybrid

     - Router

     - Oracle

#### Summary of Findings
Retrival quality
```
Hybrid > Router > Dense > BM25
```
```
Mode	MRR	nDCG
BM25	0.56	0.53
Dense	0.63	0.65
Hybrid	0.72	0.77
Router	0.68	0.71
Oracle	0.78	0.83
```
The router consistently performs better than dense and bm25, and approaches hybrid performance.

Latency

Router reduces latency compared to hybrid:
```
Mode	Latency (s)
BM25	0.007
Dense	0.040
Hybrid	0.047
Router	0.038
Oracle	0.034
```


Interpretation:

- Hybrid is the most accurate but also the slowest.

- Router provides a good balance:

   - Slight quality reduction compared to Hybrid

   - latency improvement (~20%)

#### Dependencies

This notebook is an evaluation for an adaptive retrieval router.

Training and serving are implemented in separate services:

- **Vector Search Service** (BM25, dense, hybrid)
  - Repo: https://github.com/ai-portfolio-zmin/vector-search-service
  - Provides `/preview` endpoints

- **Router Service**
  - Trained using my ML system template:
    - Repo: https://github.com/ai-portfolio-zmin/ml-system-template
  - Provides a `/batch_route` endpoint that predicts: `bm25 | dense | hybrid`

This notebook assumes those services are running locally and reachable over HTTP.

It does **not** re-train models; it focuses on evaluating routing quality and latency.

Note:
A LoRA-fine-tuned LLM router is in a separate repo:
https://github.com/ai-portfolio-zmin/llm-fine-tune