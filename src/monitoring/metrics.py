from prometheus_client import Counter, Histogram, Info

# ── Request Metrics ───────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG query requests received",
    ["endpoint"]                  # /query or /query-stream
)

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total number of errors across the RAG pipeline",
    ["endpoint", "error_type"]    # endpoint + ValueError / Exception
)

# ── Document Metrics ──────────────────────────────────────────────────────────
DOCUMENT_UPLOAD_COUNT = Counter(
    "documents_uploaded_total",
    "Total number of documents successfully uploaded",
    ["file_type"]                 # .pdf / .txt / .docx / .md
)

# ── Latency Metrics ───────────────────────────────────────────────────────────
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency in seconds",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Latency of the retrieval step in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)

RERANK_LATENCY = Histogram(
    "rerank_latency_seconds",
    "Latency of the reranking step in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
)

GENERATION_LATENCY = Histogram(
    "generation_latency_seconds",
    "Latency of the LLM generation step in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

# ── Token & Cost Metrics ──────────────────────────────────────────────────────
INPUT_TOKENS = Counter(
    "input_tokens_total",
    "Total number of input tokens sent to the LLM",
    ["model"]                     # flash / flash25 / pro
)

OUTPUT_TOKENS = Counter(
    "output_tokens_total",
    "Total number of output tokens received from the LLM",
    ["model"]
)

ESTIMATED_COST = Counter(
    "estimated_cost_usd_total",
    "Total estimated cost in USD across all queries",
    ["model"]
)

# ── Pipeline Info ─────────────────────────────────────────────────────────────
PIPELINE_INFO = Info(
    "rag_pipeline",
    "Static metadata about the running RAG pipeline"
)

PIPELINE_INFO.info({
    "embedding_model" : "sentence-transformers/all-MiniLM-L6-v2",
    "vector_db"       : "FAISS",
    "reranker"        : "enabled",
    "memory"          : "enabled",
})