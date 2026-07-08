# Plan: Eliminate FAISS/BM25 "Index Not Found" Errors

## Current State
- FastAPI pod crashes at startup because `RAGPipeline.__init__` → `Retriever` → `HybridRetriever` eagerly calls `vector_store_manager.load_vector_store()` (`src/retrieval/hybrid_retriever.py:34`).
- No FAISS index (`data/models/faiss_index`) or BM25 index (`data/models/bm25_index`) exists on disk.
- `ingest_pipeline.py` only chunks documents and saves `data/processed/chunks.json`; it does **not** build the FAISS or BM25 indices.
- The existing `k8s/fastapi/ingest-job.yaml` is broken (malformed `volumes` block) and only processes `data/raw/uploads`, missing the bulk docs in `data/raw/aws_docs`, `git_docs`, etc.

## Goal
1. Build the FAISS + BM25 indices **now** from the existing `data/raw` documents.
2. Ensure the FastAPI pod **never** starts without a valid index again.

## Implementation Steps

### Step 1: Create a deterministic index-build entrypoint
Create `src/scripts/build_index.py` (or `run_build_index.py` at repo root) that:
- Loads `data/raw` via `DocumentLoader`
- Cleans via `TextCleaner`
- Chunks via `DocumentChunker`
- Builds FAISS index via `VectorStoreManager.create_vector_store()` + `save_vector_store()`
- Builds BM25 index via `BM25Retriever.build()` + `save()`
- Saves both to `data/models/`

This is the single source of truth for index creation. `vector_store.py` and `bm25_retriever.py` already have the building blocks.

### Step 2: Build the index locally (one-time)
Run the new script from the repo root:
```bash
python -m src.scripts.build_index
# or
python run_build_index.py
```

Expected output: `data/models/faiss_index/` and `data/models/bm25_index/` directories populated.

### Step 3: Fix the Kubernetes ingestion Job
Replace `k8s/fastapi/ingest-job.yaml` with a corrected version that:
- Uses the same image as FastAPI (`wodlldd/production-rag-system:latest`)
- Mounts the `rag-data` PVC at `/app/data`
- Runs the index-build command for **all** of `data/raw` (not just `uploads`)
- Has a proper `volumes` section
- Runs as a **Kubernetes Job** with `restartPolicy: Never`

### Step 4: Make FastAPI deployment resilient
In `k8s/fastapi/deployment.yaml`:
- Keep `DISABLE_EAGER_INIT=true` (already applied via kubectl set env on revision `fastapi-5c87dd564b`). This prevents the pod from crashing if the index is ever missing.
- Improve the `/api/v1/health` endpoint to return `200` when the app is up but `pipeline_ready=false` (it already does this — confirmed by logs showing `"pipeline_ready":false` with HTTP 200).
- Optionally add an **init container** or **startup probe** that waits for `data/models/faiss_index` to exist before the main container starts. This is a stronger guarantee than `DISABLE_EAGER_INIT`.

### Step 5: Wire Job → Deployment ordering
Update `k8s/fastapi/deployment.yaml` so the FastAPI pod waits for the index Job:
- The simplest approach: add an **init container** to the FastAPI deployment that runs a lightweight check (`test -d /app/data/models/faiss_index`) in a loop until the directory exists.
- The index Job and FastAPI Deployment both mount the same PVC (`rag-data-pvc`), so once the Job finishes, the index is visible to the pod.

### Step 6: Validate
1. Delete the existing index (or use a fresh PVC) to simulate a clean cluster.
2. Deploy the index Job: `kubectl apply -f k8s/fastapi/index-job.yaml`
3. Watch it complete: `kubectl wait --for=condition=complete job/fastapi-index-build -n production-rag-system --timeout=600s`
4. Deploy FastAPI: `kubectl apply -f k8s/fastapi/deployment.yaml`
5. Verify pod is `Running/Ready` and health returns `200` with `pipeline_ready:true`.
6. Port-forward frontend and send a query; confirm it succeeds.

## Open Question
Do you want the index build to happen **automatically on every deploy** (via the Kubernetes Job), or only **manually when documents change**?

- **Recommended**: Automatic Job on deploy ensures the index is always fresh. It adds ~2–5 minutes to deployment time for the first run, but subsequent runs can be skipped if the index already exists (add a pre-check in the Job command).
