# Backend Connection Failure — Diagnosis Plan

## Current State
Frontend Gradio app returns "backend not connected." Two separate configuration bugs and one likely runtime mismatch were found.

## Evidence Gathered
- `frontend/gradio_app.py:15` reads `BACKEND_URL` (default `http://127.0.0.1:8000/api/v1/query`).
- `docker-compose.yml:136` sets `FASTAPI_URL=http://fastapi:8000/api/v1` — wrong env-var name.
- `docker-compose.yml:129` references `dockerfile: Dockerfile.frontend` but only `frontend/Dockerfile` exists.
- `k8s/frontend/deployment.yaml:46` correctly sets `BACKEND_URL=http://fastapi-service:80/api/v1/query`.

## Identified Issues
1. **docker-compose env mismatch**: `FASTAPI_URL` is never read by the app, so in Compose the frontend falls back to `127.0.0.1:8000`, which resolves to itself rather than the FastAPI container.
2. **Missing Dockerfile.frontend reference**: Compose build will fail looking for a root-level `Dockerfile.frontend` instead of `frontend/Dockerfile`.
3. **Potential runtime mismatch**: If you are running the frontend locally (not in Compose/K8s), `BACKEND_URL` stays at `localhost:8000`. The previous `kubectl port-forward` command also used the wrong service name and namespace.

## Remediation Steps (for implementation agent)
1. In `docker-compose.yml`, change `FASTAPI_URL` to `BACKEND_URL` and set it to `http://fastapi:8000/api/v1/query`.
2. In `docker-compose.yml`, change `dockerfile: Dockerfile.frontend` to `dockerfile: frontend/Dockerfile`.
3. Ensure the FastAPI service is reachable from wherever the frontend runs:
   -Local development: run backend on `127.0.0.1:8000` and frontend with `BACKEND_URL=http://127.0.0.1:8000/api/v1/query`.
   - Docker Compose: use the service name `fastapi` on port `8000`.
   - Kubernetes: the existing `k8s/frontend/deployment.yaml` value `http://fastapi-service:80/api/v1/query` is correct.

## Validation
- `docker-compose config` should show valid frontend env and build context.
- Frontend logs should show backend URL it is actually using.
- Health endpoint `http://<backend>/api/v1/health` must return 200 before frontend probes it.

## Open Question
**How are you currently running the frontend?** (Local `python gradio_app.py`, `docker-compose up`, or Kubernetes pod?)
- **Recommended answer**: Identify the runtime so the correct `BACKEND_URL` value and connectivity path are chosen.
