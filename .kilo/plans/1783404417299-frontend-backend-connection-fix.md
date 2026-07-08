# Plan: Fix Frontend ↔ Backend Connectivity in Kubernetes

## Context
- Frontend (Gradio) is running in a Kubernetes pod in namespace `production-rag-system`.
- The Gradio UI renders but returns "backend not connected" when making requests to `BACKEND_URL`.
- FastAPI deployment in `k8s/fastapi/deployment.yaml` uses `hostNetwork: true`.
- Frontend deployment in `k8s/frontend/deployment.yaml` sets `BACKEND_URL=http://fastapi-service:80/api/v1/query`.
- No ingress or port-forward was verified for the frontend service yet.

## Root Cause Hypotheses
1. **FastAPI pod is unhealthy/crashing** → `fastapi-service` has no endpoints, so in-cluster DNS resolves but connections fail.
2. **`hostNetwork: true` + Service mismatch** → service endpoints may not route correctly to pod port when hostNetwork is enabled.
3. **Frontend access path is wrong** → User may be reaching the UI but the `BACKEND_URL` env var is not being applied correctly.
4. **Ingress path stripping** → If user is accessing via `rag.local` ingress, `/` maps to frontend but frontend app may not work correctly behind path root.

## Implementation Steps

### Step 1: Verify Pod & Service Health
Check the status of all pods and verify service endpoints:
```bash
kubectl get pods -n production-rag-system
kubectl get svc -n production-rag-system
kubectl describe svc fastapi-service -n production-rag-system
kubectl get endpoints fastapi-service -n production-rag-system
```

Expected: FastAPI pod `Running` / `Ready`, frontend pod `Running` / `Ready`, and `fastapi-service` shows at least 1 endpoint.

### Step 2: Diagnose Frontend → Backend Connectivity
Exec into the frontend pod and test the backend URL directly:
```bash
FRONTEND_POD=$(kubectl get pod -n production-rag-system -l app=frontend -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n production-rag-system $FRONTEND_POD -- curl -v http://fastapi-service:80/api/v1/health
```

If this fails, try `fastapi-service/api/v1/query` or test via node IP/pod IP.

### Step 3: Verify Environment Variable in Frontend Pod
```bash
kubectl exec -n production-rag-system $FRONTEND_POD -- env | grep BACKEND_URL
```

Ensure it shows `http://fastapi-service:80/api/v1/query`. If not, the deployment env was not applied correctly.

### Step 4: Remove `hostNetwork: true` (If Misrouted)
If the service has endpoints but connectivity still fails from frontend pod:

1. Edit `k8s/fastapi/deployment.yaml` to remove:
   - `hostNetwork: true`
   - `dnsPolicy: ClusterFirstWithHostNet` (reverts to `ClusterFirst`, which is correct for non-hostNetwork pods)
2. Re-apply deployment: `kubectl apply -f k8s/fastapi/deployment.yaml`
3. Wait for pod rollout: `kubectl rollout status deployment/fastapi -n production-rag-system`
4. Verify endpoints are populated again.

### Step 5: Confirm External Access Path
The frontend is a `ClusterIP` service. To access it externally, the user must port-forward:
```bash
kubectl port-forward svc/frontend-service 7860:80 -n production-rag-system
```
Or use the ingress host `rag.local` if the local hosts file maps it to the cluster IP.

### Step 6: Validate End-to-End
1. Port-forward frontend service locally on `7860`.
2. Open `http://localhost:7860` in browser.
3. Verify Gradio UI loads.
4. Ask a question and confirm backend responds (not "backend not connected").

## Open Question for User
Are you accessing the Gradio UI via `rag.local` in your browser, or via `kubectl port-forward`? This determines whether the ingress path stripping issue (Step 7, if needed) or port-forward is the relevant access method.
