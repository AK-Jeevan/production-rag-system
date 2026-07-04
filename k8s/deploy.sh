#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-production-rag-system}"
KUSTOMIZE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
fail()  { echo "[ERROR] $*"; exit 1; }

command -v kubectl >/dev/null 2>&1 || fail "kubectl is required but not found in PATH"

info "Target namespace : ${NAMESPACE}"
info "Kustomize dir   : ${KUSTOMIZE_DIR}"

info "Ensuring namespace '${NAMESPACE}' exists..."
kubectl apply -f "${KUSTOMIZE_DIR}/namespace.yaml"

info "Applying all manifests via Kustomize..."
kubectl apply -k "${KUSTOMIZE_DIR}"

info "Waiting for deployments..."
for deployment in fastapi frontend grafana mlflow prometheus; do
  info "  -> ${deployment}"
  kubectl rollout status deployment/${deployment} -n "${NAMESPACE}" --timeout=600s || true
done

info "Pod status in '${NAMESPACE}':"
kubectl get pods -n "${NAMESPACE}" -o wide

info "All resources in '${NAMESPACE}':"
kubectl get all -n "${NAMESPACE}"

info "Deployment to namespace '${NAMESPACE}' complete."
