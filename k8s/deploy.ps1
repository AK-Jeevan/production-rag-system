param([string]$Namespace = "production-rag-system")
$ErrorActionPreference = "Stop"
$K8sDir = $PSScriptRoot
Write-Host "[INFO] Namespace: $Namespace"
Write-Host "[INFO] Ensuring namespace exists..."
kubectl apply -f "$K8sDir/namespace.yaml"
Write-Host "[INFO] Applying all manifests via Kustomize..."
kubectl apply -k "$K8sDir"
# Waiting for deployments
@("fastapi","frontend","grafana","mlflow","prometheus") | ForEach-Object {
    $dep = $_
    Write-Host "  -> $dep"
    kubectl rollout status deployment/$dep -n $Namespace --timeout=600s
}

kubectl get pods -n $Namespace -o wide
Write-Host "[INFO] Pod status:"
Write-Host "[INFO] All resources:"
kubectl get all -n $Namespace
Write-Host "[INFO] Deployment complete."
