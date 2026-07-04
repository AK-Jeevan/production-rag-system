import logging
from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
    pipeline_ready: bool


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the current health status of the RAG service, including pipeline readiness.",
)
async def health_check() -> HealthResponse:
    from app.api.routes import rag_service

    pipeline_ready = rag_service.pipeline is not None
    status_str = "healthy" if pipeline_ready else "starting"

    if pipeline_ready:
        logger.info("💚 Health check requested — pipeline ready.")
    else:
        logger.warning("⚠️ Health check requested — pipeline NOT yet initialized.")

    return HealthResponse(
        status=status_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="RAG Assistant",
        version="1.0.0",
        pipeline_ready=pipeline_ready,
    )
