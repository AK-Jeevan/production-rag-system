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


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the current health status of the RAG service.",
)
async def health_check() -> HealthResponse:
    logger.info("💚 Health check requested.")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="RAG Assistant",
        version="1.0.0",
    )
