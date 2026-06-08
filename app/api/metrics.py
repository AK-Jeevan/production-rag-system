import logging
from fastapi import APIRouter, status
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring.metrics import (
    REQUEST_COUNT,
    ERROR_COUNT,
    QUERY_LATENCY,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Prometheus Metrics",
    description="Exposes Prometheus metrics for scraping by a Prometheus server.",
    include_in_schema=False,
)

async def metrics() -> Response:
    logger.info("📊 Prometheus metrics scraped.")
    try:
        return Response(
            content    = generate_latest(),
            media_type = CONTENT_TYPE_LATEST,
            status_code= status.HTTP_200_OK,
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate metrics: {e}")
        return Response(
            content    = b"Failed to generate metrics.",
            media_type = "text/plain",
            status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
        )