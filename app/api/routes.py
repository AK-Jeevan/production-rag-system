import time
import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from src.monitoring.metrics import (
    REQUEST_COUNT,
    ERROR_COUNT,
    QUERY_LATENCY,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Eagerly create the RAGService instance at module load time.
# The actual pipeline is still initialized lazily via get_pipeline(),
# which main.py's lifespan will call during startup to avoid cold-start.
rag_service = RAGService()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG pipeline",
    description="Submit a question and retrieve a full AI-generated answer with sources.",
)
async def query_rag(request: QueryRequest) -> QueryResponse:
    logger.info(f"📥 Received query: {request.question!r}")
    REQUEST_COUNT.labels(endpoint="/query").inc()

    try:
        start_time = time.time()
        result = rag_service.query(request.question, top_k=request.top_k)
        QUERY_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence", None),
        )

    except FileNotFoundError as e:
        logger.error(f"❌ FAISS index not found: {e}")
        ERROR_COUNT.labels(endpoint="/query", error_type="FileNotFoundError").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized. Please upload documents first.",
        )

    except ValueError as e:
        logger.warning(f"⚠️ Validation error: {e}")
        ERROR_COUNT.labels(endpoint="/query", error_type="ValueError").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        ERROR_COUNT.labels(endpoint="/query", error_type="Exception").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again.",
        )


@router.post(
    "/query-stream",
    status_code=status.HTTP_200_OK,
    summary="Stream query the RAG pipeline",
    description="Submit a question and receive a token-by-token streamed response via SSE.",
)
async def query_rag_stream(request: QueryRequest) -> StreamingResponse:
    logger.info(f"🌊 Received stream query: {request.question!r}")
    REQUEST_COUNT.labels(endpoint="/query-stream").inc()

    if not request.question or not request.question.strip():
        ERROR_COUNT.labels(endpoint="/query-stream", error_type="ValueError").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty.",
        )

    try:
        start_time = time.time()
        pipeline = rag_service.get_pipeline()
        response = StreamingResponse(
            pipeline.ask_stream(request.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        QUERY_LATENCY.labels(endpoint="/query-stream").observe(time.time() - start_time)
        return response

    except FileNotFoundError as e:
        logger.error(f"❌ FAISS index not found: {e}")
        ERROR_COUNT.labels(
            endpoint="/query-stream", error_type="FileNotFoundError"
        ).inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized. Please upload documents first.",
        )

    except ValueError as e:
        logger.warning(f"⚠️ Validation error: {e}")
        ERROR_COUNT.labels(endpoint="/query-stream", error_type="ValueError").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    except Exception as e:
        logger.error(f"❌ Stream query failed: {e}")
        ERROR_COUNT.labels(endpoint="/query-stream", error_type="Exception").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming failed. Please try again.",
        )
