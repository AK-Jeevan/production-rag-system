import json
import logging
import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from datetime import datetime, timezone
from app.schemas.feedback import FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter()

FEEDBACK_FILE = "data/feedback/feedback.json"


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str
    received_at: str


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Feedback",
    description="Submit a rating and optional comment for a RAG query response.",
)
async def save_feedback(
    feedback: FeedbackRequest,
) -> FeedbackResponse:
    feedback_id = uuid.uuid4().hex
    received_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"📥 Feedback received: id={feedback_id!r}, rating={feedback.rating}")

    feedback_entry = {
        "feedback_id": feedback_id,
        "received_at": received_at,
        **feedback.model_dump(),
    }

    # Ensure directory exists
    import os

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

    try:
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        logger.info(f"✅ Feedback saved: {feedback_id!r}")

    except Exception as e:
        logger.error(f"❌ Failed to save feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback. Please try again.",
        )

    return FeedbackResponse(
        message="Feedback saved successfully.",
        feedback_id=feedback_id,
        received_at=received_at,
    )
