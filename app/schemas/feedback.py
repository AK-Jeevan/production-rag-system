from pydantic import BaseModel, Field
from typing import Optional

class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question being rated")
    answer: str = Field(..., min_length=1, description="The answer provided")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional feedback comment")