from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to query")
    top_k: int = Field(5, ge=1, le=20, description="Number of top results to retrieve")

class QueryResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="The answer to the query")
    sources: list[str] = Field(default_factory=list, description="List of source references")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of the answer")