import os
import logging
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from datetime import datetime, timezone
from src.monitoring.metrics import DOCUMENT_UPLOAD_COUNT

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "data/raw/uploads"
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


class UploadResponse(BaseModel):
    message: str
    filename: str
    saved_as: str
    file_type: str
    size_kb: float
    uploaded_at: str


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a Document",
    description="Upload a document (PDF, TXT, DOCX, MD) to the RAG ingestion pipeline.",
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    logger.info(f"📥 Upload requested: {file.filename!r}")

    # Validate extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB.",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    # Save with unique filename to avoid overwrites
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(filepath, "wb") as f:
            f.write(contents)
        logger.info(f"✅ File saved: {filepath}")

        # Track upload count by file type
        DOCUMENT_UPLOAD_COUNT.labels(file_type=ext).inc()

    except Exception as e:
        logger.error(f"❌ Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the uploaded file.",
        )

    return UploadResponse(
        message=f"{file.filename} uploaded successfully.",
        filename=file.filename,
        saved_as=unique_filename,
        file_type=ext,
        size_kb=round(len(contents) / 1024, 2),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
