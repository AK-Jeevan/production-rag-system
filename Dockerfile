# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment Variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Project ──────────────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p data/raw/uploads \
             data/feedback

# ── Non-root User (Security) ──────────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Expose Port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start Server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]