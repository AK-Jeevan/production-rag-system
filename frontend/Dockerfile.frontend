FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/frontend

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /frontend

# ── Dependencies ──────────────────────────────────────────────────────────────
# Build context is ./frontend/, so COPY paths are relative to that directory
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────────────
COPY . .

# ── Non-Root User ─────────────────────────────────────────────────────────────
RUN addgroup --system appgroup \
 && adduser  --system --ingroup appgroup appuser \
 && chown -R appuser:appgroup /frontend

USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=5).read()" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "gradio_app.py"]
