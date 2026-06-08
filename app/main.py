import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from app.api.routes import router as rag_router
from app.api.upload import router as upload_router
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from src.monitoring import metrics as _metrics  # noqa: F401 — registers all Prometheus metrics on import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RAG API server...")
    yield
    logger.info("🛑 Shutting down RAG API server...")


app = FastAPI(
    title="Production RAG System",
    description="A production-grade RAG API powered by Gemini + FAISS + Reranker + LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router,   prefix="/api/v1", tags=["Health"])
app.include_router(rag_router,      prefix="/api/v1", tags=["RAG"])
app.include_router(upload_router,   prefix="/api/v1", tags=["Upload"])
app.include_router(feedback_router, prefix="/api/v1", tags=["Feedback"])
app.include_router(metrics_router,  prefix="/api/v1", tags=["Monitoring"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Root endpoint - redirects to API documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Production RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            p { color: #666; line-height: 1.6; }
            .links { margin-top: 20px; }
            a { display: inline-block; margin-right: 15px; padding: 10px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; }
            a:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Production RAG System</h1>
            <p>A production-grade RAG API powered by Gemini + FAISS + Reranker + LangChain</p>
            <div class="links">
                <a href="/docs">📖 API Documentation (Swagger)</a>
                <a href="/redoc">📚 ReDoc</a>
                <a href="/openapi.json">🔗 OpenAPI Schema</a>
            </div>
        </div>
    </body>
    </html>
    """