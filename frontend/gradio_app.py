import os
import json
import logging
import requests
import gradio as gr
from requests.exceptions import ConnectionError, Timeout, RequestException, HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL      = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1/query")
REQUEST_TIMEOUT  = int(os.getenv("REQUEST_TIMEOUT", "120"))   # LLM can be slow


# ── Chatbot Function ──────────────────────────────────────────────────────────
def chatbot(message: str, history: list) -> str:
    if not message or not message.strip():
        return "⚠️ Please enter a question."

    payload = {
        "question": message.strip(),
        "top_k"   : 5,
    }

    logger.info(f"📥 Sending query to backend: {message!r}")

    try:
        response = requests.post(
            BACKEND_URL,
            json    = payload,
            timeout = REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data    = response.json()
        answer  = data.get("answer", "").strip()
        sources = data.get("sources", [])

        if not answer:
            return "⚠️ No answer returned from the RAG pipeline."

        # Append sources if available
        if sources:
            sources_text = "\n".join(f"  • {s}" for s in sources)
            answer = f"{answer}\n\n📚 **Sources:**\n{sources_text}"

        logger.info("✅ Answer received successfully.")
        return answer

    except HTTPError as e:
        # Extract detailed error message from backend response body
        detail = "Unknown backend error"
        try:
            detail = e.response.json().get("detail", detail)
        except Exception:
            detail = e.response.text[:200] if e.response.text else detail
        logger.error(f"❌ Backend returned {e.response.status_code}: {detail}")
        return f"❌ Backend error ({e.response.status_code}): {detail}"

    except ConnectionError:
        logger.error("❌ Could not connect to the backend.")
        return "❌ Could not connect to the RAG backend. Please check if the service is running."

    except Timeout:
        logger.error("❌ Backend request timed out.")
        return f"❌ Request timed out after {REQUEST_TIMEOUT}s. The model may be overloaded — please try again."

    except RequestException as e:
        logger.error(f"❌ Request failed: {e}")
        return f"❌ Request failed: {str(e)}"

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return "❌ Something went wrong. Please try again."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn          = chatbot,
    title       = "🤖 Production AI Engineering Knowledge Assistant",
    description = """
### Enterprise RAG System

Powered by **Gemini** · **FastAPI** · **FAISS** · **MLflow** · **Kubernetes** · **Prometheus** · **Grafana**

Ask any question about your ingested documents and get AI-generated answers with source references.
    """,
    examples    = [
        "What is AWS?",
        "What is AWS EC2?",
        "What is AWS Lambda?",
    ],
)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"🚀 Starting Gradio UI — backend: {BACKEND_URL}")
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        show_error  = True,
        theme       = gr.themes.Soft(),
    )
