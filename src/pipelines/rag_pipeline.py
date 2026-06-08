import time
import logging
from typing import Generator
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import GeminiGenerator
from src.utils.mlflow_tracker import MLflowTracker
from src.retrieval.query_rewriter import QueryRewriter
from src.memory.chat_memory import ChatMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(
        self,
        retrieval_top_k: int   = 20,
        rerank_top_k   : int   = 5,
        model_key      : str   = "flash25",
        temperature    : float = 0.7
    ):
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k    = rerank_top_k

        logger.info("🚀 Initializing RAG Pipeline...")

        self.retriever = Retriever(top_k=retrieval_top_k)
        self.reranker  = Reranker(top_k=rerank_top_k)
        self.generator = GeminiGenerator(
            model_key   = model_key,
            temperature = temperature
        )
        self.rewriter  = QueryRewriter()
        self.memory    = ChatMemory(max_history=20)
        self.tracker   = MLflowTracker()

        logger.info("✅ RAG Pipeline ready.")

    def ask(self, question: str) -> dict:
        """Run the full RAG pipeline for a question (non-streaming)."""
        if not question or not isinstance(question, str):
            raise ValueError("❌ Question must be a non-empty string.")

        logger.info(f"❓ Question: {question}")

        with self.tracker.start_run("rag_query"):

            # --- Step 0: Rewrite query using memory ---
            logger.info("✏️  Step 0: Rewriting query...")
            history_text    = self.memory.get_history_as_text()
            rewritten_query = self.rewriter.rewrite(
                question = question,
                history  = history_text
            )
            logger.info(f"   Original : {question}")
            logger.info(f"   Rewritten: {rewritten_query}")

            # --- Step 1: Retrieve Top 20 ---
            logger.info(f"🔍 Step 1: Retrieving top {self.retrieval_top_k} chunks...")
            retrieval_start   = time.time()
            docs              = self.retriever.retrieve(
                rewritten_query,
                top_k = self.retrieval_top_k
            )
            retrieval_latency = round(time.time() - retrieval_start, 4)

            if not docs:
                logger.warning("⚠️  No documents retrieved.")

            # --- Step 2: Rerank to Top 5 ---
            logger.info(f"🔀 Step 2: Reranking to top {self.rerank_top_k} chunks...")
            rerank_start   = time.time()
            reranked_docs  = self.reranker.rerank(
                rewritten_query,
                docs,
                top_k = self.rerank_top_k
            )
            rerank_latency = round(time.time() - rerank_start, 4)

            # --- Step 3: Build prompt ---
            logger.info("📝 Step 3: Building prompt...")
            prompt       = PromptBuilder.build_prompt(
                query          = rewritten_query,
                retrieved_docs = reranked_docs
            )
            prompt_stats = PromptBuilder.get_prompt_stats(prompt)

            # --- Step 4: Generate ---
            logger.info("🤖 Step 4: Generating answer...")
            generation_start   = time.time()
            answer             = self.generator.generate_answer(prompt)
            generation_latency = round(time.time() - generation_start, 4)

            total_latency = round(
                retrieval_latency + rerank_latency + generation_latency, 4
            )

            # --- Step 5: Extract sources ---
            sources = list({
                doc.metadata.get("source", "unknown")
                for doc in reranked_docs
            })

            # --- Step 6: Update memory ---
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(answer)

            # --- Step 7: Calculate tokens + cost ---
            input_tokens  = prompt_stats["approx_tokens"]
            output_tokens = len(answer) // 4
            total_tokens  = input_tokens + output_tokens

            estimated_cost = round(
                (input_tokens  / 1_000_000) * 0.075 +
                (output_tokens / 1_000_000) * 0.30,
                6
            )

            # --- Step 8: Track metrics ---
            logger.info("📊 Step 8: Logging metrics to MLflow...")
            self.tracker.log_rag_parameters(
                chunk_size      = 1000,
                chunk_overlap   = 200,
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
                top_k           = self.rerank_top_k,
                vector_db       = "FAISS",
                llm_model       = self.generator.get_model_name()
            )
            self.tracker.log_latency_metrics(
                retrieval_latency  = retrieval_latency,
                generation_latency = generation_latency,
                total_latency      = total_latency
            )
            self.tracker.log_metric("docs_retrieved",  len(docs))
            self.tracker.log_metric("docs_reranked",   len(reranked_docs))
            self.tracker.log_metric("rerank_latency",  rerank_latency)
            self.tracker.log_metric("prompt_tokens",   prompt_stats["approx_tokens"])
            self.tracker.log_metric("memory_messages", self.memory.message_count())
            self.tracker.log_metric("input_tokens",    input_tokens)
            self.tracker.log_metric("output_tokens",   output_tokens)
            self.tracker.log_metric("total_tokens",    total_tokens)
            self.tracker.log_metric("estimated_cost",  estimated_cost)

            # --- Summary ---
            logger.info("✅ RAG Pipeline complete.")
            logger.info(f"   Original   : {question}")
            logger.info(f"   Rewritten  : {rewritten_query}")
            logger.info(f"   Retrieved  : {len(docs)} chunks")
            logger.info(f"   Reranked   : {len(reranked_docs)} chunks")
            logger.info(f"   Retrieval  : {retrieval_latency}s")
            logger.info(f"   Rerank     : {rerank_latency}s")
            logger.info(f"   Generation : {generation_latency}s")
            logger.info(f"   Total      : {total_latency}s")
            logger.info(f"   Tokens     : {total_tokens}")
            logger.info(f"   Cost       : ${estimated_cost}")
            logger.info(f"   Sources    : {sources}")
            logger.info(f"   Memory     : {self.memory.message_count()} messages")

            return {
                "question"          : question,
                "rewritten_query"   : rewritten_query,
                "answer"            : answer,
                "sources"           : sources,
                "retrieved_docs"    : reranked_docs,
                "retrieval_latency" : retrieval_latency,
                "rerank_latency"    : rerank_latency,
                "generation_latency": generation_latency,
                "total_latency"     : total_latency,
                "prompt_tokens"     : prompt_stats["approx_tokens"],
                "input_tokens"      : input_tokens,
                "output_tokens"     : output_tokens,
                "total_tokens"      : total_tokens,
                "estimated_cost"    : estimated_cost,
                "memory_messages"   : self.memory.message_count(),
            }

    def ask_stream(self, question: str) -> Generator[str, None, None]:
        """Run a lightweight RAG pipeline and stream the answer token by token."""
        if not question or not isinstance(question, str):
            raise ValueError("❌ Question must be a non-empty string.")

        logger.info(f"🌊 Stream query: {question!r}")

        # Step 0: Rewrite query using memory
        history_text    = self.memory.get_history_as_text()
        rewritten_query = self.rewriter.rewrite(
            question = question,
            history  = history_text
        )
        logger.info(f"   Rewritten: {rewritten_query}")

        # Step 1: Retrieve
        docs = self.retriever.retrieve(
            rewritten_query,
            top_k = self.retrieval_top_k
        )
        if not docs:
            logger.warning("⚠️  No documents retrieved for stream query.")

        # Step 2: Rerank
        reranked_docs = self.reranker.rerank(
            rewritten_query,
            docs,
            top_k = self.rerank_top_k
        )

        # Step 3: Build prompt
        prompt = PromptBuilder.build_prompt(
            query          = rewritten_query,
            retrieved_docs = reranked_docs
        )

        # Step 4: Stream answer
        logger.info("🤖 Streaming answer...")
        full_answer = []
        for chunk in self.generator.generate_answer_stream(prompt):
            full_answer.append(chunk)
            yield chunk

        # Step 5: Update memory after stream completes
        self.memory.add_user_message(question)
        self.memory.add_assistant_message("".join(full_answer))
        logger.info("✅ Stream complete. Memory updated.")

    def clear_memory(self) -> None:
        """Reset conversation memory."""
        self.memory.clear()
        logger.info("🗑️  Memory cleared.")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = RAGPipeline(
        retrieval_top_k = 20,
        rerank_top_k    = 5,
        model_key       = "flash25",
        temperature     = 0.7
    )

    questions = [
        "What is FastAPI?",
        "How does it handle authentication?",
        "What is MLflow used for?",
    ]

    for question in questions:
        print(f"\n{'='*80}")
        result = pipeline.ask(question)
        print(f"❓ Question       : {result['question']}")
        print(f"✏️  Rewritten      : {result['rewritten_query']}")
        print(f"💬 Answer         : {result['answer'][:300]}...")
        print(f"📚 Sources        : {result['sources']}")
        print(f"⏱️  Total Latency  : {result['total_latency']}s")
        print(f"🔢 Total Tokens   : {result['total_tokens']}")
        print(f"💰 Estimated Cost : ${result['estimated_cost']}")
        print(f"🧠 Memory         : {result['memory_messages']} messages")
        print(f"{'='*80}")

    # Streaming test
    print(f"\n{'='*80}")
    print("🌊 Streaming Test:")
    print("Streaming: ", end="", flush=True)
    for chunk in pipeline.ask_stream("What is FastAPI?"):
        print(chunk, end="", flush=True)
    print(f"\n{'='*80}")