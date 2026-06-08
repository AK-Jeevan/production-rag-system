import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean a single text string."""
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,;:!?'\"-]", " ", text)

        # Collapse multiple newlines into one
        text = re.sub(r"\n+", "\n", text)

        # Collapse multiple spaces/tabs into one
        text = re.sub(r"[ \t]+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def is_valid_document(doc, min_length: int = 20) -> bool:
        """Return False if document is too short or empty after cleaning."""
        return bool(doc.page_content and len(doc.page_content) >= min_length)

    def clean_documents(self, documents: list, min_length: int = 20) -> list:
        """Clean all documents and filter out empty/short ones."""
        if not documents:
            logger.warning("⚠️  No documents provided to clean.")
            return []

        cleaned = []
        skipped = 0

        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

            if self.is_valid_document(doc, min_length=min_length):
                cleaned.append(doc)
            else:
                skipped += 1
                logger.warning(
                    f"⚠️  Skipped short/empty doc from: "
                    f"{doc.metadata.get('source', 'unknown')}"
                )

        logger.info(f"✅ Cleaned  : {len(cleaned)} documents")
        logger.info(f"🗑️  Skipped  : {skipped} documents (too short or empty)")
        return cleaned


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    sample_docs = [
        Document(page_content="  Hello   world!\n\nThis is   a test.  ", metadata={"source": "test.txt"}),
        Document(page_content="Visit https://example.com for more info.", metadata={"source": "web.md"}),
        Document(page_content="Contact us at hello@example.com today.", metadata={"source": "email.txt"}),
        Document(page_content="   ", metadata={"source": "empty.txt"}),           # should be skipped
        Document(page_content="Hi", metadata={"source": "short.txt"}),            # should be skipped
    ]

    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(sample_docs)

    print(f"\nCleaned {len(cleaned_docs)} documents.\n")
    for doc in cleaned_docs:
        print(f"Source : {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content}")
        print("-" * 40)