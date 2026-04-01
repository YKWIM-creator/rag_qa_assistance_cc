import logging
from src.models import ScrapedPage
from src.ingestion.chunker import chunk_page
from src.retrieval.retriever import build_vectorstore

logger = logging.getLogger(__name__)


async def ingest_pages(pages: list[ScrapedPage], embedding_model):
    """Chunk all scraped pages and store in ChromaDB."""
    all_chunks = []
    for page in pages:
        chunks = chunk_page(page)
        all_chunks.extend(chunks)
        logger.info(f"Chunked {page.url}: {len(chunks)} chunks")

    logger.info(f"Total chunks to index: {len(all_chunks)}")
    vectorstore = build_vectorstore(all_chunks, embedding_model)
    return vectorstore
