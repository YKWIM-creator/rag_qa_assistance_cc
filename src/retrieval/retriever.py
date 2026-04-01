import logging
from langchain_community.vectorstores import Chroma
from config.settings import settings

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: list[dict], embedding_model) -> Chroma:
    """Build (or load) a ChromaDB vector store from chunks."""
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{c['metadata']['url']}__chunk_{c['metadata']['chunk_index']}" for c in chunks]

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
    )

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)
        logger.info(f"Indexed batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(vectorstore: Chroma, k: int = 5, metadata_filter: dict = None):
    """Return a LangChain MMR retriever from a ChromaDB vector store."""
    search_kwargs = {"k": k, "fetch_k": k * 5}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
