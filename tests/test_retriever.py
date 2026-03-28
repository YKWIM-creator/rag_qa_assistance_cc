import pytest
import tempfile
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import build_vectorstore, get_retriever


def test_build_vectorstore_returns_chroma_instance():
    chunks = [
        {"text": "Baruch offers a BBA program.", "metadata": {"url": "http://baruch.cuny.edu", "school": "baruch", "title": "Programs", "chunk_index": 0, "scraped_at": "2026-01-01"}},
        {"text": "Hunter has a nursing program.", "metadata": {"url": "http://hunter.cuny.edu", "school": "hunter", "title": "Programs", "chunk_index": 0, "scraped_at": "2026-01-01"}},
    ]
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
    mock_embeddings.embed_query = MagicMock(return_value=[0.15] * 384)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.retrieval.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = tmpdir
            mock_settings.chroma_collection_name = "test_collection"
            vs = build_vectorstore(chunks, mock_embeddings)

    assert vs is not None


def test_get_retriever_returns_langchain_retriever():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever = MagicMock(return_value=MagicMock())
    retriever = get_retriever(mock_vectorstore, k=3)
    mock_vectorstore.as_retriever.assert_called_once()
    assert retriever is not None
