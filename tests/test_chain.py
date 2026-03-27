import pytest
from unittest.mock import patch, MagicMock
from src.generation.providers import get_llm
from src.generation.chain import build_rag_chain, RAGResponse


def test_get_llm_openai():
    with patch("src.generation.providers.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.llm_model = "gpt-4o"
        mock_settings.openai_api_key = "sk-test"
        llm = get_llm()
    assert llm is not None
    assert "OpenAI" in type(llm).__name__


def test_get_llm_unknown_raises():
    with patch("src.generation.providers.settings") as mock_settings:
        mock_settings.llm_provider = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()


def test_build_rag_chain_returns_chain():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    chain = build_rag_chain(mock_retriever, mock_llm)
    assert chain is not None


def test_rag_response_has_required_fields():
    resp = RAGResponse(answer="42", sources=[{"url": "http://x.com", "school": "baruch", "title": "Test"}])
    assert resp.answer == "42"
    assert len(resp.sources) == 1
