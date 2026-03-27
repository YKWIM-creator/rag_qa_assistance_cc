import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.embedder import get_embedding_model


def test_get_embedding_model_openai():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.openai_api_key = "sk-test"
        model = get_embedding_model()
    assert model is not None
    assert "OpenAI" in type(model).__name__


def test_get_embedding_model_anthropic():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "anthropic"
        mock_settings.embedding_model = "voyage-3"
        mock_settings.anthropic_api_key = "sk-ant-test"
        model = get_embedding_model()
    assert model is not None


def test_get_embedding_model_unknown_raises():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_model()
