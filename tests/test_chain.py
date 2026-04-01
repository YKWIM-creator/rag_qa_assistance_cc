import pytest
from unittest.mock import patch, MagicMock
from src.generation.providers import get_llm
from src.models import RAGResponse


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


def test_rag_response_has_required_fields():
    resp = RAGResponse(answer="42", sources=[{"url": "http://x.com", "school": "baruch", "title": "Test"}])
    assert resp.answer == "42"
    assert len(resp.sources) == 1


def test_format_docs_includes_provenance():
    from src.generation.chain import _format_docs
    from unittest.mock import MagicMock
    doc = MagicMock()
    doc.page_content = "Apply by December 1st."
    doc.metadata = {
        "school": "baruch",
        "page_type": "admissions",
        "section_heading": "Application Deadlines",
    }
    result = _format_docs([doc])
    assert "Baruch" in result or "baruch" in result
    assert "admissions" in result
    assert "Application Deadlines" in result
    assert "Apply by December 1st." in result


def test_ask_uses_rewriter(monkeypatch):
    from src.generation.chain import ask
    from unittest.mock import MagicMock, patch, create_autospec
    from src.generation.rewriter import RewrittenQuery
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

    mock_doc = MagicMock()
    mock_doc.page_content = "Baruch admissions info."
    mock_doc.metadata = {"url": "http://baruch.cuny.edu", "school": "baruch",
                         "title": "Admissions", "page_type": "admissions",
                         "section_heading": "Requirements"}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_llm = create_autospec(BaseChatModel, instance=True)
    mock_llm.invoke.return_value = AIMessage(content="You need a 3.0 GPA.")

    with patch("src.generation.chain.rewrite_query") as mock_rw:
        mock_rw.return_value = RewrittenQuery(query="Baruch admissions", school="baruch")
        result = ask("how do i get into baruch", mock_retriever, mock_llm)

    assert result.answer == "You need a 3.0 GPA."
    mock_rw.assert_called_once()


def test_ask_with_school_uses_filtered_retriever():
    from src.generation.chain import ask
    from unittest.mock import MagicMock, patch, create_autospec
    from src.generation.rewriter import RewrittenQuery
    from src.retrieval.retriever import get_retriever
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

    mock_doc = MagicMock()
    mock_doc.page_content = "Baruch info."
    mock_doc.metadata = {"url": "http://b.edu", "school": "baruch",
                         "title": "T", "page_type": "admissions", "section_heading": "S"}

    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.vectorstore = mock_vectorstore

    mock_filtered_retriever = MagicMock()
    mock_filtered_retriever.invoke.return_value = [mock_doc]

    mock_llm = create_autospec(BaseChatModel, instance=True)
    mock_llm.invoke.return_value = AIMessage(content="You need a 3.0 GPA.")

    with patch("src.generation.chain.rewrite_query") as mock_rw, \
         patch("src.generation.chain.get_retriever", return_value=mock_filtered_retriever) as mock_gr:
        mock_rw.return_value = RewrittenQuery(query="Baruch admissions", school="baruch")
        result = ask("how do i get into baruch", mock_retriever, mock_llm)

    mock_gr.assert_called_once_with(mock_vectorstore, metadata_filter={"school": "baruch"})
    assert result.answer == "You need a 3.0 GPA."
