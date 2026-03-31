import pytest
from unittest.mock import MagicMock, patch
from src.generation.rewriter import rewrite_query, RewrittenQuery


def test_rewritten_query_dataclass():
    q = RewrittenQuery(query="admissions requirements", school="baruch")
    assert q.query == "admissions requirements"
    assert q.school == "baruch"


def test_rewritten_query_no_school():
    q = RewrittenQuery(query="what are CUNY programs?", school=None)
    assert q.school is None


def test_rewrite_query_calls_llm(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"school": "baruch", "query": "Baruch admissions requirements"}'

    result = rewrite_query("how do i get into baruch", mock_llm)

    assert isinstance(result, RewrittenQuery)
    assert result.school == "baruch"
    assert "baruch" in result.query.lower()
    mock_llm.invoke.assert_called_once()


def test_rewrite_query_handles_no_school(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"school": null, "query": "CUNY financial aid options"}'

    result = rewrite_query("what financial aid does cuny offer", mock_llm)

    assert result.school is None
    assert result.query == "CUNY financial aid options"


def test_rewrite_query_falls_back_on_bad_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "I cannot parse this"

    result = rewrite_query("original question", mock_llm)

    assert result.query == "original question"
    assert result.school is None
