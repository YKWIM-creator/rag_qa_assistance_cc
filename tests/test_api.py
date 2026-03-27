import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_health_endpoint():
    with patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_returns_answer():
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "Baruch is located in Manhattan."
    mock_rag_response.sources = [{"url": "http://baruch.cuny.edu", "school": "baruch", "title": "About"}]

    with patch("src.api.main.ask", return_value=mock_rag_response), \
         patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": "Where is Baruch?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_ask_endpoint_rejects_empty_question():
    with patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
