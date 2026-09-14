"""Tests the /query pipeline end-to-end with the Pinecone vector store and
the OpenAI chat model both mocked out — so CI can verify the wiring
(retrieve -> build context -> generate -> shape the response) without real
API keys or network calls."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)


def _fake_result(page, heading, text, score):
    return (Document(page_content=text, metadata={"page_number": page, "section_heading": heading}), score)


def test_query_returns_answer_with_sources():
    fake_results = [_fake_result(12, "Section 3.2 Limits", "The max is 42.", 0.87)]
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_score.return_value = fake_results

    fake_llm_response = MagicMock()
    fake_llm_response.content = 'The max is 42 [p.12, "Section 3.2 Limits"].'

    with (
        patch("app.rag.pipeline.get_vectorstore", return_value=fake_vectorstore),
        patch("app.rag.pipeline._chain.invoke", return_value=fake_llm_response),
    ):
        res = client.post("/query", json={"query": "What is the max?"})

    assert res.status_code == 200
    body = res.json()
    assert "42" in body["answer"]
    assert body["sources"] == [
        {"page": 12, "heading": "Section 3.2 Limits", "snippet": "The max is 42.", "score": 0.87}
    ]


def test_query_with_no_matches_returns_not_found_without_calling_llm():
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_score.return_value = []

    with (
        patch("app.rag.pipeline.get_vectorstore", return_value=fake_vectorstore),
        patch("app.rag.pipeline._chain.invoke") as mock_invoke,
    ):
        res = client.post("/query", json={"query": "Anything not in the doc?"})

    assert res.status_code == 200
    assert res.json()["sources"] == []
    assert "does not appear to contain" in res.json()["answer"]
    mock_invoke.assert_not_called()
