"""Tests the /query pipeline end-to-end with HF, Pinecone, and Groq all
mocked out — so CI can verify the wiring (retrieve -> build context ->
generate -> shape the response) without real API keys or network calls."""

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _fake_match(page, heading, text, score):
    return SimpleNamespace(
        metadata={"page_number": page, "section_heading": heading, "chunk_text": text},
        score=score,
    )


def test_query_returns_answer_with_sources():
    fake_matches = [_fake_match(12, "Section 3.2 Limits", "The max is 42.", 0.87)]
    fake_groq_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='The max is 42 [p.12, "Section 3.2 Limits"].'))]
    )

    with (
        patch("app.rag.pipeline.embed_text", return_value=[0.0] * 384),
        patch("app.rag.pipeline.query_chunks", return_value=fake_matches),
        patch("app.rag.pipeline._client.chat.completions.create", return_value=fake_groq_response),
    ):
        res = client.post("/query", json={"query": "What is the max?"})

    assert res.status_code == 200
    body = res.json()
    assert "42" in body["answer"]
    assert body["sources"] == [
        {"page": 12, "heading": "Section 3.2 Limits", "snippet": "The max is 42.", "score": 0.87}
    ]


def test_query_with_no_matches_returns_not_found_without_calling_groq():
    with (
        patch("app.rag.pipeline.embed_text", return_value=[0.0] * 384),
        patch("app.rag.pipeline.query_chunks", return_value=[]),
        patch("app.rag.pipeline._client.chat.completions.create") as mock_groq,
    ):
        res = client.post("/query", json={"query": "Anything not in the doc?"})

    assert res.status_code == 200
    assert res.json()["sources"] == []
    assert "does not appear to contain" in res.json()["answer"]
    mock_groq.assert_not_called()
