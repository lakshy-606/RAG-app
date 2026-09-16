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
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_llm_response

    with (
        patch("app.rag.get_vectorstore", return_value=fake_vectorstore),
        patch("app.rag._get_chain", return_value=fake_chain),
        # Reranking defaults on; this test isn't about reranking, so make
        # it a passthrough rather than hitting the real Pinecone rerank API.
        patch("app.rag._rerank", side_effect=lambda query, results, top_k: results),
    ):
        res = client.post("/query", json={"query": "What is the max?"})

    assert res.status_code == 200
    body = res.json()
    assert "42" in body["answer"]
    assert body["sources"] == [
        {"page": 12, "heading": "Section 3.2 Limits", "snippet": "The max is 42.", "score": 0.87}
    ]


def test_query_reranks_results_when_enabled():
    # Two candidates, deliberately returned from dense search in the
    # "wrong" order — the fake reranker flips them, and the response
    # should reflect the reranked order/scores, not the dense-search ones.
    fake_results = [
        _fake_result(3, "Irrelevant Section", "Bananas are yellow.", 0.50),
        _fake_result(21, "Input Current Limit", "The max input current is 3A.", 0.40),
    ]
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_score.return_value = fake_results

    fake_rerank_item_0 = MagicMock(index=1, score=0.99)  # the current-limit chunk
    fake_rerank_item_1 = MagicMock(index=0, score=0.01)  # the banana chunk
    fake_rerank_result = MagicMock(data=[fake_rerank_item_0, fake_rerank_item_1])
    fake_pc = MagicMock()
    fake_pc.inference.rerank.return_value = fake_rerank_result

    fake_llm_response = MagicMock()
    fake_llm_response.content = 'The max input current is 3A [p.21, "Input Current Limit"].'
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_llm_response

    with (
        patch("app.rag.get_vectorstore", return_value=fake_vectorstore),
        patch("app.rag._get_chain", return_value=fake_chain),
        patch("app.rag._get_pinecone_client", return_value=fake_pc),
    ):
        res = client.post("/query", json={"query": "What is the max input current?", "use_reranking": True})

    assert res.status_code == 200
    sources = res.json()["sources"]
    assert sources[0]["page"] == 21
    assert sources[0]["score"] == 0.99
    assert sources[1]["page"] == 3
    assert sources[1]["score"] == 0.01


def test_query_with_no_matches_returns_not_found_without_calling_llm():
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_score.return_value = []

    fake_chain = MagicMock()

    with (
        patch("app.rag.get_vectorstore", return_value=fake_vectorstore),
        patch("app.rag._get_chain", return_value=fake_chain),
    ):
        res = client.post("/query", json={"query": "Anything not in the doc?"})

    assert res.status_code == 200
    assert res.json()["sources"] == []
    assert "does not appear to contain" in res.json()["answer"]
    fake_chain.invoke.assert_not_called()


def test_query_rejects_unknown_embedding_model():
    res = client.post("/query", json={"query": "hi", "embedding_model": "not-a-real-model"})
    assert res.status_code == 400


def test_query_rejects_unknown_chat_model():
    res = client.post("/query", json={"query": "hi", "chat_model": "not-a-real-model"})
    assert res.status_code == 400


def test_config_lists_allowed_models():
    res = client.get("/config")
    assert res.status_code == 200
    body = res.json()
    assert "text-embedding-3-small" in body["embedding_models"]
    assert body["default_embedding_model"] == "text-embedding-3-small"
    assert "gpt-4o-mini" in body["chat_models"]
    assert body["default_chat_model"] == "gpt-4o-mini"


def test_reindex_rejects_unknown_embedding_model():
    res = client.post("/reindex", json={"embedding_model": "not-a-real-model"})
    assert res.status_code == 400


def test_reindex_is_a_noop_when_already_indexed():
    with patch("app.retriever.index_has_data", return_value=True):
        res = client.post("/reindex", json={"embedding_model": "text-embedding-3-small"})

    assert res.status_code == 200
    assert res.json()["status"] == "already_indexed"
