"""Smoke test: the app boots and /health responds. This is the fast,
dependency-free check the CI test job runs on every push."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
