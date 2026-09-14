"""Turns text into vectors via the HuggingFace Inference API (free tier).

Groq (used for answer generation) has no embeddings endpoint, so embedding
is a separate hop through HF's serverless `feature_extraction` task.

Two things this file exists specifically to get right (see SPECS.md §8
risks 1-2):
  1. `provider` must be set explicitly to "hf-inference" — that task is
     only supported by "hf-inference" and "Scaleway"; the client's default
     "auto" provider selection cannot be trusted to pick a working one.
  2. Free-tier serverless calls can silently queue instead of returning an
     explicit rate-limit error, so every call has an explicit timeout and
     is retried with backoff rather than hanging indefinitely.
"""

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_REQUEST_TIMEOUT_SECONDS = 30

_client = InferenceClient(
    provider="hf-inference",
    api_key=settings.hf_api_token,
    timeout=_REQUEST_TIMEOUT_SECONDS,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def embed_text(text: str) -> list[float]:
    """Embed a single string, returning a flat list of floats.

    The raw feature-extraction response can come back as a nested
    (per-token) array for some models rather than a single pooled vector —
    we mean-pool defensively so callers always get one flat vector back,
    regardless of which candidate model (all-MiniLM-L6-v2 / bge-small) ends
    up being the warm one on a given day.
    """
    result = _client.feature_extraction(text, model=settings.hf_embedding_model)
    vector = result.tolist() if hasattr(result, "tolist") else result

    if vector and isinstance(vector[0], list):
        # token-level output (list of per-token vectors) -> mean-pool to one vector
        dim = len(vector[0])
        vector = [sum(token[i] for token in vector) / len(vector) for i in range(dim)]

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings, one HF call per text.

    HF's serverless feature-extraction endpoint doesn't reliably support
    large batched inputs on the free tier, so we keep this simple and
    sequential rather than risk a silent partial failure on a big batch.
    Fine for the scale this app targets (a single ~40-page PDF's chunks).
    """
    return [embed_text(t) for t in texts]
