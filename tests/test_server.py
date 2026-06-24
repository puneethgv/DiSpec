"""End-to-end smoke test for the FastAPI serving layer.

Boots the app (which starts the background scheduler thread), generates over HTTP, and
checks the Prometheus endpoint reflects the request. Uses the 0.5B model; needs CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from dispec.config import DRAFT_MODEL
    from dispec.models.loader import load_model, load_tokenizer
    from dispec.router.app import create_app

    app = create_app(load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL), num_blocks=512)
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_generate_and_metrics(client):
    r = client.post("/generate", json={"prompt": "The capital of France is",
                                       "max_new_tokens": 16, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["num_tokens"] > 0
    assert isinstance(body["text"], str) and body["text"]

    metrics = client.get("/metrics").text
    assert "dispec_requests_total" in metrics
    assert "dispec_ttft_seconds" in metrics


def test_openai_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_openai_chat_streaming(client):
    chunks = []
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Count: one two"}],
            "max_tokens": 12, "stream": True}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(line)
    assert chunks  # got streamed chunks
    assert any('"delta"' in c for c in chunks)


def test_concurrent_requests(client):
    # Several requests in flight exercise continuous batching through the server.
    import concurrent.futures as cf

    def one(i):
        return client.post("/generate", json={"prompt": f"Count to {i}:",
                                              "max_new_tokens": 12}).json()["num_tokens"]

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        counts = list(ex.map(one, range(4)))
    assert all(c > 0 for c in counts)
