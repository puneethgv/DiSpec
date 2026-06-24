"""FastAPI serving layer for DiSpec.

Endpoints:
  POST /generate              {prompt, max_new_tokens, temperature} -> {text, num_tokens}
  POST /v1/chat/completions   OpenAI-compatible chat (streaming + non-streaming)
  GET  /health
  GET  /metrics               Prometheus exposition (scrape target)
  GET  /dashboard             built-in live metrics page

Run: python -m dispec.router.app   (loads the target model and serves on :8000)
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from dispec.router import metrics as M
from dispec.router.dashboard import DASHBOARD_HTML
from dispec.router.server import InferenceServer


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    priority: int = 0  # higher = scheduled sooner (SLO-aware routing)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "dispec"
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


def create_app(model, tokenizer, **server_kwargs) -> FastAPI:
    server = InferenceServer(model, tokenizer, **server_kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.start()
        yield
        server.stop()

    app = FastAPI(title="DiSpec", lifespan=lifespan)
    app.state.server = server

    @app.get("/health")
    async def health():
        return {"status": "ok",
                "running": len(server.engine.running),
                "waiting": len(server.engine.waiting)}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        return await server.generate(req.prompt, req.max_new_tokens,
                                     req.temperature, req.priority)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completions (works with standard OpenAI clients)."""
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            async def event_stream():
                async for delta in server.generate_stream(prompt, req.max_tokens, req.temperature):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": req.model,
                             "choices": [{"index": 0, "delta": {"content": delta},
                                          "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = await server.generate(prompt, req.max_tokens, req.temperature)
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": result["text"]}}],
            "usage": {"completion_tokens": result["num_tokens"]},
        }

    @app.get("/metrics")
    async def get_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/stats")
    async def stats():
        """Lightweight JSON snapshot for the built-in dashboard (no Prometheus needed)."""
        def avg_ms(hist):
            # prometheus_client Histogram count = sum of its (non-cumulative) buckets.
            cnt = sum(b.get() for b in hist._buckets)
            return (hist._sum.get() / cnt * 1e3) if cnt else 0.0
        return {
            "running": M.RUNNING._value.get(),
            "waiting": M.WAITING._value.get(),
            "requests_total": M.REQUESTS._value.get(),
            "tokens_total": M.TOKENS._value.get(),
            "ttft_avg_ms": avg_ms(M.TTFT),
            "tpot_avg_ms": avg_ms(M.TPOT),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    return app


def main() -> None:
    import uvicorn

    from dispec.models.loader import load_target
    model, tok = load_target()
    # Fast path: batched Triton attention + fused GEMMs.
    app = create_app(model, tok, attn_backend="triton", fuse=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
