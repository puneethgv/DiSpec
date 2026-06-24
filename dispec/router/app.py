"""FastAPI serving layer for DiSpec.

Endpoints:
  POST /generate  {prompt, max_new_tokens, temperature} -> {text, num_tokens}
  GET  /health
  GET  /metrics   Prometheus exposition (scrape target)

Run: python -m dispec.router.app   (loads the target model and serves on :8000)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from dispec.router.server import InferenceServer


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0


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
        return await server.generate(req.prompt, req.max_new_tokens, req.temperature)

    @app.get("/metrics")
    async def get_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn

    from dispec.models.loader import load_target
    model, tok = load_target()
    app = create_app(model, tok)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
