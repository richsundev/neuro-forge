"""Real-model provider adapters. Structurally complete, network calls are the only thing that
differs from the mock provider — the optimizer and evaluators never know the difference."""

from __future__ import annotations

import os
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from neuroforge.genomes.schema import ModelConfig
from neuroforge.providers.base import GenerationRequest, GenerationResult, LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible chat-completions endpoint (OpenAI, vLLM, Ollama's OpenAI
    shim, etc). Requires `base_url` and `api_key` — disabled unless both are configured, so CI and
    local dev never accidentally depend on a paid API (see docs/design-decisions.md)."""

    name = "openai_compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("NEUROFORGE_OPENAI_BASE_URL", "")
        self.api_key = api_key or os.environ.get("NEUROFORGE_OPENAI_API_KEY", "")

    def generate(self, model: ModelConfig, request: GenerationRequest) -> GenerationResult:
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "OpenAICompatibleProvider requires NEUROFORGE_OPENAI_BASE_URL and "
                "NEUROFORGE_OPENAI_API_KEY — use MockLLMProvider for local/CI runs."
            )
        import json

        payload = json.dumps(
            {
                "model": model.name,
                "temperature": model.temperature,
                "max_tokens": model.max_tokens,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_input},
                ],
            }
        ).encode("utf-8")
        req = urlrequest.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.monotonic()
        try:
            with urlrequest.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
        except urlerror.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return GenerationResult(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
            capability_signal=0.0,
        )


class AnthropicCompatibleProvider(LLMProvider):
    """Talks to the Anthropic Messages API. Same activation guard as OpenAICompatibleProvider."""

    name = "anthropic_compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or os.environ.get(
            "NEUROFORGE_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
        )
        self.api_key = api_key or os.environ.get("NEUROFORGE_ANTHROPIC_API_KEY", "")

    def generate(self, model: ModelConfig, request: GenerationRequest) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError(
                "AnthropicCompatibleProvider requires NEUROFORGE_ANTHROPIC_API_KEY — "
                "use MockLLMProvider for local/CI runs."
            )
        import json

        payload = json.dumps(
            {
                "model": model.name,
                "max_tokens": model.max_tokens,
                "temperature": model.temperature,
                "system": request.system_prompt,
                "messages": [{"role": "user", "content": request.user_input}],
            }
        ).encode("utf-8")
        req = urlrequest.Request(
            url=f"{self.base_url.rstrip('/')}/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        start = time.monotonic()
        try:
            with urlrequest.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
        except urlerror.URLError as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000
        text = "".join(block.get("text", "") for block in body.get("content", []))
        usage = body.get("usage", {})
        return GenerationResult(
            text=text,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
            capability_signal=0.0,
        )
