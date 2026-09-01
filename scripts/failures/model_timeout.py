#!/usr/bin/env python3
"""Failure injection: a real-model provider pointed at an unreachable endpoint must fail loudly
and specifically (not hang, not silently return garbage) — section 51.

    python scripts/failures/model_timeout.py
"""

from __future__ import annotations

import time

from neuroforge.genomes.schema import ModelConfig
from neuroforge.providers.base import GenerationRequest
from neuroforge.providers.compatible import OpenAICompatibleProvider


def main() -> None:
    # Port 1 on localhost: almost certainly nothing is listening, so the OS refuses the connection
    # immediately (fast, deterministic) rather than us waiting out a black-hole timeout.
    provider = OpenAICompatibleProvider(base_url="http://127.0.0.1:1", api_key="test-key")
    request = GenerationRequest(system_prompt="You are a test.", user_input="ping")

    print("Calling an unreachable OpenAI-compatible endpoint...")
    start = time.perf_counter()
    try:
        provider.generate(ModelConfig(provider="openai_compatible", name="gpt-test"), request)
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start
        print(f"PASS: raised RuntimeError after {elapsed:.2f}s: {exc}")
        return
    raise AssertionError("expected a RuntimeError from an unreachable provider endpoint")


if __name__ == "__main__":
    main()
