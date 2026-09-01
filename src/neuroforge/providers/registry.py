from __future__ import annotations

from neuroforge.providers.base import LLMProvider
from neuroforge.providers.compatible import AnthropicCompatibleProvider, OpenAICompatibleProvider
from neuroforge.providers.mock import MockLLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "mock": MockLLMProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic_compatible": AnthropicCompatibleProvider,
    "local": MockLLMProvider,
}


def get_provider(provider_name: str) -> LLMProvider:
    cls = _REGISTRY.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider_name}'. Known: {list(_REGISTRY)}")
    return cls()
