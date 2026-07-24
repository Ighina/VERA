"""
Provider registry — maps provider names to their implementations
and loads configuration from environment variables and a config file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from vera.providers.base import BaseProvider, ProviderConfig
from vera.providers.claude_provider import ClaudeProvider
from vera.providers.deepseek_provider import DeepSeekProvider
from vera.providers.gemini_provider import GeminiProvider
from vera.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# Mapping from provider name → (class, default_model, env_var_for_api_key)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, Any] = {
    "openai": {
        "class": OpenAIProvider,
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "class": DeepSeekProvider,
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "gemini": {
        "class": GeminiProvider,
        "default_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "claude": {
        "class": ClaudeProvider,
        "default_model": "claude-sonnet-5",
        "env_key": "ANTHROPIC_AUTH_TOKEN",
    },
}


def list_providers() -> List[str]:
    """Return the names of all known providers."""
    return list(_PROVIDER_REGISTRY.keys())


def create_provider(
    name: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    extra: Optional[Dict[str, Any]] = None,
) -> BaseProvider:
    """Create a provider instance by name.

    API keys are resolved in this order:
    1. Explicit ``api_key`` argument
    2. Environment variable (``OPENAI_API_KEY``, ``DEEPSEEK_API_KEY``, etc.)
    3. Falls back to empty string (provider raises on use if needed)

    Parameters
    ----------
    name:
        One of ``openai``, ``deepseek``, ``gemini``, ``claude``.
    model:
        Override the default model for this provider.
    api_key:
        Explicit API key.
    temperature:
        LLM sampling temperature (default 0 for deterministic output).
    max_tokens:
        Maximum output tokens.
    extra:
        Additional provider-specific kwargs (e.g. ``base_url``).
    """
    entry = _PROVIDER_REGISTRY.get(name.lower())
    if entry is None:
        raise ValueError(
            f"Unknown provider '{name}'. Known: {list_providers()}"
        )

    # Resolve API key
    resolved_key = api_key or os.environ.get(entry["env_key"], "")

    # Resolve model
    resolved_model = model or entry["default_model"]

    config = ProviderConfig(
        model=resolved_model,
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra=extra or {},
    )

    return entry["class"](config)


def create_providers_from_env(
    *,
    names: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> List[BaseProvider]:
    """Create all configured providers whose API keys are available.

    Parameters
    ----------
    names:
        Subset of provider names to try.  ``None`` means all registered.
    temperature / max_tokens:
        Passed to every provider.
    """
    if names is None:
        names = list_providers()

    providers: List[BaseProvider] = []
    for name in names:
        cfg = _PROVIDER_REGISTRY[name]
        env_key = cfg["env_key"]
        api_key = os.environ.get(env_key, "")
        if not api_key:
            continue  # skip providers without credentials
        prov = create_provider(
            name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        providers.append(prov)
    return providers
