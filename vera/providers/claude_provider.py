"""
Anthropic Claude provider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from anthropic import Anthropic

from vera.providers.base import BaseProvider, ErrorResult, ProviderConfig
from vera.providers.prompts import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseProvider):
    """Error extraction via the Anthropic Claude API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        model = config.model or "claude-sonnet-5"
        base_url = config.extra.get("base_url")
        kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)  # type: ignore[arg-type]
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    def extract_errors(
        self,
        paper_id: str,
        paper_text: str,
        review_text: str,
        taxonomy: List[Dict[str, Any]],
    ) -> List[ErrorResult]:
        system_prompt = build_system_prompt(taxonomy)
        user_prompt = build_user_prompt(paper_text, review_text)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            logger.exception("Claude API call failed for %s", paper_id)
            return []

        # Claude returns content blocks
        raw = ""
        for block in response.content:
            if block.type == "text":
                raw += block.text

        parsed = parse_response(raw, paper_id, self.name)
        return [ErrorResult(**p) for p in parsed]
