"""
OpenAI provider (GPT-4, GPT-4o, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import OpenAI

from vera.providers.base import BaseProvider, ErrorResult, ProviderConfig
from vera.providers.prompts import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Error extraction via the OpenAI Chat Completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        model = config.model or "gpt-4o"
        self._client = OpenAI(api_key=config.api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("OpenAI API call failed for %s", paper_id)
            return []

        raw = response.choices[0].message.content or ""
        parsed = parse_response(raw, paper_id, self.name)
        return [ErrorResult(**p) for p in parsed]
