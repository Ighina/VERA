"""
DeepSeek provider.

Uses the DeepSeek API (compatible with the OpenAI client).
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

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(BaseProvider):
    """Error extraction via the DeepSeek API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        model = config.model or "deepseek-chat"
        base_url = config.extra.get("base_url", DEEPSEEK_BASE_URL)
        self._client = OpenAI(api_key=config.api_key, base_url=base_url)
        self._model = model

    @property
    def name(self) -> str:
        return "deepseek"

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
            )
        except Exception:
            logger.exception("DeepSeek API call failed for %s", paper_id)
            return []

        raw = response.choices[0].message.content or ""
        parsed = parse_response(raw, paper_id, self.name)
        return [ErrorResult(**p) for p in parsed]
