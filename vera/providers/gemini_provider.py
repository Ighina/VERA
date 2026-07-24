"""
Google Gemini provider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types

from vera.providers.base import BaseProvider, ErrorResult, ProviderConfig
from vera.providers.prompts import (
    build_system_prompt,
    build_user_prompt,
    parse_response,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Error extraction via the Google Gemini API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        model = config.model or "gemini-2.5-flash"
        self._client = genai.Client(api_key=config.api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

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
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            logger.exception("Gemini API call failed for %s", paper_id)
            return []

        raw = response.text or ""
        parsed = parse_response(raw, paper_id, self.name)
        return [ErrorResult(**p) for p in parsed]
