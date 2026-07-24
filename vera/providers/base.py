"""
Abstract base for LLM providers.

Every provider implements ``extract_errors``, which takes a paper text,
its review, and a taxonomy definition, and returns zero or more
structured error findings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ErrorResult:
    """A single error found by a provider."""

    paper_id: str
    provider: str
    error_type: str          # taxonomy category id, e.g. "methodological_flaw"
    error_name: str          # human-readable category name
    location: str            # section / paragraph reference in the paper
    rationale: str           # why the provider considers this an error
    confidence: float = 1.0  # 0-1 confidence score (provider-specific)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderConfig:
    """Configuration passed to a provider constructor."""

    model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Interface every LLM provider must implement."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'openai', 'deepseek', 'gemini', 'claude'."""
        ...

    @abstractmethod
    def extract_errors(
        self,
        paper_id: str,
        paper_text: str,
        review_text: str,
        taxonomy: List[Dict[str, Any]],
    ) -> List[ErrorResult]:
        """Analyse a paper + review pair and return extracted errors.

        Parameters
        ----------
        paper_id:
            Unique identifier for the paper.
        paper_text:
            Full concatenated text of the paper.
        review_text:
            Full concatenated text of the peer review(s).
        taxonomy:
            List of error-category dicts, each with ``id``, ``name``,
            ``description`` and optionally ``examples``.

        Returns
        -------
        list[ErrorResult]
            Zero or more structured error findings.  An empty list
            signals that no errors from the taxonomy were detected.
        """
        ...
