"""
Error-extraction pipeline.

Orchestrates:
1. Loading processed samples from PeerRead
2. Running each sample through multiple LLM providers
3. Aggregating results into a unified error dataset
4. Exporting to disk (JSONL / Parquet)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from vera.data.loader import load_processed_samples
from vera.providers.base import BaseProvider, ErrorResult
from vera.providers.registry import create_provider, list_providers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------


def load_taxonomy(path: str | Path) -> List[Dict[str, Any]]:
    """Load the error taxonomy from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    taxonomy = data.get("taxonomy", [])
    if not taxonomy:
        raise ValueError(f"No taxonomy entries in {path}")
    return taxonomy


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class ErrorExtractionPipeline:
    """Run error extraction across papers and providers.

    Typical usage::

        pipeline = ErrorExtractionPipeline(
            data_root="data/PeerRead-master/data",
            taxonomy_path="config/taxonomy.yaml",
            providers=["openai", "claude"],
        )
        dataset = pipeline.run(max_samples=10)
        pipeline.export(dataset, "output/errors.jsonl")
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        taxonomy_path: str | Path,
        providers: Optional[List[str]] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        splits: Optional[List[str]] = None,
        conferences: Optional[List[str]] = None,
        require_reviews: bool = True,
        only_rejected: bool = True,
    ):
        self.data_root = str(data_root)
        self.taxonomy = load_taxonomy(taxonomy_path)

        # Resolve providers
        provider_names = providers or list_providers()
        self.providers: List[BaseProvider] = []
        for name in provider_names:
            kwargs = provider_kwargs or {}
            try:
                prov = create_provider(name, **kwargs)
                self.providers.append(prov)
            except Exception as exc:
                logger.warning("Could not create provider '%s': %s", name, exc)

        if not self.providers:
            raise RuntimeError(
                "No providers available. Set the appropriate API key "
                "environment variables (OPENAI_API_KEY, DEEPSEEK_API_KEY, "
                "GEMINI_API_KEY, ANTHROPIC_AUTH_TOKEN)."
            )

        self.splits = splits
        self.conferences = conferences
        self.require_reviews = require_reviews
        self.only_rejected = only_rejected

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        max_samples: Optional[int] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Execute the full pipeline and return the error dataset.

        Parameters
        ----------
        max_samples:
            Limit the number of papers processed (useful for testing).
        progress_callback:
            Optional callable(paper_id, idx, total) called after each paper.
        """
        samples = list(
            load_processed_samples(
                self.data_root,
                splits=self.splits,
                conferences=self.conferences,
                require_reviews=self.require_reviews,
                only_rejected=self.only_rejected,
            )
        )

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]

        total = len(samples)
        logger.info("Loaded %d samples; running %d provider(s)", total, len(self.providers))

        all_errors: List[Dict[str, Any]] = []
        papers_with_errors = 0
        papers_without_errors = 0

        for idx, sample in enumerate(samples):
            paper_id = sample["paper_id"]
            paper_text = sample["full_text"]
            review_text = sample["full_review"]

            if progress_callback:
                progress_callback(paper_id, idx, total)

            found_any = False
            for provider in self.providers:
                logger.debug(
                    "Running %s on %s (%d/%d)",
                    provider.name,
                    paper_id,
                    idx + 1,
                    total,
                )
                errors = provider.extract_errors(
                    paper_id=paper_id,
                    paper_text=paper_text,
                    review_text=review_text,
                    taxonomy=self.taxonomy,
                )

                for err in errors:
                    all_errors.append(self._enrich_error(err, sample))
                    found_any = True

            if found_any:
                papers_with_errors += 1
            else:
                papers_without_errors += 1

            if (idx + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d papers; %d errors found so far",
                    idx + 1,
                    total,
                    len(all_errors),
                )

        logger.info(
            "Pipeline complete: %d papers, %d errors. "
            "%d papers had errors, %d had none.",
            total,
            len(all_errors),
            papers_with_errors,
            papers_without_errors,
        )

        return all_errors

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export(dataset: List[Dict[str, Any]], output_path: str | Path) -> Path:
        """Write the error dataset to JSONL and a summary stats file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        jsonl_path = output_path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for row in dataset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Exported %d rows to %s", len(dataset), jsonl_path)

        # Write a run metadata file
        run_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_errors": len(dataset),
            "unique_papers": len(set(r["paper_id"] for r in dataset)),
            "providers": list(set(r["provider"] for r in dataset)),
            "error_types": _summarise_error_types(dataset),
        }
        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(run_meta, fh, indent=2, ensure_ascii=False)

        logger.info("Run metadata written to %s", meta_path)
        return jsonl_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich_error(
        self, err: ErrorResult, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add metadata fields from the source sample to the error row."""
        meta = sample.get("metadata", {})
        return {
            **asdict(err),
            "paper_title": meta.get("title", ""),
            "paper_authors": meta.get("authors", []),
            "paper_conference": meta.get("conference", ""),
            "paper_year": meta.get("year"),
            "paper_accepted": meta.get("accepted"),
            "num_reviews": meta.get("num_reviews", 0),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarise_error_types(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count occurrences of each error type."""
    counts: Dict[str, int] = {}
    for row in dataset:
        etype = row.get("error_type", "unknown")
        counts[etype] = counts.get(etype, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
