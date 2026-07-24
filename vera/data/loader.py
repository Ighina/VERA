"""
PeerRead data loader.

Reads the raw PeerRead JSON files from disk and produces a unified
representation where each sample has:
  - paper_id:   unique identifier
  - full_text:  the entire paper text pasted together from sections
  - full_review: the entire review text pasted from the review comments
  - metadata:   additional fields (conference, authors, accepted, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema of a processed sample
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA = {
    "paper_id": str,
    "full_text": str,
    "full_review": str,
    "metadata": dict,
}


def _paste_sections(sections: List[Dict[str, Any]]) -> str:
    """Concatenate section headings and text into one continuous string."""
    parts: List[str] = []
    for sec in sections:
        heading = sec.get("heading")
        text = sec.get("text", "")
        if heading:
            parts.append(f"## {heading}\n\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _paste_reviews(reviews: List[Dict[str, Any]]) -> str:
    """Concatenate review comments into one continuous string.

    Each review's ``comments`` field is the main free-text content.
    We also include numeric scores when present (prefixed for clarity).
    """
    blocks: List[str] = []
    for i, rev in enumerate(reviews):
        lines: List[str] = [f"### Review {i + 1}"]
        # Numeric / categorical scores first
        for key in sorted(rev):
            if key in ("comments", "is_meta_review", "IS_META_REVIEW"):
                continue
            val = rev[key]
            if val is not None:
                lines.append(f"- **{key}**: {val}")
        # Then the free-text comment
        comment = rev.get("comments") or ""
        if comment.strip():
            lines.append(f"\n{comment}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_processed_samples(
    data_root: str | Path,
    *,
    splits: Optional[List[str]] = None,
    conferences: Optional[List[str]] = None,
    require_reviews: bool = True,
    only_rejected: bool = False,
    min_review_length: int = 50,
    min_text_length: int = 100,
) -> Iterator[Dict[str, Any]]:
    """Walk the PeerRead directory tree and yield processed samples.

    Parameters
    ----------
    data_root:
        Path to the extracted ``PeerRead-master/data`` directory.
    splits:
        Subset of ``["train", "dev", "test"]``.  ``None`` means all.
    conferences:
        Subset of conference directories (e.g. ``["acl_2017"]``).
        ``None`` means all.
    require_reviews:
        If *True* (default), skip papers that have no review text.
    only_rejected:
        If *True*, keep only papers explicitly marked as rejected
        (``accepted == False``).  Papers with no acceptance label
        (e.g. the acl_2017 and conll_2016 sections, which contain
        accepted papers only) are also skipped.
    min_review_length:
        Minimum number of characters in the pasted review.  Shorter
        reviews are skipped.  Default 50.
    min_text_length:
        Minimum number of characters in the pasted paper text.  Shorter
        texts are skipped.  Default 100.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if splits is None:
        splits = ["train", "dev", "test"]

    # Discover conferences
    conf_dirs: List[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if conferences and child.name not in conferences:
            continue
        conf_dirs.append(child)

    total_yielded = 0
    total_skipped = 0

    for conf_dir in conf_dirs:
        conf_name = conf_dir.name
        for split in splits:
            split_dir = conf_dir / split
            pdf_dir = split_dir / "parsed_pdfs"
            review_dir = split_dir / "reviews"

            if not pdf_dir.is_dir():
                continue  # e.g. NIPS data not downloaded

            for pdf_path in sorted(pdf_dir.glob("*.json")):
                # Derive the review filename – it is usually the same stem
                # but *without* the ``.pdf`` infix.
                pdf_stem = pdf_path.stem  # e.g. "316.pdf" → stem "316.pdf"
                paper_id = pdf_stem.replace(".pdf", "")

                # Look for matching review
                review_path = review_dir / f"{paper_id}.json"
                if not review_path.is_file():
                    logger.debug("No review for %s/%s/%s", conf_name, split, paper_id)
                    total_skipped += 1
                    continue

                try:
                    sample = _process_one(
                        pdf_path,
                        review_path,
                        paper_id,
                        conf_name,
                        split,
                        require_reviews=require_reviews,
                        only_rejected=only_rejected,
                        min_review_length=min_review_length,
                        min_text_length=min_text_length,
                    )
                except Exception:
                    logger.exception(
                        "Failed to process %s/%s/%s", conf_name, split, paper_id
                    )
                    total_skipped += 1
                    continue

                if sample is None:
                    total_skipped += 1
                    continue

                total_yielded += 1
                yield sample

    logger.info(
        "PeerRead loading complete: %d samples yielded, %d skipped",
        total_yielded,
        total_skipped,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_one(
    pdf_path: Path,
    review_path: Path,
    paper_id: str,
    conference: str,
    split: str,
    *,
    require_reviews: bool,
    only_rejected: bool,
    min_review_length: int,
    min_text_length: int,
) -> Optional[Dict[str, Any]]:
    """Load and validate a single paper + review pair."""

    with open(pdf_path, encoding="utf-8") as fh:
        pdf_data = json.load(fh)
    with open(review_path, encoding="utf-8") as fh:
        review_data = json.load(fh)

    if only_rejected and review_data.get("accepted") is not False:
        logger.debug(
            "Skipping %s: accepted=%s (only_rejected)",
            paper_id,
            review_data.get("accepted"),
        )
        return None

    # --- Build full_text from sections ---
    meta = pdf_data.get("metadata", {})
    sections = meta.get("sections") or []

    # Some entries have an abstractText but empty sections
    abstract = meta.get("abstractText") or ""
    full_text = _paste_sections(sections)
    if not full_text.strip() and abstract.strip():
        full_text = abstract
    if abstract.strip() and abstract.strip() not in full_text:
        full_text = f"**Abstract**: {abstract}\n\n{full_text}"

    if len(full_text.strip()) < min_text_length:
        logger.debug("Skipping %s: text too short (%d chars)", paper_id, len(full_text))
        return None

    # --- Build full_review from review comments ---
    reviews = review_data.get("reviews") or []
    full_review = _paste_reviews(reviews)

    if require_reviews and not full_review.strip():
        logger.debug("Skipping %s: empty review", paper_id)
        return None

    if len(full_review.strip()) < min_review_length:
        logger.debug("Skipping %s: review too short (%d chars)", paper_id, len(full_review))
        return None

    # --- Metadata ---
    metadata: Dict[str, Any] = {
        "conference": conference,
        "split": split,
        "title": meta.get("title") or review_data.get("title", ""),
        "authors": meta.get("authors") or review_data.get("authors", []),
        "year": meta.get("year"),
        "accepted": review_data.get("accepted"),
        "source": meta.get("source", ""),
        "num_reviews": len(reviews),
        "num_sections": len(sections),
    }

    return {
        "paper_id": f"{conference}/{paper_id}",
        "full_text": full_text.strip(),
        "full_review": full_review.strip(),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Convenience: load everything into a list (useful for testing / small runs)
# ---------------------------------------------------------------------------


def load_all(data_root: str | Path, **kwargs: Any) -> List[Dict[str, Any]]:
    """Return all processed samples as a list."""
    return list(load_processed_samples(data_root, **kwargs))
