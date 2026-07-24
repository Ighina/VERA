"""
Shared prompt construction for error extraction across all providers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def build_system_prompt(taxonomy: List[Dict[str, Any]]) -> str:
    """Build the system prompt that describes the error-extraction task."""
    tax_lines: List[str] = []
    for cat in taxonomy:
        tax_lines.append(f"- **{cat['id']}** ({cat['name']}): {cat['description']}")
        for ex in cat.get("examples", []):
            tax_lines.append(f"  *Example*: {ex}")

    tax_block = "\n".join(tax_lines)

    return f"""You are an expert peer-review auditor. Your job is to read a scientific paper
and its peer reviews, then identify errors in the paper according to a fixed taxonomy.

## Error taxonomy

{tax_block}

## Instructions

1. Carefully read the entire paper text and the peer review(s).
2. For each category in the taxonomy, determine whether the paper contains any
   errors that match that category.  Use the review text as supporting evidence
   — reviewers may have already identified issues.
3. For each error you find, note:
   - The taxonomy category id
   - The specific section or paragraph in the paper where the error occurs
   - A clear rationale explaining why this qualifies as an error
4. Be rigorous: only flag genuine issues.  If you are unsure, do NOT flag.
5. If no errors from the taxonomy are present, return an empty list.

## Output format

Return a JSON object with a single key ``errors`` containing a list of error
objects.  Each error object MUST have these fields:

- ``error_type``: (string) the taxonomy category id
- ``error_name``: (string) the human-readable category name
- ``location``: (string) the section/paragraph in the paper
- ``rationale``: (string) why this qualifies as an error

If no errors found, return: {{"errors": []}}

Return ONLY the JSON object, nothing else."""


def build_user_prompt(paper_text: str, review_text: str) -> str:
    """Build the user prompt containing the paper and review to analyse."""

    # Truncate paper text if it's extremely long (providers have context limits)
    max_paper_chars = 16000
    paper_section = paper_text if len(paper_text) <= max_paper_chars else (
        paper_text[:max_paper_chars]
        + f"\n\n[... paper truncated at {max_paper_chars} characters; "
        f"total length was {len(paper_text)} characters ...]"
    )

    return f"""## Paper text

{paper_section}

## Peer review(s)

{review_text}

Please identify any errors in the paper according to the taxonomy.
Return your answer as the JSON object described above."""


def parse_response(raw: str, paper_id: str, provider_name: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of error dicts.

    Handles models that wrap JSON in markdown fences or prefix with text.
    """
    text = raw.strip()

    # Try to extract JSON from markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        # Find the JSON block
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{"):
                text = p
                break

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    errors = data.get("errors", [])
    if not isinstance(errors, list):
        return []

    results: List[Dict[str, Any]] = []
    for err in errors:
        results.append({
            "paper_id": paper_id,
            "provider": provider_name,
            "error_type": err.get("error_type", "unknown"),
            "error_name": err.get("error_name", "Unknown"),
            "location": err.get("location", ""),
            "rationale": err.get("rationale", ""),
            "confidence": float(err.get("confidence", 1.0)),
        })

    return results
