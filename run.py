#!/usr/bin/env python3
"""
VERA — Verification Dataset Generator from Peer-Review Data.

Usage::

    # Run with a single provider
    python run.py --provider openai --max-samples 10

    # Run with all available providers (requires API keys)
    python run.py --max-samples 50

    # Full run
    python run.py --output output/errors.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vera.pipeline.extractor import ErrorExtractionPipeline

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VERA — Verification Dataset from Peer Review"
    )
    # Data
    p.add_argument(
        "--data-root",
        default="data/PeerRead-master/data",
        help="Path to PeerRead-master/data directory",
    )
    p.add_argument(
        "--taxonomy",
        default="config/taxonomy.yaml",
        help="Path to error taxonomy YAML file",
    )
    # Providers
    p.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="Provider name (can be repeated). If omitted, uses all with API keys set.",
    )
    p.add_argument(
        "--model",
        help="Override the default model for every provider",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default 0)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens per API call",
    )
    # Pipeline
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of papers to process (useful for testing)",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Subset of splits, e.g. --splits train dev",
    )
    p.add_argument(
        "--conferences",
        nargs="*",
        default=None,
        help="Subset of conferences, e.g. --conferences acl_2017 iclr_2017",
    )
    p.add_argument(
        "--include-accepted",
        action="store_true",
        help="Also process accepted/unlabelled papers (default: rejected only)",
    )
    # Output
    p.add_argument(
        "--output",
        default="output/errors.jsonl",
        help="Path for the output JSONL file",
    )
    # Logging
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("vera")

    # Check data exists
    if not Path(args.data_root).is_dir():
        logger.error("Data root not found: %s", args.data_root)
        logger.error(
            "Download PeerRead first: "
            "curl -L -o data/peerread_master.zip "
            "https://github.com/allenai/PeerRead/archive/master.zip && "
            "unzip -q data/peerread_master.zip -d data/"
        )
        sys.exit(1)

    # Check taxonomy exists
    if not Path(args.taxonomy).is_file():
        logger.error("Taxonomy file not found: %s", args.taxonomy)
        sys.exit(1)

    # Build provider kwargs
    provider_kwargs: dict = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.model:
        provider_kwargs["model"] = args.model

    # Run pipeline
    pipeline = ErrorExtractionPipeline(
        data_root=args.data_root,
        taxonomy_path=args.taxonomy,
        providers=args.providers,
        provider_kwargs=provider_kwargs,
        splits=args.splits,
        conferences=args.conferences,
        only_rejected=not args.include_accepted,
    )

    logger.info(
        "Starting VERA pipeline with providers: %s",
        [p.name for p in pipeline.providers],
    )

    dataset = pipeline.run(max_samples=args.max_samples)
    output_path = ErrorExtractionPipeline.export(dataset, args.output)

    print(f"\nDone. {len(dataset)} errors written to {output_path}")


if __name__ == "__main__":
    main()
