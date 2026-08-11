"""Manual validation for tampCOCO entry sources.

This script validates the real tampCOCO dataset layout against owl's built-in
entry sources.

It performs two kinds of checks:

1. Load every official tampCOCO list file and report the number of entries.
2. Randomly sample entries from every logical source and verify that the
   referenced input image and ground-truth mask both exist.

The script intentionally does not decode image resources. Resource decoding,
mask alignment, and transform behavior should be inspected separately through
``OwlDataset`` and the dataset visualization utilities.

Example:

    python tests/data/entry/manual/tamp_coco.py \
        "D:/BaiduNetdiskDownload/高速下载- datasets/tampCOCO/tampCOCO"

An optional sample count may be provided:

    python tests/data/entry/manual/tamp_coco.py \
        "D:/datasets/tampCOCO" \
        --samples 100
"""

import argparse
import random
from pathlib import Path

from owl.data.entry.tamp_coco import (
    BcmCOCOEntrySource,
    BcmcCOCOEntrySource,
    CmCOCOEntrySource,
    SpCOCOEntrySource,
)


DEFAULT_SAMPLE_COUNT = 50
RANDOM_SEED = 42


def validate_source(
    *,
    name: str,
    source: object,
    sample_count: int,
    random_generator: random.Random,
) -> int:
    """Validate randomly sampled resource paths from one entry source.

    Args:
        name:
            Human-readable logical source name.
        source:
            Initialized tampCOCO entry source.
        sample_count:
            Maximum number of entries to sample.
        random_generator:
            Deterministic random generator used for sampling.

    Returns:
        Number of sampled entries containing at least one missing resource.
    """
    source_size = len(source)

    print()
    print(f"[{name}]")
    print(f"entries: {source_size}")

    first = source[0]
    last = source[-1]

    print("first:")
    print(f"  tp: {first.tp}")
    print(f"  gt: {first.gt}")

    print("last:")
    print(f"  tp: {last.tp}")
    print(f"  gt: {last.gt}")

    actual_sample_count = min(sample_count, source_size)
    indices = random_generator.sample(
        range(source_size),
        actual_sample_count,
    )

    failed = 0

    for index in indices:
        entry = source[index]

        tp_exists = entry.tp.is_file()
        gt_exists = entry.gt is not None and entry.gt.is_file()

        if tp_exists and gt_exists:
            continue

        failed += 1

        print()
        print(f"missing resource at index {index}:")
        print(f"  tp: {entry.tp} ({'ok' if tp_exists else 'missing'})")
        print(
            f"  gt: {entry.gt} "
            f"({'ok' if gt_exists else 'missing'})"
        )

    passed = actual_sample_count - failed

    print(
        f"random checks: "
        f"{passed}/{actual_sample_count} valid"
    )

    return failed


def main() -> None:
    """Run manual tampCOCO entry-source validation."""
    parser = argparse.ArgumentParser(
        description="Validate owl tampCOCO entry sources.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the tampCOCO dataset root.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=(
            "Number of random entries checked per source "
            f"(default: {DEFAULT_SAMPLE_COUNT})."
        ),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")

    root = args.root

    print(f"tampCOCO root: {root}")
    print(f"random seed: {RANDOM_SEED}")
    print(f"samples per source: {args.samples}")

    random_generator = random.Random(RANDOM_SEED)

    sources = [
        ("sp_COCO", SpCOCOEntrySource(root)),
        ("cm_COCO", CmCOCOEntrySource(root)),
        ("bcm_COCO", BcmCOCOEntrySource(root)),
        ("bcmc_COCO", BcmcCOCOEntrySource(root)),
    ]

    total_entries = 0
    total_failed = 0

    for name, source in sources:
        total_entries += len(source)
        total_failed += validate_source(
            name=name,
            source=source,
            sample_count=args.samples,
            random_generator=random_generator,
        )

    print()
    print("=" * 60)
    print(f"total entries: {total_entries}")
    print(f"failed random checks: {total_failed}")

    if total_failed:
        raise RuntimeError(
            f"tampCOCO validation failed with "
            f"{total_failed} missing sampled entries"
        )

    print("tampCOCO entry validation passed.")


if __name__ == "__main__":
    main()