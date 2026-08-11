"""Manual visualization for tampCOCO entry sources.

This script randomly selects one sample from each logical tampCOCO source,
materializes the selected entries through ``OwlDataset``, and displays the
result with owl's dataset visualization helper.

The four visualized sources are:

- ``sp_COCO``
- ``cm_COCO``
- ``bcm_COCO``
- ``bcmc_COCO``

This validates the complete path from official tampCOCO metadata to the final
owl dataset item:

    tampCOCO list
    -> EntrySource
    -> SampleEntry
    -> OwlDataset
    -> DatasetItem
    -> visualize_dataset

Example:

    python tests/data/entry/manual/tamp_coco_visual.py \
        "D:/datasets/tampCOCO"
"""

import argparse
import random
from pathlib import Path

from owl.data.dataset import OwlDataset, visualize_dataset
from owl.data.entry import SampleEntry
from owl.data.entry.tamp_coco import (
    BcmCOCOEntrySource,
    BcmcCOCOEntrySource,
    CmCOCOEntrySource,
    SpCOCOEntrySource,
)


RANDOM_SEED = 42


class _SelectedEntrySource:
    """Entry source containing only manually selected samples."""

    def __init__(self, entries: list[SampleEntry]) -> None:
        self._entries = tuple(entries)

    def __len__(self) -> int:
        """Return the number of selected entries."""
        return len(self._entries)

    def __getitem__(self, index: int) -> SampleEntry:
        """Return the selected entry at the given index."""
        return self._entries[index]


def main() -> None:
    """Visualize one random sample from each tampCOCO source."""
    parser = argparse.ArgumentParser(
        description="Visualize random tampCOCO samples through OwlDataset.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the tampCOCO dataset root.",
    )
    args = parser.parse_args()

    root = args.root
    random_generator = random.Random(RANDOM_SEED)

    sources = [
        ("sp_COCO", SpCOCOEntrySource(root)),
        ("cm_COCO", CmCOCOEntrySource(root)),
        ("bcm_COCO", BcmCOCOEntrySource(root)),
        ("bcmc_COCO", BcmcCOCOEntrySource(root)),
    ]

    selected_entries: list[SampleEntry] = []

    print(f"tampCOCO root: {root}")
    print(f"random seed: {RANDOM_SEED}")
    print()

    for name, source in sources:
        index = random_generator.randrange(len(source))
        entry = source[index]

        selected_entries.append(entry)

        print(f"[{name}]")
        print(f"index: {index}")
        print(f"tp: {entry.tp}")
        print(f"gt: {entry.gt}")
        print()

    selected_source = _SelectedEntrySource(selected_entries)
    dataset = OwlDataset(selected_source)

    print(f"visualizing {len(dataset)} samples")
    print("use Next to advance and Exit to close")

    visualize_dataset(dataset)


if __name__ == "__main__":
    main()