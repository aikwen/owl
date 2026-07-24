"""Manual dataset visualization test.

Expected owl v1 dataset layout:

    D:/sample/
    ├── sample.json
    ├── tp/
    ├── gt/
    └── edge/        # optional

Run from the repository root:

    python tests/manual/test_dataset_visual.py
"""

import matplotlib

matplotlib.use("TkAgg")

from pathlib import Path

from owl.data.augment import train
from owl.data.dataset import OwlDataset, visualize_dataset
from owl.data.entry import OwlV1EntrySource


SAMPLE_ROOT = Path("D:/example")
IMAGE_SIZE = (512, 512)


def main() -> None:
    source = OwlV1EntrySource(SAMPLE_ROOT)

    dataset = OwlDataset(
        source=source,
        transform=train(IMAGE_SIZE),
    )

    print(f"dataset size: {len(dataset)}")
    print(f"has edge: {dataset.has_edge}")

    visualize_dataset(dataset)


if __name__ == "__main__":
    main()