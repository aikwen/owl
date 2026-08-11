"""tampCOCO dataset entry sources.

This module converts the official tampCOCO list files into indexed
collections of ``SampleEntry`` objects.

The tampCOCO dataset contains four logical training sources:

- ``sp_COCO``
- ``cm_COCO``
- ``bcm_COCO``
- ``bcmc_COCO``

All four sources share the same dataset root and use one comma-separated list
file to describe image and ground-truth mask pairs.

Example dataset layout:

    tampCOCO/
    ├── sp_images/
    ├── sp_masks/
    ├── sp_COCO_list.txt
    ├── cm_images/
    ├── cm_masks/
    ├── cm_COCO_list.txt
    ├── bcm_images/
    ├── bcm_masks/
    ├── bcm_COCO_list.txt
    ├── bcmc_images/
    └── bcmc_COCO_list.txt

Each list file contains one sample per line:

    sp_images/sample.jpg,sp_masks/sample.png

The ``bcmc_COCO`` source reuses masks from ``bcm_masks``:

    bcmc_images/sample.jpg,bcm_masks/sample.png

Paths stored in the official list files are relative to the tampCOCO dataset
root and are preserved as dataset facts.

These sources only interpret dataset metadata. They do not load images,
inspect resource contents, infer labels, generate masks, or derive edge
supervision.
"""

from pathlib import Path

from .entry import SampleEntry


class _TampCOCOEntrySource:
    """Base indexed entry source for one tampCOCO logical subset.

    The official list file is read once during initialization and converted
    into an immutable tuple of ``SampleEntry`` objects. Entry order is
    identical to the order of lines in the list file.

    Args:
        root:
            tampCOCO dataset root directory.

        manifest_name:
            Official list filename describing the logical subset.

    Raises:
        FileNotFoundError:
            If the dataset root or list file does not exist.
        NotADirectoryError:
            If the dataset root is not a directory.
        ValueError:
            If a list line is malformed or contains invalid paths.
    """

    def __init__(
        self,
        root: str | Path,
        manifest_name: str,
    ) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"dataset root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"dataset root is not a directory: {self.root}"
            )

        self.manifest_path = self.root / manifest_name
        self._entries = self._load_entries()

    def __len__(self) -> int:
        """Return the number of sample entries."""
        return len(self._entries)

    def __getitem__(self, index: int) -> SampleEntry:
        """Return the sample entry at the given index."""
        return self._entries[index]

    def _load_entries(self) -> tuple[SampleEntry, ...]:
        """Read and convert all lines from the tampCOCO list file."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"tampCOCO list file does not exist: {self.manifest_path}"
            )

        with self.manifest_path.open("r", encoding="utf-8") as file:
            return tuple(
                self._parse_entry(
                    line=line,
                    index=index,
                )
                for index, line in enumerate(file)
            )

    def _parse_entry(
        self,
        *,
        line: str,
        index: int,
    ) -> SampleEntry:
        """Convert one tampCOCO list line into a ``SampleEntry``."""
        line = line.strip()

        if not line:
            raise ValueError(
                f"tampCOCO list line {index} must not be empty"
            )

        fields = line.split(",")

        if len(fields) != 2:
            raise ValueError(
                f"tampCOCO list line {index} must contain exactly "
                f"two comma-separated paths, got {len(fields)}"
            )

        tp = self._parse_path(
            value=fields[0],
            field="tp",
            index=index,
        )
        gt = self._parse_path(
            value=fields[1],
            field="gt",
            index=index,
        )

        return SampleEntry(
            tp=tp,
            gt=gt,
            label=None,
            edge=None,
        )

    def _parse_path(
        self,
        *,
        value: str,
        field: str,
        index: int,
    ) -> Path:
        """Resolve one tampCOCO relative resource path."""
        value = value.strip()

        if not value:
            raise ValueError(
                f"tampCOCO list line {index} field {field!r} must be "
                "a non-empty path"
            )

        relative_path = Path(value)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"tampCOCO list line {index} field {field!r} must be "
                f"a safe relative path, got {value!r}"
            )

        return self.root / relative_path


class SpCOCOEntrySource(_TampCOCOEntrySource):
    """Indexed sample entries loaded from ``sp_COCO_list.txt``.

    Args:
        root:
            tampCOCO dataset root directory.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(
            root=root,
            manifest_name="sp_COCO_list.txt",
        )


class CmCOCOEntrySource(_TampCOCOEntrySource):
    """Indexed sample entries loaded from ``cm_COCO_list.txt``.

    Args:
        root:
            tampCOCO dataset root directory.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(
            root=root,
            manifest_name="cm_COCO_list.txt",
        )


class BcmCOCOEntrySource(_TampCOCOEntrySource):
    """Indexed sample entries loaded from ``bcm_COCO_list.txt``.

    Args:
        root:
            tampCOCO dataset root directory.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(
            root=root,
            manifest_name="bcm_COCO_list.txt",
        )


class BcmcCOCOEntrySource(_TampCOCOEntrySource):
    """Indexed sample entries loaded from ``bcmc_COCO_list.txt``.

    ``bcmc_COCO`` contains JPEG-compressed manipulated images and reuses
    ground-truth masks stored under ``bcm_masks``. The official list file
    already records those mask paths, so no special path handling is needed.

    Args:
        root:
            tampCOCO dataset root directory.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(
            root=root,
            manifest_name="bcmc_COCO_list.txt",
        )


__all__ = [
    "BcmCOCOEntrySource",
    "BcmcCOCOEntrySource",
    "CmCOCOEntrySource",
    "SpCOCOEntrySource",
]