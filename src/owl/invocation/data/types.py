"""Type definitions for data invocation configurations.

This module defines the declarative types used by owl data invocation objects.

A data invocation does not load resources or construct datasets directly.
Instead, it records enough information for the invocation resolver to build the
data pipeline in three distinct stages:

    dataset root
    -> EntrySource
    -> OwlDataset
    -> DataLoader

Keeping these stages separate preserves the responsibilities of the existing
data layer:

- ``EntrySource`` describes stable source-level sample facts.
- ``OwlDataset`` loads resources and applies transforms and sample hooks.
- ``DataLoader`` controls batching and iteration behavior.

The aliases in this module describe configuration values only. They do not
perform path normalization, file-system validation, entry-source construction,
or dataloader argument validation.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

from ...data.entry import EntrySource


PathLike: TypeAlias = str | Path
"""Path representation accepted by data invocation declarations.

String paths are preserved while the invocation configuration is being
declared. The data resolver converts each value to ``Path`` immediately before
constructing the configured entry source.

The invocation layer intentionally does not normalize paths eagerly because
configuration construction should not depend on the current working directory
or access the file system.
"""


EntrySourceType: TypeAlias = Callable[[Path], EntrySource]
"""Callable that constructs an entry source from a dataset root.

The value is normally an entry-source class whose constructor accepts one
normalized ``Path`` and returns an instance satisfying the ``EntrySource``
structural protocol:

    entry_source = entry_source_type(root)

For example, the following class satisfies this alias:

    class CustomEntrySource:
        def __init__(self, root: Path) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, index: int) -> SampleEntry:
            ...

The class object itself is callable:

    entry_source_type = CustomEntrySource
    entry_source = entry_source_type(Path("datasets/custom"))

The alias describes the required construction signature rather than the
concrete inheritance hierarchy. Entry-source implementations therefore do not
need to inherit from ``EntrySource`` explicitly.
"""


DataDeclaration: TypeAlias = (
    PathLike
    | tuple[PathLike, EntrySourceType]
)
"""Declaration of one logical dataset source.

A declaration supports two forms.

The compact form contains only a dataset root:

    "datasets/casia_v1"

The owning data configuration supplies the default entry-source type for this
form.

The override form contains a dataset root and a local entry-source type:

    (
        "datasets/custom",
        CustomEntrySource,
    )

The local type takes precedence over the default entry-source type configured
at the data-object level.

Entry sources are expected to accept one normalized dataset root as their only
resolver-injected construction argument in v0.0.2.
"""


LoaderOptions: TypeAlias = Mapping[str, Any]
"""Keyword arguments forwarded to ``torch.utils.data.DataLoader``.

The invocation layer stores dataloader options as a generic mapping instead of
duplicating the complete PyTorch ``DataLoader`` constructor in an owl-specific
configuration class.

Configuration objects copy this mapping during initialization so later
top-level mutations to the caller-owned mapping do not change the stored
configuration. Data resolvers create their own ``dict`` before forwarding the
options to ``DataLoader``.

Compatibility between individual options, such as ``shuffle`` and ``sampler``,
remains the responsibility of PyTorch's ``DataLoader`` implementation.
"""


__all__ = [
    "DataDeclaration",
    "EntrySourceType",
    "LoaderOptions",
    "PathLike",
]
