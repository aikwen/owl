"""Training data invocation configuration and resolver.

This module defines the declarative configuration used to construct the
training data pipeline and the resolver that materializes that configuration.

Training data may contain multiple logical dataset sources. Each declaration
is resolved into an independent ``OwlDataset``. Multiple datasets are combined
in declaration order with ``ConcatDataset`` and exposed through one training
``DataLoader``:

    data declarations
    -> OwlDataset objects
    -> combined training dataset
    -> one DataLoader
"""

from collections.abc import Sequence
from dataclasses import dataclass

from torch.utils.data import ConcatDataset, DataLoader

from .base import _DataConfig, _build_dataset
from .types import DataDeclaration


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainData(_DataConfig):
    """Configuration describing the training data pipeline.

    Every source declaration describes one logical training dataset. A
    declaration may contain only a dataset root:

        "datasets/casia_v2"

    In that form, the inherited ``entry`` field supplies the entry-source
    constructor.

    A declaration may instead provide a local entry-source override:

        (
            "datasets/custom_train",
            CustomEntrySource,
        )

    The local entry-source constructor takes precedence over the default
    constructor stored in ``entry``.

    All resolved datasets share the configured augmentation pipeline and sample
    hook. DataLoader options apply to the final combined training dataset rather
    than to each source independently.

    Attributes:
        sources:
            Ordered training dataset declarations. The supplied sequence is
            converted into a tuple during initialization so later mutations to
            the caller-owned sequence do not affect the configuration.
    """

    sources: Sequence[DataDeclaration]

    def __post_init__(self) -> None:
        """Copy shared options and freeze the source collection shape."""

        _DataConfig.__post_init__(self)

        object.__setattr__(
            self,
            "sources",
            tuple(self.sources),
        )


def resolve_train_data(config: TrainData) -> DataLoader:
    """Resolve a training data configuration into one dataloader.

    Every source declaration is independently resolved into an ``OwlDataset``.
    When multiple sources are declared, their datasets are concatenated in
    declaration order. A single source is passed directly to ``DataLoader`` so
    an unnecessary ``ConcatDataset`` wrapper is avoided.

    Args:
        config:
            Declarative training data configuration.

    Returns:
        DataLoader constructed from the resolved training dataset.

    Raises:
        ValueError:
            If no training sources are declared.
    """

    if not config.sources:
        raise ValueError(
            "training data must declare at least one dataset source"
        )

    datasets = [
        _build_dataset(
            declaration,
            default_entry=config.entry,
            augment=config.augment,
            hook=config.hook,
        )
        for declaration in config.sources
    ]

    dataset = (
        datasets[0]
        if len(datasets) == 1
        else ConcatDataset(datasets)
    )

    return DataLoader(
        dataset,
        **dict(config.loader),
    )


__all__ = [
    "TrainData",
    "resolve_train_data",
]