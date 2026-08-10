"""Training data invocation configuration and resolver.

This module defines the declarative configuration used to construct the
training data pipeline and the resolver that materializes that configuration.

Training data may contain one or multiple logical dataset sources. Each
declaration is resolved into an independent ``OwlDataset``. Multiple datasets
are combined in declaration order with ``ConcatDataset`` and exposed through
one training ``DataLoader``:

    data declarations
    -> OwlDataset objects
    -> combined training dataset
    -> one DataLoader

A single dataset may be declared directly:

    TrainData(
        sources="datasets/casia_v2",
    )

A dataset with a local entry-source override may also be declared directly:

    TrainData(
        sources=(
            "datasets/custom_train",
            CustomEntrySource,
        ),
    )

Multiple datasets must be declared with a list:

    TrainData(
        sources=[
            "datasets/casia_v2",
            "datasets/coverage",
            (
                "datasets/custom_train",
                CustomEntrySource,
            ),
        ],
    )

Using a list exclusively for multiple declarations keeps the syntax
unambiguous: tuples represent one declaration with a local entry-source
constructor, while lists represent multiple declarations.
"""

from dataclasses import dataclass, field

from torch.utils.data import ConcatDataset, DataLoader

from .base import _DataConfig, _build_dataset
from .types import DataDeclaration, SamplerDeclaration


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainData(_DataConfig):
    """Configuration describing the training data pipeline.

    One dataset may be supplied directly as a ``DataDeclaration``.

    A compact declaration contains only a dataset root:

        sources="datasets/casia_v2"

    In that form, the inherited ``default_entry_source`` constructor is used.

    A declaration may instead provide a local entry-source constructor:

        sources=(
            "datasets/custom_train",
            CustomEntrySource,
        )

    The local constructor takes precedence over
    ``default_entry_source`` for that declaration.

    Multiple datasets must be supplied as a list:

        sources=[
            "datasets/casia_v2",
            "datasets/coverage",
            (
                "datasets/custom_train",
                CustomEntrySource,
            ),
        ]

    All resolved datasets share the configured augmentation pipeline and sample
    hook. DataLoader options apply to the final combined training dataset rather
    than to each source independently.

    Attributes:
        sources:
            One training dataset declaration or a list of declarations.

            A direct ``DataDeclaration`` represents one dataset. A list
            represents multiple datasets in declaration order.

            The declarations are normalized into an immutable tuple during
            initialization so later mutations to a caller-owned list do not
            affect the configuration.

        sampler:
            Optional declaration describing how to construct the training sampler.

            A sampler constructor may be supplied directly when it requires only the
            resolved training dataset:

                sampler=CustomSampler

            Sampler-specific keyword arguments may also be supplied with the
            constructor:

                sampler=(
                    CustomSampler,
                    {
                        "seed": 42,
                    },
                )

            The sampler is constructed by the training data resolver after the final
            training dataset has been resolved. The resolved training dataset is
            supplied as the first constructor argument.
    """

    sources: DataDeclaration | list[DataDeclaration] = field(
        default_factory=list,
    )
    sampler: SamplerDeclaration | None = None

    def __post_init__(self) -> None:
        """Copy mutable options and normalize source declarations."""

        _DataConfig.__post_init__(self)

        sources = (
            tuple(self.sources)
            if isinstance(self.sources, list)
            else (self.sources,)
        )

        object.__setattr__(
            self,
            "sources",
            sources,
        )

        if isinstance(self.sampler, tuple):
            constructor, options = self.sampler

            object.__setattr__(
                self,
                "sampler",
                (
                    constructor,
                    dict(options),
                ),
            )


def resolve_train_data(config: TrainData) -> DataLoader:
    """Resolve a training data configuration into one dataloader.

    Every source declaration is independently resolved into an ``OwlDataset``.
    When multiple sources are declared, their datasets are concatenated in
    declaration order. A single source is passed directly to ``DataLoader`` so
    an unnecessary ``ConcatDataset`` wrapper is avoided.

    When a sampler declaration is configured, the sampler is constructed after
    the final training dataset has been resolved and is then supplied to the
    ``DataLoader``. A configured sampler cannot be combined with
    ``shuffle=True``.

    Args:
        config:
            Declarative training data configuration.

    Returns:
        DataLoader constructed from the resolved training dataset.

    Raises:
        ValueError:
            If no training sources are declared, or if a training sampler is
            configured together with ``shuffle=True``.
    """
    if not config.sources:
        raise ValueError(
            "training data must declare at least one dataset source"
        )

    datasets = [
        _build_dataset(
            declaration,
            default_entry=config.default_entry_source,
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

    loader_options = dict(config.loader)

    if config.sampler is not None:
        if loader_options.get("shuffle", False):
            raise ValueError(
                "training sampler cannot be used with "
                "loader option shuffle=True"
            )

        if isinstance(config.sampler, tuple):
            constructor, options = config.sampler
            sampler = constructor(
                dataset,
                **options,
            )
        else:
            sampler = config.sampler(dataset)

        loader_options["sampler"] = sampler

    return DataLoader(
        dataset,
        **loader_options,
    )


__all__ = [
    "TrainData",
    "resolve_train_data",
]