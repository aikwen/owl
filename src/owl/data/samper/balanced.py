"""Balanced sampling across concatenated datasets."""

from collections.abc import Iterator

import torch
from torch.utils.data import ConcatDataset, Sampler


class BalancedSampler(Sampler[int]):
    """Sample each child dataset with an equal per-iteration quota.

    The sampler operates on a ``ConcatDataset`` and yields global indices
    understood by that dataset. Each child dataset contributes exactly
    ``samples_per_dataset`` samples during one iteration.

    Sampling is performed with replacement at the sample level. The same local
    sample may therefore be selected multiple times within one iteration,
    regardless of whether the child dataset is larger or smaller than the
    configured quota.

    Args:
        dataset:
            Concatenated dataset whose child datasets should be sampled
            independently.

        samples_per_dataset:
            Number of samples contributed by each child dataset during one
            iteration.

        seed:
            Optional seed used to initialize an internal ``torch.Generator``.
            This argument is ignored when ``generator`` is provided.

        generator:
            Optional random generator used for sample selection and ordering.
            When provided, it takes precedence over ``seed``. When neither
            ``generator`` nor ``seed`` is provided, PyTorch's default random
            generator is used.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        samples_per_dataset: int = 1800,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if samples_per_dataset <= 0:
            raise ValueError(
                "samples_per_dataset must be positive, "
                f"got {samples_per_dataset}"
            )

        self.dataset = dataset
        self.samples_per_dataset = samples_per_dataset

        if generator is not None:
            self.generator = generator
        elif seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

        self._lengths = [
            len(child_dataset)
            for child_dataset in dataset.datasets
        ]

        if any(length == 0 for length in self._lengths):
            raise ValueError(
                "all child datasets in ConcatDataset must contain "
                "at least one sample"
            )

        self._offsets = [
            0,
            *dataset.cumulative_sizes[:-1],
        ]

        self._num_datasets = len(self._lengths)

    def __iter__(self) -> Iterator[int]:
        """Yield one balanced sequence of global dataset indices."""

        indices = torch.cat(
            [
                torch.randint(
                    high=length,
                    size=(self.samples_per_dataset,),
                    generator=self.generator,
                )
                + offset
                for length, offset in zip(
                    self._lengths,
                    self._offsets,
                    strict=True,
                )
            ]
        )

        permutation = torch.randperm(
            len(indices),
            generator=self.generator,
        )

        yield from indices[permutation].tolist()

    def __len__(self) -> int:
        """Return the number of samples produced per iteration."""

        return (
            self._num_datasets
            * self.samples_per_dataset
        )


__all__ = [
    "BalancedSampler",
]