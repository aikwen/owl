"""Learning-rate scheduler factory protocol definitions.

This module defines the protocol used by owl clients to construct learning-rate
schedulers. A scheduler factory is created during configuration and invoked
later after the optimizer and training plan have become available.
"""

from typing import Protocol

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SchedulerFactory(Protocol):
    """Protocol implemented by learning-rate scheduler factories.

    Scheduler construction is divided into two stages.

    The outer function receives user-defined options such as the warmup
    duration and minimum learning-rate ratio. It captures those options and
    returns a scheduler factory.

    The owl client later invokes that factory with the instantiated optimizer
    and the complete training plan. The returned scheduler is then stored in
    the training session.

    Example:
        Define an outer configuration function:

        >>> import math
        >>>
        >>> from torch.optim import Optimizer
        >>> from torch.optim.lr_scheduler import LambdaLR, LRScheduler
        >>>
        >>> def create_cosine_scheduler(
        ...     *,
        ...     min_lr_ratio: float = 0.0,
        ... ) -> SchedulerFactory:
        ...     def factory(
        ...         *,
        ...         optimizer: Optimizer,
        ...         total_epochs: int,
        ...         total_steps: int,
        ...     ) -> LRScheduler:
        ...         def lr_lambda(step: int) -> float:
        ...             progress = min(step / total_steps, 1.0)
        ...             cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ...             return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        ...
        ...         return LambdaLR(optimizer, lr_lambda)
        ...
        ...     return factory

        Configure the scheduler before the optimizer is available:

        >>> scheduler_factory = create_cosine_scheduler(
        ...     min_lr_ratio=0.01,
        ... )

        The owl client later injects the optimizer and training plan:

        >>> scheduler = scheduler_factory(
        ...     optimizer=optimizer,
        ...     total_epochs=100,
        ...     total_steps=50000,
        ... )

    Notes:
        This protocol constrains only the inner factory. The signature of the
        outer configuration function is intentionally unrestricted because
        different scheduler implementations require different options.

        A scheduler factory may ignore training-plan values that are not needed
        by its scheduling strategy.
    """

    def __call__(
        self,
        *,
        optimizer: Optimizer,
        total_epochs: int,
        total_steps: int,
    ) -> LRScheduler:
        """Create a scheduler for the injected optimizer and training plan.

        Args:
            optimizer:
                Instantiated optimizer whose learning rate will be scheduled.
            total_epochs:
                Total number of epochs configured for the training run.
            total_steps:
                Total number of planned optimizer updates across the complete
                training run.

        Returns:
            Learning-rate scheduler associated with the supplied optimizer.
        """
        ...