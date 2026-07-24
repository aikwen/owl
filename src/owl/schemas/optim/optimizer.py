"""Optimizer factory protocol definitions.

This module defines the protocol used by owl clients to construct optimizers.
An optimizer factory is created during configuration and invoked later after
the model has been instantiated.
"""

from typing import Protocol

from torch.nn import Module
from torch.optim import Optimizer


class OptimizerFactory(Protocol):
    """Protocol implemented by optimizer factories.

    Optimizer construction is divided into two stages.

    The outer function receives user-defined options such as the learning rate
    and weight decay. It captures those options and returns an optimizer
    factory.

    The owl client later invokes that factory with the instantiated model and
    receives the optimizer used by the training session.

    Example:
        Define an outer configuration function:

        >>> from torch.nn import Module
        >>> from torch.optim import AdamW, Optimizer
        >>>
        >>> def create_adamw(
        ...     *,
        ...     lr: float = 1e-4,
        ...     weight_decay: float = 1e-2,
        ... ) -> OptimizerFactory:
        ...     def factory(*, model: Module) -> Optimizer:
        ...         return AdamW(
        ...             model.parameters(),
        ...             lr=lr,
        ...             weight_decay=weight_decay,
        ...         )
        ...
        ...     return factory

        Configure the optimizer before the model is available:

        >>> optimizer_factory = create_adamw(
        ...     lr=2e-4,
        ...     weight_decay=1e-2,
        ... )

        The owl client later injects the instantiated model:

        >>> optimizer = optimizer_factory(model=model)

    Notes:
        This protocol constrains only the inner factory. The signature of the
        outer configuration function is intentionally unrestricted because
        different optimizer implementations require different options.
    """

    def __call__(self, *, model: Module) -> Optimizer:
        """Create an optimizer for the injected model.

        Args:
            model:
                Instantiated model whose parameters will be optimized.

        Returns:
            Optimizer associated with the supplied model.
        """
        ...