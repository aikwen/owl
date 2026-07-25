"""Optimizer component declaration and resolution types.

This module defines the optimizer forms accepted by owl invocations and
provides the resolver that constructs the final optimizer instance.

Optimizer construction follows the two-stage factory protocol defined in
``owl.schemas.optim``.

The outer callable receives user-defined configuration values, such as the
learning rate and weight decay, and returns an ``OptimizerFactory``:

    optimizer_factory = create_optimizer(
        lr=1e-4,
        weight_decay=1e-2,
    )

The returned factory receives the resolved model and constructs the optimizer:

    optimizer = optimizer_factory(
        model=model,
    )

An invocation may therefore receive either an already configured
``OptimizerFactory`` or an outer builder together with the keyword arguments
required to create that factory.

An already configured factory may be supplied directly:

    optimizer=optimizer_factory

An outer builder may be paired with its user-defined options:

    optimizer=(
        create_optimizer,
        {
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
    )

During ``owl.invoke()``, ``resolve_optimizer()`` converts either form into a
``torch.optim.Optimizer`` instance. The resolved model is supplied explicitly
because optimizer construction depends on the final model instance selected by
the invocation.
"""

from typing import Any, Mapping, Protocol, TypeAlias

from torch.nn import Module
from torch.optim import Optimizer

from ...schemas.optim import OptimizerConstructor


class OptimizerFactoryBuilder(Protocol):
    """Protocol for callables that create optimizer factories.

    The builder represents the outer stage of optimizer construction. It
    receives arbitrary user-defined keyword arguments and returns an
    ``OptimizerFactory``.

    The keyword signature is intentionally unrestricted because different
    optimizer implementations may expose different configuration options.

    The returned factory follows the stable owl optimizer protocol and receives
    the resolved model during ``resolve_optimizer()``.
    """

    def __call__(self, **kwargs: Any) -> OptimizerConstructor:
        """Create and return an optimizer factory.

        Args:
            **kwargs: User-defined optimizer configuration values. Typical
                options include the learning rate, weight decay, momentum,
                epsilon, and parameter-group-specific settings.

        Returns:
            Factory that constructs an optimizer for a resolved model.
        """
        ...


OptimizerArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to an optimizer factory builder.

The mapping is expanded when ``resolve_optimizer()`` resolves a configured
builder declaration:

    optimizer_factory = builder(**dict(optimizer_args))

A generic mapping is required because the outer builder signature is defined by
the optimizer implementation rather than by owl.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored declaration.
"""


OptimizerDeclaration: TypeAlias = (
        OptimizerConstructor
        | tuple[OptimizerFactoryBuilder, OptimizerArgs]
)
"""Optimizer construction specification accepted by an owl invocation.

The direct form contains an already configured ``OptimizerFactory``:

    optimizer_factory

This form is used when the outer configuration function has already been
called:

    optimizer_factory = adamw(
        lr=1e-4,
        weight_decay=1e-2,
    )

The configured-builder form contains an outer callable and the keyword
arguments used to produce the factory:

    (
        create_optimizer,
        {
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
    )

The configured-builder form is resolved in two stages:

    optimizer_factory = create_optimizer(
        **optimizer_args,
    )

    optimizer = optimizer_factory(
        model=model,
    )

Concrete ``Optimizer`` instances are deliberately excluded. The optimizer must
be constructed after the final model instance has been resolved so its
parameter groups reference the model used by the training session.
"""


def resolve_optimizer(
    declaration: OptimizerDeclaration,
    *,
    model: Module,
) -> Optimizer:
    """Resolve an optimizer declaration for a model.

    An already configured optimizer factory is invoked directly with the
    supplied model. A configured-builder declaration is first expanded into an
    optimizer factory and then invoked with that model.

    Exceptions raised by user-defined builders and factories are allowed to
    propagate unchanged so callers retain the original exception type and
    traceback.

    Args:
        declaration: Optimizer factory or configured factory-builder
            declaration.
        model: Resolved model whose parameters will be optimized.

    Returns:
        Optimizer constructed for the supplied model.

    Raises:
        TypeError: If the declaration does not match a supported optimizer form,
            if a builder does not return a callable factory, or if the factory
            does not return a ``torch.optim.Optimizer`` instance.
    """
    if isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "optimizer builder declaration must contain exactly "
                "a builder and its keyword arguments"
            )

        builder, optimizer_args = declaration

        if not callable(builder) or not isinstance(optimizer_args, Mapping):
            raise TypeError(
                "optimizer builder declaration must contain a callable "
                "builder and a mapping of keyword arguments"
            )

        optimizer_factory = builder(**dict(optimizer_args))
    else:
        optimizer_factory = declaration

    if not callable(optimizer_factory):
        raise TypeError(
            "optimizer declaration must resolve to a callable "
            "OptimizerFactory"
        )

    optimizer = optimizer_factory(model=model)

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            "optimizer factory must return a torch.optim.Optimizer instance"
        )

    return optimizer


__all__ = [
    "OptimizerArgs",
    "OptimizerDeclaration",
    "OptimizerFactoryBuilder",
    "resolve_optimizer",
]
