"""Scheduler component declaration and resolution types.

This module defines the learning-rate scheduler forms accepted by owl
invocations and provides the resolver that constructs the final scheduler
instance.

Scheduler construction follows the two-stage factory protocol defined in
``owl.schemas.optim``.

The outer callable receives user-defined scheduler options and returns a
``SchedulerFactory``:

    scheduler_factory = create_scheduler(
        warmup_steps=1000,
        min_lr_ratio=0.01,
    )

The returned factory receives the resolved optimizer and training plan:

    scheduler = scheduler_factory(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

An invocation may receive one of three scheduler forms:

- an already configured ``SchedulerFactory``;
- an outer builder paired with the keyword arguments used to create a factory;
- ``None``, requesting owl's built-in constant learning-rate scheduler.

An already configured factory may be supplied directly:

    scheduler=scheduler_factory

An outer builder may be paired with user-defined options:

    scheduler=(
        create_scheduler,
        {
            "warmup_steps": 1000,
            "min_lr_ratio": 0.01,
        },
    )

The absence of an explicit scheduling strategy may be expressed with ``None``:

    scheduler=None

During ``owl.invoke()``, ``resolve_scheduler()`` converts each form into a
``torch.optim.lr_scheduler.LRScheduler`` instance. The resolved optimizer and
training-plan values are supplied explicitly because scheduler construction
depends on resources produced earlier in the invocation pipeline.
"""

from typing import Any, Mapping, Protocol, TypeAlias

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...optim.scheduler import constant
from ...schemas.optim import SchedulerFactory


class SchedulerFactoryBuilder(Protocol):
    """Protocol for callables that create scheduler factories.

    The builder represents the outer stage of scheduler construction. It
    receives arbitrary user-defined keyword arguments and returns a
    ``SchedulerFactory``.

    The outer signature is intentionally unrestricted because scheduling
    strategies may expose different configuration values, such as warmup
    duration, polynomial power, minimum learning-rate ratio, milestones, or
    decay factors.

    The returned factory follows the stable owl scheduler protocol and receives
    the resolved optimizer and training plan during ``resolve_scheduler()``.
    """

    def __call__(self, **kwargs: Any) -> SchedulerFactory:
        """Create and return a scheduler factory.

        Args:
            **kwargs: User-defined scheduler configuration values accepted by
                the concrete builder.

        Returns:
            Factory that constructs a learning-rate scheduler for a resolved
            optimizer and training plan.
        """
        ...


SchedulerArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to a scheduler factory builder.

The mapping is expanded when ``resolve_scheduler()`` resolves a configured
builder declaration:

    scheduler_factory = builder(**dict(scheduler_args))

A generic mapping is used because the outer builder signature belongs to the
scheduler implementation rather than to owl.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored declaration.
"""


SchedulerDeclaration: TypeAlias = (
    SchedulerFactory
    | tuple[SchedulerFactoryBuilder, SchedulerArgs]
    | None
)
"""Scheduler construction specification accepted by an owl invocation.

The direct form contains an already configured ``SchedulerFactory``:

    scheduler_factory

This form is used when the outer configuration function has already been
called:

    scheduler_factory = poly(
        power=0.9,
    )

The configured-builder form contains an outer callable and the keyword
arguments used to produce the scheduler factory:

    (
        create_scheduler,
        {
            "warmup_steps": 1000,
            "min_lr_ratio": 0.01,
        },
    )

The configured-builder form is resolved in two stages:

    scheduler_factory = create_scheduler(
        **scheduler_args,
    )

    scheduler = scheduler_factory(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

The ``None`` form requests the built-in constant learning-rate strategy:

    scheduler_factory = constant()

That factory is invoked through the same scheduler protocol, keeping the
training session structure uniform when no explicit learning-rate decay
strategy is configured.

Concrete ``LRScheduler`` instances are deliberately excluded. Scheduler
construction depends on the final optimizer and resolved training plan, so it
must occur after those resources are available.
"""


def resolve_scheduler(
    declaration: SchedulerDeclaration,
    *,
    optimizer: Optimizer,
    total_epochs: int,
    total_steps: int,
) -> LRScheduler:
    """Resolve a scheduler declaration for an optimizer and training plan.

    An already configured scheduler factory is invoked directly. A configured
    builder declaration is first expanded into a scheduler factory. ``None`` is
    resolved to owl's built-in constant scheduler factory.

    The resulting factory receives the supplied optimizer, total epoch count,
    and total optimizer-step count.

    Exceptions raised by user-defined builders and factories are allowed to
    propagate unchanged so callers retain the original exception type and
    traceback.

    Args:
        declaration: Scheduler factory, configured factory-builder declaration,
            or ``None`` for the built-in constant scheduler.
        optimizer: Resolved optimizer whose learning rate will be scheduled.
        total_epochs: Total number of epochs in the training plan.
        total_steps: Total number of planned optimizer updates.

    Returns:
        Learning-rate scheduler constructed for the supplied optimizer and
        training plan.

    Raises:
        TypeError: If the declaration does not match a supported scheduler
            form, if a builder does not return a callable factory, or if the
            factory does not return an ``LRScheduler`` instance.
    """
    if declaration is None:
        scheduler_factory = constant()
    elif isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "scheduler builder declaration must contain exactly "
                "a builder and its keyword arguments"
            )

        builder, scheduler_args = declaration

        if not callable(builder) or not isinstance(scheduler_args, Mapping):
            raise TypeError(
                "scheduler builder declaration must contain a callable "
                "builder and a mapping of keyword arguments"
            )

        scheduler_factory = builder(**dict(scheduler_args))
    else:
        scheduler_factory = declaration

    if not callable(scheduler_factory):
        raise TypeError(
            "scheduler declaration must resolve to a callable "
            "SchedulerFactory"
        )

    scheduler = scheduler_factory(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            "scheduler factory must return a "
            "torch.optim.lr_scheduler.LRScheduler instance"
        )

    return scheduler


__all__ = [
    "SchedulerArgs",
    "SchedulerDeclaration",
    "SchedulerFactoryBuilder",
    "resolve_scheduler",
]
