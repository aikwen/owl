"""Owl checkpoint v1 schema definitions.

This module defines the first version of the checkpoint dictionary written and
consumed by owl training workflows.

An Owl checkpoint v1 contains the snapshot of the model, optimizer, and scheduler
at the end of one completed epoch:

    {
        "format_version": 1,
        "epoch": 4,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

The ``epoch`` field is the zero-based index of the completed epoch represented
by the stored component states. It describes which epoch the checkpoint belongs
to rather than where a resumed training run should begin.

A training invocation derives its resume position from this value:

    start_epoch = checkpoint["epoch"] + 1

Checkpoint v1 supports epoch-boundary recovery only. It does not preserve
partial progress within an epoch, batch indexes, or global step counters.

This schema describes a plain dictionary suitable for ``torch.save()``. Runtime
validation, snapshot restoration, and checkpoint saving are handled by separate
invocation and execution components.
"""

from typing import Any, Literal, TypedDict


class OwlCheckpointV1(TypedDict):
    """Dictionary schema for an Owl checkpoint v1.

    Attributes:
        format_version:
            Checkpoint schema version. Version 1 checkpoints must contain the
            literal value ``1``.

        epoch:
            Zero-based index of the last fully completed epoch represented by
            the stored states.

        model:
            State dictionary returned by ``torch.nn.Module.state_dict()``.

        optimizer:
            State dictionary returned by
            ``torch.optim.Optimizer.state_dict()``.

        scheduler:
            State dictionary returned by
            ``torch.optim.lr_scheduler.LRScheduler.state_dict()``.

    Notes:
        All fields are required.

        ``TypedDict`` provides static typing only. Values matching this schema
        remain ordinary dictionaries at runtime and can be passed directly to
        ``torch.save()``.
    """

    format_version: Literal[1]
    epoch: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


__all__ = [
    "OwlCheckpointV1",
]