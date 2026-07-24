"""Checkpoint loading declarations and resolution.

This module defines how an owl invocation requests checkpoint restoration and
provides the resolver that applies serialized snapshot to resolved runtime
components.

Checkpoint loading and checkpoint saving are intentionally represented by
different concerns.

``CheckpointLoad`` describes how previously saved snapshot should be restored
before runtime execution begins. Checkpoint saving describes when and where the
current training snapshot should be written and belongs to the execution domain.

Two loading modes are supported.

Model-only loading restores model parameters without changing optimizer,
scheduler, or training progress snapshot. It is suitable for pretrained weights,
transfer learning, and fine-tuning:

    checkpoint=CheckpointLoad(
        path="checkpoints/pretrained.pth",
        model_only=True,
    )

Model-only loading accepts either an Owl checkpoint v1 dictionary or a bare
model snapshot dictionary.

Full-snapshot loading restores an Owl checkpoint v1 containing:

- the completed epoch represented by the checkpoint;
- model snapshot;
- optimizer snapshot;
- scheduler snapshot.

It is intended for resuming an interrupted training run:

    checkpoint=CheckpointLoad(
        path="checkpoints/latest.pth",
        model_only=False,
    )

During ``owl.invoke()``, ``resolve_checkpoint()`` applies the requested snapshot
after the runtime components have been resolved. For full-snapshot loading, it
returns the zero-based completed epoch stored in the checkpoint. The invocation
pipeline derives the effective resume position as:

    start_epoch = checkpoint_epoch + 1

Checkpoint tensors are loaded onto CPU by default. A different deserialization
device may be selected through the ``map_location`` argument passed to
``resolve_checkpoint()``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...schemas.checkpoint import OwlCheckpointV1
from ..data.types import PathLike


_CHECKPOINT_V1_KEYS = frozenset(
    {
        "format_version",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
    }
)
"""Required top-level keys of an Owl checkpoint v1 dictionary."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointLoad:
    """Configuration describing how checkpoint snapshot should be restored.

    This object records the checkpoint source and loading policy. File access,
    format validation, device mapping, and component snapshot restoration are
    deferred until ``resolve_checkpoint()`` is called.

    Attributes:
        path:
            Path to one concrete checkpoint file.

            Directory scanning, aliases such as ``latest``, and checkpoint
            filename conventions are outside this configuration's scope.

        model_only:
            Whether only model parameters should be restored.

            When ``True``, an Owl checkpoint v1 or bare model snapshot dictionary
            is accepted. Optimizer, scheduler, and training progress remain
            unchanged.

            When ``False``, the file must contain a complete Owl checkpoint v1.
            Model, optimizer, scheduler, and completed epoch snapshot are restored.

        strict:
            Whether model snapshot keys must exactly match the current model snapshot.

            This value is forwarded to
            ``torch.nn.Module.load_state_dict(..., strict=strict)``. It affects
            model loading only; optimizer and scheduler restoration retain their
            native PyTorch behavior.
    """

    path: PathLike
    model_only: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        """Normalize the checkpoint path without accessing the file system."""

        object.__setattr__(
            self,
            "path",
            Path(self.path),
        )


def resolve_checkpoint(
    declaration: CheckpointLoad | None,
    *,
    model: Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> int | None:
    """Resolve and apply a checkpoint loading declaration.

    ``None`` leaves all supplied components unchanged.

    Model-only loading restores model parameters from either a complete Owl
    checkpoint v1 or a bare model snapshot dictionary. It does not restore or
    return training progress.

    Full-snapshot loading requires a complete Owl checkpoint v1. The model,
    optimizer, and scheduler states are restored, and the zero-based completed
    epoch represented by the checkpoint is returned.

    Checkpoint tensors are deserialized onto ``map_location`` before their
    snapshot dictionaries are applied. The default CPU mapping is independent of
    the devices currently used by the supplied components. PyTorch copies model
    snapshot values into the model's existing parameters and buffers during
    ``load_state_dict()``.

    Deserialization errors and native PyTorch snapshot restoration errors are
    allowed to propagate unchanged.

    Args:
        declaration:
            Checkpoint loading configuration, or ``None`` when no checkpoint
            should be restored.
        model:
            Resolved model that receives the checkpoint model snapshot.
        optimizer:
            Resolved optimizer that receives the checkpoint optimizer snapshot.
            Required for full-snapshot loading.
        scheduler:
            Resolved scheduler that receives the checkpoint scheduler snapshot.
            Required for full-snapshot loading.
        map_location:
            Device onto which checkpoint tensors are deserialized by
            ``torch.load()``. Defaults to ``"cpu"``.

    Returns:
        The zero-based completed epoch stored in a full Owl checkpoint v1, or
        ``None`` when no checkpoint is loaded or model-only loading is used.

    Raises:
        TypeError:
            If the loaded object or one of its required fields has an invalid
            type, or if full-snapshot loading is requested without an optimizer or
            scheduler.
        ValueError:
            If the checkpoint format version is unsupported, the completed
            epoch is negative, or required Owl checkpoint v1 fields are absent.
    """
    if declaration is None:
        return None

    loaded = torch.load(
        declaration.path,
        map_location=map_location,
        weights_only=True,
    )

    if declaration.model_only:
        model_state = _resolve_model_state(loaded)

        model.load_state_dict(
            model_state,
            strict=declaration.strict,
        )

        return None

    if optimizer is None:
        raise TypeError(
            "optimizer is required when loading a full checkpoint"
        )

    if scheduler is None:
        raise TypeError(
            "scheduler is required when loading a full checkpoint"
        )

    checkpoint = _validate_checkpoint_v1(loaded)

    model.load_state_dict(
        checkpoint["model"],
        strict=declaration.strict,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["epoch"]


def resolve_model_checkpoint(
    declaration: CheckpointLoad | None,
    *,
    model: Module,
    map_location: str | torch.device = "cpu",
) -> None:
    """Restore model parameters from an optional checkpoint declaration.

    This resolver is intended for workflows that only own a model, such as
    standalone inference. It accepts the shared ``CheckpointLoad`` declaration
    but intentionally ignores its ``model_only`` field.

    Both complete Owl checkpoint v1 dictionaries and bare model state
    dictionaries are supported. For a complete Owl checkpoint, only the model
    snapshot is restored; optimizer, scheduler, and epoch state are ignored.

    Args:
        declaration:
            Optional checkpoint loading declaration.

        model:
            Resolved model that receives the checkpoint parameters.

        map_location:
            Device used while deserializing checkpoint tensors.
    """
    if declaration is None:
        return None

    loaded = torch.load(
        declaration.path,
        map_location=map_location,
        weights_only=True,
    )

    model_state = _resolve_model_state(loaded)

    model.load_state_dict(
        model_state,
        strict=declaration.strict,
    )

    return None

def _resolve_model_state(loaded: Any) -> Mapping[str, Any]:
    """Extract model snapshot from an Owl checkpoint or bare snapshot dictionary.

    Args:
        loaded:
            Object returned by ``torch.load()``.

    Returns:
        Mapping suitable for ``torch.nn.Module.load_state_dict()``.

    Raises:
        TypeError:
            If the loaded object is neither an Owl checkpoint containing model
            snapshot nor a bare mapping.
    """
    if not isinstance(loaded, Mapping):
        raise TypeError(
            "model checkpoint must contain a mapping"
        )

    if _looks_like_checkpoint_v1(loaded):
        model_state = loaded["model"]

        if not isinstance(model_state, Mapping):
            raise TypeError(
                "checkpoint model snapshot must be a mapping"
            )

        return model_state

    return loaded


def _validate_checkpoint_v1(loaded: Any) -> OwlCheckpointV1:
    """Validate an object against the Owl checkpoint v1 schema.

    Args:
        loaded:
            Object returned by ``torch.load()``.

    Returns:
        Validated checkpoint dictionary.

    Raises:
        TypeError:
            If the checkpoint or one of its snapshot fields has an invalid type.
        ValueError:
            If required fields are absent, the format version is unsupported,
            or the completed epoch is negative.
    """
    if not isinstance(loaded, Mapping):
        raise TypeError(
            "full checkpoint must contain a mapping"
        )

    missing_keys = _CHECKPOINT_V1_KEYS.difference(loaded)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))

        raise ValueError(
            f"checkpoint is missing required fields: {missing}"
        )

    format_version = loaded["format_version"]

    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise TypeError(
            "checkpoint format_version must be an integer"
        )

    if format_version != 1:
        raise ValueError(
            f"unsupported checkpoint format version: {format_version}"
        )

    epoch = loaded["epoch"]

    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError(
            "checkpoint epoch must be an integer"
        )

    if epoch < 0:
        raise ValueError(
            "checkpoint epoch must not be negative"
        )

    for field in ("model", "optimizer", "scheduler"):
        if not isinstance(loaded[field], Mapping):
            raise TypeError(
                f"checkpoint {field} snapshot must be a mapping"
            )

    return cast(OwlCheckpointV1, loaded)


def _looks_like_checkpoint_v1(value: Mapping[Any, Any]) -> bool:
    """Return whether a mapping identifies itself as an Owl checkpoint v1."""

    return (
        value.get("format_version") == 1
        and "model" in value
    )


__all__ = [
    "CheckpointLoad",
    "resolve_checkpoint",
    "resolve_model_checkpoint",
]