from typing import TYPE_CHECKING

from .._internal.lazy import attach_lazy_modules


if TYPE_CHECKING:
    from . import client


__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "client": ".client",
    },
)