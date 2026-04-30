from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset
    from . import dataloader
    from . import types

from ..._internal.lazy import attach_lazy_modules

attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "dataset": ".dataset",
        "dataloader": ".dataloader",
        "types": ".types",
    },
)