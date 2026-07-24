from torch import optim
from . import BATCH_SCHEDULERS


@BATCH_SCHEDULERS.register(name="poly")
def poly(optimizer: optim.Optimizer, power: float, epochs: int, batches: int) -> optim.lr_scheduler.LRScheduler:
    """初始化多项式衰减学习率调度器"""
    total_iters = max(1, epochs * batches)
    return optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        total_iters=total_iters,
        power=power
    )
