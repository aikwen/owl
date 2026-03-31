from . import schemas
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .context import OwlTrainContext


def standard_train_step(
        batch_data: schemas.DataSetBatch,
        ctx: "OwlTrainContext",
        current_epoch: int,
        current_batch: int
) -> float:
    """标准的单步训练逻辑。

    Args:
        batch_data (dict): DataLoader 产出的一个 Batch 数据。
        ctx (OwlTrainContext): 训练上下文。
        current_epoch (int): 当前 Epoch 索引。
        current_batch (int): 当前 Batch 索引。

    Returns:
        float: 本次 Step 的 Loss 值。
    """
    model = ctx.model
    criterion = ctx.criterion
    optimizer = ctx.optimizer
    device = ctx.main.device

    # 数据搬运
    inputs = batch_data['tp_tensor'].to(device)
    gt = batch_data['gt_tensor'].to(device)

    # 前向传播
    outputs = model(inputs)

    # 构建 Loss 上下文
    bundle = schemas.CriterionBundle(
        current_epoch=current_epoch,
        current_batch=current_batch,
        model_output=outputs,
        gt=gt,
        inputs=inputs
    )

    # 计算 Loss
    loss = criterion(bundle)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 学习率调度器更新 (Batch Level)
    if ctx.scheduler:
        ctx.scheduler.step()

    return loss.item()
