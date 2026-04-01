from typing import TypedDict, Any

class CheckpointDict(TypedDict):
    """定义保存到 .pth 文件中的权重字典结构。

    Attributes:
        epoch (int): 训练结束时的轮次索引。
        model_state (dict): 模型的 state_dict。
        optimizer_state (dict): 优化器的 state_dict。
        scheduler_state (Optional[dict]): 学习率调度器的 state_dict。
    """
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None