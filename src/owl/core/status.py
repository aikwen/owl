from dataclasses import dataclass
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader

@dataclass
class Status:
    """
    保存训练过程中所有的动态对象（模型、优化器、进度等），
    并负责管理 Checkpoint 的状态字典。
    """
    # 当前是第几个 epoch
    epoch: int = 0
    # 模型
    model: torch.nn.Module | None = None
    # 优化器
    optimizer: torch.optim.Optimizer | None = None
    # 学习率优化器
    scheduler: Any | None = None
    # 数据集
    train_loader: DataLoader | None = None
    val_loader: Dict[str, DataLoader] | None = None
    # device default cpu
    device: torch.device = torch.device("cpu")

    def state_dict(self) -> Dict[str, Any]:
        """
        生成 Checkpoint 字典
        """
        return {
            "model_state": self.model.state_dict() if self.model else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch
        }

    def load_state_dict(self, checkpoint: Dict[str, Any], only_model: bool = False):
        """
        从 Checkpoint 字典恢复状态
        """
        # 如果需要迁移学习
        if only_model:
            if self.model and "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            return

        # 恢复 Epoch
        epoch = checkpoint.get("epoch", None)
        if epoch is not None:
            self.epoch = epoch + 1
        else:
            self.epoch = 0

        # 恢复组件
        if self.model and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])

        if self.optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
