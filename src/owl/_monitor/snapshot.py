from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TypeAlias


MetricValue: TypeAlias = str | int | float | bool | None
Metrics: TypeAlias = dict[str, MetricValue]


@dataclass(slots=True)
class MonitorSnapshot:
    """训练监控快照。

    Snapshot 只描述一次训练状态更新，不负责序列化，也不感知 gRPC。

    Attributes:
        epoch: 当前 epoch。
        step: 当前 step。
        model_metrics: 模型侧提交的轻量级指标。
        loss_metrics: 损失函数侧提交的轻量级指标。
        name: 快照名称，默认为 train。
        seq: 全局递增序号，由 MonitorRing 分配。
        timestamp: 快照创建时间戳。
    """

    epoch: int
    step: int
    model_metrics: Metrics = field(default_factory=dict)
    loss_metrics: Metrics = field(default_factory=dict)
    name: str = "train"

    seq: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_train_step(
        cls,
        *,
        epoch: int,
        step: int,
        model_metrics: Metrics | None = None,
        loss_metrics: Metrics | None = None,
    ) -> MonitorSnapshot:
        """从训练 step 指标创建监控快照。

        Args:
            epoch: 当前 epoch。
            step: 当前 step。
            model_metrics: 模型侧指标。
            loss_metrics: 损失函数侧指标。

        Returns:
            训练监控快照。
        """
        return cls(
            epoch=epoch,
            step=step,
            model_metrics=model_metrics or {},
            loss_metrics=loss_metrics or {},
        )

    def to_dict(self) -> dict:
        """返回快照的原始字典表示。"""
        return asdict(self)