from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .sanitize import to_jsonable


@dataclass(slots=True)
class MonitorSnapshot:
    """训练监控快照。

    snapshot 只描述一次训练状态更新，不约束 payload 的内部结构。

    Attributes:
        epoch: 当前 epoch。
        step: 当前 step。
        payload: 训练侧提交的状态数据。
        seq: 全局递增序号，由 MonitorState 分配。
        timestamp: 快照创建时间戳。
        name: 快照名称，默认为 train。
    """

    epoch: int
    step: int
    payload: dict[str, Any] = field(default_factory=dict)

    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    name: str = "train"

    @classmethod
    def from_train_step(
        cls,
        *,
        epoch: int,
        step: int,
        step_result: dict[str, Any] | None,
    ) -> MonitorSnapshot:
        """从训练 step 结果创建监控快照。

        Args:
            epoch: 当前 epoch。
            step: 当前 step。
            step_result: 训练 step 返回的状态数据。

        Returns:
            训练监控快照。
        """
        return cls(
            epoch=epoch,
            step=step,
            payload=to_jsonable(step_result or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 安全的字典表示。

        Returns:
            JSON 可序列化字典。
        """
        return to_jsonable(asdict(self))