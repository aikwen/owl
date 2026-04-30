from __future__ import annotations
import threading
from collections import deque
from typing import Any

from .snapshot import MonitorSnapshot


class MonitorState:
    """训练监控运行状态。

    state 负责维护最近的快照窗口和全局递增序号。不依赖 HTTP
    服务，也不感知具体的传输方式。

    Args:
        buffer_size: 内存中保留的最近快照数量。
    """

    def __init__(self, buffer_size: int = 50):
        self._seq = 0
        self._items: deque[MonitorSnapshot] = deque(maxlen=buffer_size)
        self._cond = threading.Condition()

    @property
    def seq(self) -> int:
        """返回当前最新快照序号。

        Returns:
            当前全局序号。
        """
        with self._cond:
            return self._seq

    @property
    def buffer_size(self) -> int:
        """返回快照缓存容量。

        Returns:
            ring buffer 最大容量。
        """
        return self._items.maxlen or 0

    def update(self, snapshot: MonitorSnapshot) -> MonitorSnapshot:
        """写入新的训练快照。

        update 会为 snapshot 分配新的 seq，并唤醒等待新数据的客户端。

        Args:
            snapshot: 待写入的训练快照。

        Returns:
            写入后的快照。
        """
        with self._cond:
            self._seq += 1
            snapshot.seq = self._seq
            self._items.append(snapshot)
            self._cond.notify_all()
            return snapshot

    def window(self) -> list[MonitorSnapshot]:
        """返回当前缓存窗口。

        Returns:
            最近的快照列表，按 seq 从小到大排序。
        """
        with self._cond:
            return list(self._items)

    def stats(self) -> dict[str, Any]:
        """返回 state 当前状态摘要。

        Returns:
            监控状态摘要。
        """
        with self._cond:
            return {
                "seq": self._seq,
                "buffer_size": self.buffer_size,
                "cached": len(self._items),
            }

    def wait_next(
        self,
        last_seq: int,
        timeout: float | None = None,
    ) -> list[MonitorSnapshot]:
        """等待并返回比 last_seq 更新的快照。

        Args:
            last_seq: 客户端已经收到的最新 seq。
            timeout: 等待超时时间，单位为秒。None 表示一直等待。

        Returns:
            seq 大于 last_seq 的快照列表。
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > last_seq,
                timeout=timeout,
            )

            return [
                item for item in self._items
                if item.seq > last_seq
            ]