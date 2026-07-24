from __future__ import annotations

import threading
from collections import deque
from typing import Any

from .snapshot import MonitorSnapshot


DEFAULT_MONITOR_BUFFER_SIZE = 25


class MonitorRing:
    """训练监控快照环形缓存。

    MonitorRing 只负责保存最近 N 条快照，并为快照分配全局递增 seq。
    """

    def __init__(self) -> None:
        self._seq = 0
        self._items: deque[MonitorSnapshot] = deque(
            maxlen=DEFAULT_MONITOR_BUFFER_SIZE,
        )
        self._cond = threading.Condition()

    @property
    def seq(self) -> int:
        """返回当前最新快照序号。"""
        with self._cond:
            return self._seq

    @property
    def buffer_size(self) -> int:
        """返回环形缓存容量。"""
        return DEFAULT_MONITOR_BUFFER_SIZE

    def append(self, snapshot: MonitorSnapshot) -> MonitorSnapshot:
        """追加一条快照。

        append 会为 snapshot 分配新的 seq，并唤醒等待新快照的客户端。

        Args:
            snapshot: 待追加的训练监控快照。

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
        """返回当前环形缓存状态。

        Returns:
            状态摘要字典：
                seq: 当前最新快照序号。
                buffer_size: 最大缓存容量。
                cached: 当前已缓存的快照数量。
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

        该方法主要用于 gRPC 流式接口：
        客户端传入自己已经收到的最新 seq，ring 会等待新的快照写入，
        然后返回当前缓存窗口中所有 seq 大于 last_seq 的快照。

        Args:
            last_seq: 客户端已经收到的最新快照序号。
            timeout: 等待超时时间，单位为秒。
                - None: 一直等待，直到有新的快照写入。
                - 大于等于 0 的浮点数: 最多等待指定秒数。

        Returns:
            seq 大于 last_seq 的快照列表。

            如果等待期间有新快照写入，返回新快照列表。
            如果超时后仍然没有新快照，返回空列表。

        注意:
            由于 ring 只保存最近固定数量的快照，如果客户端的 last_seq 太旧，
            中间部分旧快照可能已经被覆盖，此时只会返回当前 ring 中仍然存在的较新快照。
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