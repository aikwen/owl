from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass

import grpc

from .codec import encode_metrics
from .generated import monitor_pb2, monitor_pb2_grpc
from .ring import MonitorRing
from .snapshot import MonitorSnapshot


MONITOR_HOST = "127.0.0.1"
MONITOR_PORT = 0
STREAM_WAIT_TIMEOUT = 1.0


@dataclass(slots=True)
class MonitorServerHandle:
    """监控服务句柄。

    Attributes:
        address: gRPC 监听地址，例如 "127.0.0.1:39125"。
        server: gRPC server 实例，用于后续关闭服务。
    """

    address: str
    server: grpc.Server


class MonitorService(monitor_pb2_grpc.MonitorServiceServicer):
    """Owl 训练监控 gRPC 服务。"""

    def __init__(self, ring: MonitorRing) -> None:
        self._ring = ring

    def Health(
        self,
        request,
        context: grpc.ServicerContext,
    ):
        """返回当前监控状态。"""
        stats = self._ring.stats()

        return monitor_pb2.HealthResponse(
            ok=True,
            seq=stats["seq"],
            buffer_size=stats["buffer_size"],
            cached=stats["cached"],
        )

    def GetWindow(
        self,
        request,
        context: grpc.ServicerContext,
    ):
        """返回当前 ring 中缓存的快照。"""
        snapshots = [
            _snapshot_to_proto(snapshot)
            for snapshot in self._ring.window()
        ]

        return monitor_pb2.GetWindowResponse(
            snapshots=snapshots,
        )

    def StreamSnapshots(
        self,
        request,
        context: grpc.ServicerContext,
    ):
        """持续返回 seq 大于 last_seq 的快照。"""
        last_seq = int(request.last_seq)

        # 先补发当前 ring 中仍然存在的新快照。
        for snapshot in self._ring.window():
            if snapshot.seq > last_seq:
                last_seq = snapshot.seq
                yield _snapshot_to_proto(snapshot)

        # 再等待后续新快照。
        while context.is_active():
            snapshots = self._ring.wait_next(
                last_seq=last_seq,
                timeout=STREAM_WAIT_TIMEOUT,
            )

            for snapshot in snapshots:
                last_seq = snapshot.seq
                yield _snapshot_to_proto(snapshot)


def start_monitor_server(ring: MonitorRing) -> MonitorServerHandle:
    """启动本地 gRPC 监控服务。

    服务固定监听 127.0.0.1，端口由系统自动分配。

    Args:
        ring: 训练监控快照环形缓存。

    Returns:
        监控服务句柄。
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
    )

    monitor_pb2_grpc.add_MonitorServiceServicer_to_server(
        MonitorService(ring),
        server,
    )

    bind_address = f"{MONITOR_HOST}:{MONITOR_PORT}"
    actual_port = server.add_insecure_port(bind_address)

    if actual_port == 0:
        raise RuntimeError("failed to bind monitor gRPC server")

    server.start()

    return MonitorServerHandle(
        address=f"{MONITOR_HOST}:{actual_port}",
        server=server,
    )


def stop_monitor_server(
    handle: MonitorServerHandle | None,
    *,
    grace: float = 1.0,
) -> None:
    """停止 gRPC 监控服务。

    Args:
        handle: start_monitor_server 返回的服务句柄。
        grace: 优雅关闭等待时间，单位为秒。
    """
    if handle is None:
        return

    handle.server.stop(grace=grace)


def _snapshot_to_proto(snapshot: MonitorSnapshot):
    """将 MonitorSnapshot 转换为 gRPC Snapshot。"""
    return monitor_pb2.Snapshot(
        seq=snapshot.seq,
        epoch=snapshot.epoch,
        step=snapshot.step,
        timestamp=snapshot.timestamp,
        name=snapshot.name,
        model_metrics=encode_metrics(snapshot.model_metrics),
        loss_metrics=encode_metrics(snapshot.loss_metrics),
    )