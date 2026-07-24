from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import grpc

from .codec import decode_metrics
from .generated import monitor_pb2, monitor_pb2_grpc


def health_from_address(address: str) -> dict[str, Any]:
    """读取监控服务状态。

    Args:
        address: gRPC 服务地址，例如 "127.0.0.1:39125"。

    Returns:
        监控服务状态字典。
    """
    with grpc.insecure_channel(address) as channel:
        stub = monitor_pb2_grpc.MonitorServiceStub(channel)
        response = stub.Health(monitor_pb2.HealthRequest())

    return {
        "ok": response.ok,
        "seq": response.seq,
        "buffer_size": response.buffer_size,
        "cached": response.cached,
    }


def window_from_address(address: str) -> list[dict[str, Any]]:
    """读取当前 ring 中缓存的快照。

    Args:
        address: gRPC 服务地址，例如 "127.0.0.1:39125"。

    Returns:
        当前缓存窗口中的快照列表，按 seq 升序排列。
    """
    with grpc.insecure_channel(address) as channel:
        stub = monitor_pb2_grpc.MonitorServiceStub(channel)
        response = stub.GetWindow(monitor_pb2.GetWindowRequest())

    return [
        _snapshot_to_dict(snapshot)
        for snapshot in response.snapshots
    ]


def stream_from_address(
    address: str,
    *,
    last_seq: int = 0,
) -> Iterator[dict[str, Any]]:
    """持续读取新的训练监控快照。

    Args:
        address: gRPC 服务地址，例如 "127.0.0.1:39125"。
        last_seq: 客户端已经收到的最新快照序号。
            只会读取 seq 大于 last_seq 的快照。

    Yields:
        新的训练监控快照字典。
    """
    with grpc.insecure_channel(address) as channel:
        stub = monitor_pb2_grpc.MonitorServiceStub(channel)

        response_iter = stub.StreamSnapshots(
            monitor_pb2.StreamSnapshotsRequest(
                last_seq=last_seq,
            )
        )

        for snapshot in response_iter:
            yield _snapshot_to_dict(snapshot)


def _snapshot_to_dict(snapshot) -> dict[str, Any]:
    """将 gRPC Snapshot 转换为普通字典。"""
    return {
        "seq": snapshot.seq,
        "epoch": snapshot.epoch,
        "step": snapshot.step,
        "timestamp": snapshot.timestamp,
        "name": snapshot.name,
        "model_metrics": decode_metrics(snapshot.model_metrics),
        "loss_metrics": decode_metrics(snapshot.loss_metrics),
    }