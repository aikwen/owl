from __future__ import annotations

import argparse
import time
from typing import Any

import grpc

from ..monitor import client as monitor_client


def func(args: argparse.Namespace) -> None:
    """连接 Owl monitor server 并实时打印训练状态。"""
    address = args.address
    interval = max(0.0, float(getattr(args, "interval", 1.0)))
    retry = max(0, int(getattr(args, "retry", 5)))

    try:
        health = _connect_with_retry(address, retry=retry)
    except KeyboardInterrupt:
        print("\n[monitor] stopped by user")
        return

    if health is None:
        return

    print(
        f"[monitor] connected to {address} | "
        f"cached={health.get('cached')}/{health.get('buffer_size')}"
    )

    try:
        snapshots = monitor_client.window(address)
    except KeyboardInterrupt:
        print("\n[monitor] stopped by user")
        return
    except grpc.RpcError as exc:
        print(f"[fail] 读取 monitor window 失败: {exc.details()}")
        return
    except Exception as exc:
        print(f"[fail] 读取 monitor window 失败: {exc}")
        return

    last_seq = 0

    if snapshots:
        print("\n[window]")
        for snapshot in snapshots:
            print(_format_snapshot(snapshot))
        last_seq = int(snapshots[-1]["seq"])
    else:
        print("\n[window] 当前没有缓存快照")

    print(f"\n[stream] interval={interval:g}s")

    last_print_time = 0.0

    try:
        for snapshot in monitor_client.stream(address, last_seq=last_seq):
            last_seq = int(snapshot["seq"])

            if interval <= 0:
                print(_format_snapshot(snapshot))
                continue

            now = time.monotonic()
            if now - last_print_time >= interval:
                print(_format_snapshot(snapshot))
                last_print_time = now

    except KeyboardInterrupt:
        print("\n[monitor] stopped by user")
    except grpc.RpcError as exc:
        print(f"\n[monitor] 连接已断开: {exc.details()}")
    except Exception as exc:
        print(f"\n[monitor] 读取失败: {exc}")


def _connect_with_retry(
    address: str,
    *,
    retry: int,
) -> dict[str, Any] | None:
    """连接 monitor server，失败后每秒重试一次。"""
    retry = max(0, retry)

    for attempt in range(retry + 1):
        try:
            return monitor_client.health(address)

        except KeyboardInterrupt:
            raise

        except grpc.RpcError as exc:
            if attempt >= retry:
                print(f"[fail] 无法连接 monitor server: {address}")
                print(f"       {exc.details()}")
                return None

            print(
                f"[monitor] waiting for server {address} "
                f"(retry {attempt + 1}/{retry})..."
            )

            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                raise

        except Exception as exc:
            if attempt >= retry:
                print(f"[fail] 无法连接 monitor server: {address}")
                print(f"       {exc}")
                return None

            print(
                f"[monitor] waiting for server {address} "
                f"(retry {attempt + 1}/{retry})..."
            )

            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                raise

    return None


def _format_snapshot(snapshot: dict[str, Any]) -> str:
    """格式化单条监控快照。"""
    epoch = snapshot.get("epoch")
    step = snapshot.get("step")
    model_metrics = _format_metrics(snapshot.get("model_metrics", {}))
    loss_metrics = _format_metrics(snapshot.get("loss_metrics", {}))

    return (
        f"epoch={epoch} step={step} "
        f"model={{{model_metrics}}} "
        f"loss={{{loss_metrics}}}"
    )


def _format_metrics(metrics: dict[str, Any]) -> str:
    """格式化指标字典。"""
    if not metrics:
        return ""

    return " ".join(
        f"{key}={_format_metric_value(value)}"
        for key, value in metrics.items()
    )


def _format_metric_value(value: Any) -> str:
    """格式化单个指标值。"""
    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value

        if "." in value or "e" in value.lower():
            return f"{number:.4f}"

    return str(value)