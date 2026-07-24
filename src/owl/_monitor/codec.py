from __future__ import annotations

from typing import Any

from .snapshot import MetricValue, Metrics


def encode_metrics(metrics: Metrics | None) -> dict[str, str]:
    """将 Metrics 编码为 gRPC 可传输的 string map。

    Args:
        metrics: 指标字典，key 必须是 str，value 只能是
            str / int / float / bool / None。

    Returns:
        dict[str, str]，可直接写入 proto 的 map<string, string> 字段。

    Raises:
        TypeError: 当指标 key 不是 str，或指标 value 不是支持的基础类型时抛出。
    """
    if metrics is None:
        return {}

    encoded: dict[str, str] = {}

    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(
                f"monitor metric key must be str, got {type(key).__name__}"
            )

        if not _is_metric_value(value):
            raise TypeError(
                f"monitor metric value for {key!r} must be "
                f"str/int/float/bool/None, got {type(value).__name__}"
            )

        encoded[key] = metric_value_to_string(value)

    return encoded


def decode_metrics(metrics: dict[str, str] | None) -> Metrics:
    """将 gRPC 返回的 string map 解码为 Metrics。

    注意:
        gRPC 传输层无法保留原始 Python 类型。
        因此该函数默认保持 value 为 str，不自动猜测 float / int / bool。

    Args:
        metrics: gRPC 返回的 map<string, string>。

    Returns:
        Metrics 字典。
    """
    if metrics is None:
        return {}

    return dict(metrics)


def metric_value_to_string(value: MetricValue) -> str:
    """将单个指标值转换为字符串。

    Args:
        value: 指标值，只允许 str / int / float / bool / None。

    Returns:
        字符串形式的指标值。
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _is_metric_value(value: Any) -> bool:
    """判断 value 是否是合法指标值。"""
    return value is None or isinstance(value, str | int | float | bool)