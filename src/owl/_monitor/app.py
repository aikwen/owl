from __future__ import annotations
import json
from collections.abc import Iterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .snapshot import MonitorSnapshot
from .state import MonitorState


def create_monitor_app(state: MonitorState) -> FastAPI:
    """创建训练监控 HTTP 应用。

    app 只负责暴露监控接口，不负责启动服务，也不关心具体传输方式。

    Args:
        state: 训练监控运行状态。

    Returns:
        FastAPI 应用实例。
    """
    app = FastAPI(
        title="Owl Monitor",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health")
    def health() -> dict:
        """返回监控服务状态。

        Returns:
            监控服务状态摘要。
        """
        return {
            "ok": True,
            "state": state.stats(),
        }

    @app.get("/stream")
    def stream() -> StreamingResponse:
        """返回训练状态流。

        新连接会先收到当前缓存窗口中的快照，随后持续等待新的快照。

        Returns:
            NDJSON 流式响应。
        """
        return StreamingResponse(
            _snapshot_stream(state),
            media_type="application/x-ndjson",
        )

    return app


def _snapshot_stream(state: MonitorState) -> Iterator[str]:
    """生成 NDJSON 快照流。

    Args:
        state: 训练监控运行状态。

    Yields:
        每行一个 JSON 对象。
    """
    last_seq = 0

    for snapshot in state.window():
        last_seq = snapshot.seq
        yield _encode_snapshot(snapshot)

    while True:
        snapshots = state.wait_next(last_seq)

        for snapshot in snapshots:
            last_seq = snapshot.seq
            yield _encode_snapshot(snapshot)


def _encode_snapshot(snapshot: MonitorSnapshot) -> str:
    """将快照编码为 NDJSON 行。

    Args:
        snapshot: 训练监控快照。

    Returns:
        以换行符结尾的 JSON 字符串。
    """
    return json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"