from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum


class MonitorTransport(str, Enum):
    """监控服务传输方式。"""

    AUTO = "auto"
    UNIX = "unix"
    HTTP = "http"


@dataclass(slots=True)
class MonitorConfig:
    """训练监控运行配置。

    monitor 只暴露训练状态，不参与训练控制路径。

    Attributes:
        enabled: 是否启动监控服务。
        buffer_size: 内存中保留的最近快照数量。
        transport: 监控服务传输方式。auto 在 POSIX 系统使用 UDS，
            在 Windows 使用 HTTP。
        unix_socket_name: Unix Domain Socket 文件名，最终位于运行目录下。
        http_host: HTTP 监听地址。
        http_port: HTTP 监听端口。
    """

    enabled: bool = False
    buffer_size: int = 50
    transport: MonitorTransport | str = MonitorTransport.AUTO

    unix_socket_name: str = ".owl-monitor.sock"

    http_host: str = "127.0.0.1"
    http_port: int = 8765

    def normalized_transport(self) -> MonitorTransport:
        """返回当前平台下最终使用的传输方式。

        Returns:
            解析后的监控服务传输方式。
        """
        transport = MonitorTransport(self.transport)

        if transport != MonitorTransport.AUTO:
            return transport

        if platform.system().lower() == "windows":
            return MonitorTransport.HTTP

        return MonitorTransport.UNIX