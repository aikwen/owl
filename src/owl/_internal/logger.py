import sys
import pathlib
from loguru import logger as _logger

LOGO_TEXT = r"""
                 __
  ____ _      __/ /
 / __ \ | /| / / /
/ /_/ / |/ |/ / /
\____/|__/|__/_/ owl(v{})
"""

class OwlLogger:
    """全局日志管理器"""
    # 单例
    _initialized = False

    @classmethod
    def setup(cls, work_dir: str | pathlib.Path):
        """
        初始化日志
        """
        if cls._initialized:
            return

        work_dir = pathlib.Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 移除 loguru 默认的终端输出
        _logger.remove()

        # ==========================================
        # 终端控制台输出 (Console)
        # ==========================================
        _logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <level>{message}</level>",
            level="INFO",   # 只打印 INFO 及以上级别的日志
            colorize=True,  # 开启颜色
            filter=lambda record: "mode" not in record["extra"],
            enqueue=True    # 开启异步/多线程安全
        )

        # ==========================================
        # 训练日志文件 (train.log)
        # ==========================================
        train_log_path = work_dir.joinpath("train.log")
        _logger.add(
            str(train_log_path),
            # 纯文本 format
            format="{time:YYYY-MM-DD HH:mm:ss} - {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("mode", "train") == "train",  # 默认接收 train 模式的日志
            rotation="100 MB",  # 文件超过 100MB 自动打包
            retention="30 days",  # 日志保留 30 天
            enqueue=True
        )

        # ==========================================
        # 验证日志文件 (validate.log)
        # ==========================================
        val_log_path = work_dir.joinpath("validate.log")
        _logger.add(
            str(val_log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} - {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("mode") == "val",  # 只有被标记为 val 的日志才会存到这里
            rotation="100 MB",
            retention="30 days",
            enqueue=True
        )

        # 标记为已初始化
        cls._initialized = True
        _logger.info(f"日志目录: {work_dir}")

    @classmethod
    def welcome(cls):
        """打印欢迎词与 Logo"""
        try:
            from ... import __version__
        except ImportError:
            __version__ = "unknown"

        logo = LOGO_TEXT.format(__version__)

        _logger.opt(raw=True, colors=True).info(f"<cyan>{logo}</cyan>\n")
        _logger.opt(colors=True).info("<cyan>owl engine</cyan> is starting...")

    @classmethod
    def stop(cls):
        """打印结束语"""
        if not cls._initialized:
            return

        _logger.opt(colors=True).info("<cyan>owl engine</cyan> is stopped!")
        _logger.complete()

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized
logger = _logger