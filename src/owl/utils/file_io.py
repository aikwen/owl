import json
from pathlib import Path
from typing import Union, Any, Optional
from PIL import Image
import  logging

def load_json(path: Union[str, Path]) -> Any:
    """
    读取 JSON 文件。
    Args:
        path: 文件路径
    Returns:
        List or Dict: 解析后的 JSON 数据
    Raises:
        RuntimeError: 如果文件读取或解析失败
    """
    p = Path(path)
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # 抛出更具体的错误信息，方便调试
        raise RuntimeError(f"无法读取 JSON 文件: {p} -> {e}")



def load_image(path: Union[str, Path]) -> Image.Image:
    """
    读取单张图片。

    Args:
        path: 图片路径

    Returns:
        PIL.Image.Image: 图像对象

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 图像损坏或无法解码
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"图像不存在: {p}")

    try:
        img = Image.open(p)
        # 强制加载到内存，避免 Lazy Loading 导致的 "Too many open files" 问题
        img.load()
        return img
    except Exception as e:
        raise RuntimeError(f"图像读取失败: {p.name} -> {e}")

def create_dir(path: Union[str, Path]) -> None:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error: 无法创建输出目录。Error: {e}")

def create_logger(
                log_file: str,
                mode: str,
                level: int = logging.INFO,
                ) -> logging.Logger:
    p = Path(f"{log_file}_{mode}.log")
    logger = logging.getLogger(str(p.resolve()))
    logger.setLevel(level)
    # 不向根 logger 冒泡，避免重复输出到控制台
    logger.propagate = False
    # 避免重复添加 handler
    if not logger.handlers:
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger

