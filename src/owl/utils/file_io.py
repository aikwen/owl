import json
from pathlib import Path
from typing import Union, Any
from PIL import Image


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

