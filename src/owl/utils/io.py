import json
from pathlib import Path
from typing import Union, Any
from PIL import Image
import logging
import sys
import yaml
import pathlib
import torch


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


def load_yaml(path: Union[str, Path]) -> Any:
    """加载并解析指定路径的 YAML 文件。

    此函数会处理常见的文件读取错误。如果文件不存在、YAML 格式错误或读取失败，
    程序将打印错误信息并直接退出 (sys.exit)。

    Args:
        path (Union[str, Path]): YAML 文件的路径。

    Returns:
        Any: 解析后的 YAML 数据（通常是字典或列表）。
             如果文件内容为空，将打印警告并返回 None。
    """
    path = Path(path)
    if not path.exists():
        print(f"[错误] 找不到文件: {path}")
        print("   请检查文件名拼写，或使用 'owl init' 生成模板。")
        sys.exit(1)

    # 读取 yaml 文件
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            print("[警告] 文件是空的！")

        return config_data

    except yaml.YAMLError as e:
        print(f"[错误] YAML 格式不正确:")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 读取文件失败: {e}")
        sys.exit(1)


def save_checkpoint(state_dict: dict[str, Any], save_dir: str | pathlib.Path, filename: str):
    """保存状态字典到文件。

    Args:
        state_dict (Dict[str, Any]): 要保存的数据（ asdict(schemas.Checkpoint) 转成的字典）。
        save_dir (str | pathlib.Path): 保存目录。
        filename (str): 文件名。
    """
    save_path = pathlib.Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, save_path)


def load_checkpoint(file_path: str | pathlib.Path, map_location="cpu") -> dict[str, Any]:
    """加载权重文件并返回字典。

    Args:
        file_path (str | pathlib.Path): 权重文件路径。
        map_location: 设备映射 (如 "cuda:0", "cpu")。

    Returns:
        Dict[str, Any]: 加载后的原始字典数据。
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    # weights_only=False 以支持加载复杂的嵌套结构（如 dataclass 转出的 dict）
    return torch.load(file_path, map_location=map_location, weights_only=False)


def create_dir(path: Union[str, Path]) -> None:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Error: 无法创建输出目录。Error: {e}")


def get_logger(
        log_file: str,
        mode: str,
        is_format: bool = True,
        level: int = logging.INFO,
) -> logging.Logger:
    """
    获取某个文件的 logger 句柄，如果不存在就会创建
    :param log_file:
    :param mode: 日志等级（同一 log_file + mode 组合仅第一次调用真正影响 handler）
    :param is_format: 只有第一次调用生效
    :param level:
    :return:
    """
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

        if is_format:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            # “没有 format”：只输出 message 本身
            formatter = logging.Formatter(fmt="%(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger
