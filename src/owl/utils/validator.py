import json
from pathlib import Path
from typing import Union, Any, List


def data_protocol(path: Union[str, Path]) -> Path:
    """
    data_protocol 验证是否符合数据集格式
    1. 检查文件结构: gt/tp 文件夹和 json 文件
    2. 检查 json 是否被成功解析

    :param path: 数据集根路径
    :return:  pathlib.Path 对象
    """
    # 1. 基础类型与路径检查
    if not isinstance(path, (str, Path)):
        raise TypeError(f"路径必须是 str 或 pathlib.Path 类型，收到: {type(path)}")

    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {p}")

    # 2. 检查核心文件夹结构
    gt_dir = p / "gt"
    tp_dir = p / "tp"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"协议校验失败: 缺失 'gt' 文件夹 -> {gt_dir}")
    if not tp_dir.is_dir():
        raise FileNotFoundError(f"协议校验失败: 缺失 'tp' 文件夹 -> {tp_dir}")

    # 3. 检查 JSON 文件是否存在
    json_file = p / f"{p.name}.json"
    if not json_file.is_file():
        raise FileNotFoundError(f"协议校验失败: 缺失描述文件 -> {json_file}")

    # 4. 检查 JSON 内容有效性
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式解析错误: {json_file.name} -> {e}")
    except Exception as e:
        raise ValueError(f"读取 JSON 文件失败: {json_file.name} -> {e}")

    return p

def check(ok: bool, error_msg:str):
    if not ok:
        raise RuntimeError(error_msg)