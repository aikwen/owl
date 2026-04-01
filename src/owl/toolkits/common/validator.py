import json
from pathlib import Path
from typing import Union, Any, List


def data_protocol(path: Union[str, Path]) -> Path:
    """验证数据集是否严格符合规定的文件结构协议。

    **校验规则列表**::

        1. 根目录必须存在。
        2. 必须包含 'gt' 和 'tp' 两个子文件夹。
        3. 必须包含与目录同名的 .json 描述文件。

    期望的文件结构示例::

        dataset_root_name/
        ├── gt/                 # 掩码图文件夹
        ├── tp/                 # 篡改图文件夹
        └── dataset_root_name.json  # 必须与根目录同名的元数据文件

    Args:
        path: 数据集根目录路径。可以是字符串路径或 pathlib.Path 对象。

    Returns:
        Path: 校验通过后的绝对路径对象 (Resolved Path)。

    Raises:
        TypeError: 当传入的 path 类型不是 str 或 Path 时抛出。
        FileNotFoundError:
        ValueError: 当 JSON 文件存在但无法被解析（格式错误）时抛出。
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
