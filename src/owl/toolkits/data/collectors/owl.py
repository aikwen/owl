import pathlib
from typing import List
from ..types import DataRecord
from ...common import fs


def from_owl_json(root: pathlib.Path | str, json_name: str | None = None) -> List[DataRecord]:
    """Owl 协议数据集解析器。

        解析 root 目录下的索引 JSON 文件并返回绝对路径列表。

        目录结构示例::

            root/
            ├── gt/           # Ground Truth 图像
            ├── tp/           # 篡改图像
            └── root.json     # 索引文件 (必须与文件夹同名)

        json 文件内容::

            [
              {
                "tp": "tampered_image_01.jpg",
                "gt": "mask_01.png"
              },
              {
                "tp": "tampered_image_02.jpg",
                "gt": "mask_02.png"
              }
            ]

        Args:
            root: 数据集根目录路径。
            json_name: 指定 JSON 索引文件名。如果为 None，则默认为 "目录名.json"。

        Returns:
            符合 DataRecord 契约的绝对路径列表。

        Raises:
            FileNotFoundError: 当指定的 JSON 索引文件不存在时。
        """
    # 如果没传，默认找 root/root.json
    root_path = pathlib.Path(root).resolve()
    name = json_name or f"{root_path.name}.json"
    json_path = root_path / name

    if not json_path.exists():
        raise FileNotFoundError(f"Owl protocol JSON not found at: {json_path}")

    raw_data = fs.load_json(json_path)

    records: List[DataRecord] = []
    for item in raw_data:
        records.append({
            "tp": str((root_path / "tp" / item["tp"]).resolve()),
            "gt": str((root_path / "gt" / item["gt"]).resolve()) if item.get("gt") else ""
        })
    return records