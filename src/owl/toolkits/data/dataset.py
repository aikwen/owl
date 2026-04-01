import pathlib
from typing import Union, Optional, Callable, List, Dict, Any
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

from .types import DataRecord, DataSetItem
from ..common import fs, image
DataCollectorFunc = Callable[..., List[DataRecord]]


class ImageDataset(Dataset):
    def __init__(self,
                 root_dir: Union[str, pathlib.Path],
                 collector_fn: DataCollectorFunc,
                 collector_kwargs: Dict[str, Any]| None = None,
                 transform: A.Compose|None = None):
        """
        Args:
            root_dir (Union[str, pathlib.Path]): 数据集根目录。
            collector_fn (DataCollectorFunc): 映射函数。
            collector_kwargs (Optional[Dict[str, Any]]): 映射函数参数。
            transform (Optional[A.Compose]): Albumentations 增强流水线。
        """
        self.root_dir = pathlib.Path(root_dir).resolve()
        self.transform = transform

        # 获取所有参数
        kwargs = collector_kwargs or {}
        self.data_records: List[DataRecord] = collector_fn(self.root_dir, **kwargs)

        if not self.data_records:
            raise ValueError(f"Dataset root '{root_dir}' is empty or collector returned nothing.")

    def __len__(self) -> int:
        return len(self.data_records)

    def __getitem__(self, idx: int) -> DataSetItem:
        record = self.data_records[idx]

        # 获取图像路径
        tp_path: pathlib.Path = (
            pathlib.Path(record["tp"]))

        gt_path: pathlib.Path | None = (
            pathlib.Path(record["gt"])) if record.get("gt") else None

        # 图像加载
        tp_img: Image.Image = fs.load_image(tp_path)
        gt_img: Image.Image

        if gt_path is None:
            # 如果没有 gt 路径，生成全黑 Mask
            gt_img = Image.new("L", tp_img.size, 0)
        else:
            gt_img = fs.load_image(gt_path).convert("L")

        # 转 Numpy
        tp_array = image.to_numpy(tp_img, ensure_rgb=True)
        gt_array = image.to_numpy(gt_img, ensure_rgb=False)

        # 数据增强 (Albumentations)
        if self.transform is not None:
            transformed = self.transform(image=tp_array, mask=gt_array)
            tp_array = transformed['image']
            gt_array = transformed['mask']

        # 将 gt 矩阵变成概率矩阵
        gt_array = image.normalize_binary_mask(gt_array, 0.5)

        # 转换为 Tensor
        # tp: [H, W, 3] -> [3, H, W]
        tp_tensor = torch.from_numpy(tp_array).to(torch.float32).permute(2, 0, 1)
        # gt: [H, W] -> [1, H, W]
        gt_tensor = torch.from_numpy(gt_array).to(torch.float32).unsqueeze(0)

        return DataSetItem(
            tp_tensor=tp_tensor,
            gt_tensor=gt_tensor,
            tp_name=tp_path.name,
            gt_name=gt_path.name if gt_path else ""
        )