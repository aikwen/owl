from typing import TypedDict

import torch
from torch import Tensor


class ImageMaskRecord(TypedDict):
    """Collector 解析后返回的标准格式。ImageDataset 内部的每个元素包含的内容

    Attributes:
        tp (str): 篡改图像（Tampered Image）的绝对路径。
        gt (str): 真实标签（Ground Truth）的绝对路径。
    """
    tp: str
    gt: str


class ImageMaskItem(TypedDict):
    """单条数据样本的字典结构。

    Dataset.__getitem__ 的返回值。

    Attributes:
        tp_tensor (torch.Tensor): 篡改图像（Tampered Image）张量。
            Shape: [3, H, W] (RGB格式)
        gt_tensor (torch.Tensor): 真实标签（Ground Truth）张量。
            Shape: [1, H, W] (Mask格式)
        tp_name (str): 篡改图像的文件名（例如 "001_t.png"）。
        gt_name (str): 对应标签的文件名（例如 "001_mask.png"）。
    """
    tp_tensor: Tensor
    gt_tensor: Tensor
    tp_name: str
    gt_name: str


class ImageMaskBatch(TypedDict):
    """一个 Batch 的数据字典结构。由多个 DataSetItem 组成

    Attributes:
        tp_tensor (torch.Tensor): 批次级篡改图像张量。
            Shape: [B, 3, H, W]
        gt_tensor (torch.Tensor): 批次级真实标签张量。
            Shape: [B, 1, H, W]。
        tp_name (list[str]): 当前 Batch 中所有样本的图像文件名列表。
            列表长度等于 Batch Size。
        gt_name (list[str]): 当前 Batch 中所有样本的标签文件名列表。
    """
    tp_tensor: torch.Tensor
    gt_tensor: torch.Tensor
    tp_name: list[str]
    gt_name: list[str]
