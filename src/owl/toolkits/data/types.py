from typing import TypedDict

import torch
from torch import Tensor


class DataRecord(TypedDict):
    """Collector 解析后返回的标准格式。ImageDataset 内部的每个元素包含的内容

    Attributes:
        tp (str): 篡改图像（Tampered Image）的绝对路径。
        gt (str): 真实标签（Ground Truth）的绝对路径。
    """
    tp: str
    gt: str


class DataSetItem(TypedDict):
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


class DataSetBatch(TypedDict):
    """一个 Batch 的数据字典结构。由多个 DataSetItem 组成

    Attributes:
        tp_tensors (torch.Tensor): 批次级篡改图像张量。
            Shape: [B, 3, H, W]
        gt_tensors (torch.Tensor): 批次级真实标签张量。
            Shape: [B, 1, H, W]。
        tp_names (list[str]): 当前 Batch 中所有样本的图像文件名列表。
            列表长度等于 Batch Size。
        gt_names (list[str]): 当前 Batch 中所有样本的标签文件名列表。
    """
    tp_tensors: torch.Tensor
    gt_tensors: torch.Tensor
    tp_names: list[str]
    gt_names: list[str]


class DataLoaderConfig(TypedDict):
    """DataLoader 的配置参数字典。

    Attributes:
        batch_size (int): 每个 Batch 的样本数。默认值为 1。
        num_workers (int): 用于数据加载的子进程数。0 表示在主进程中加载。
        shuffle (bool): 是否在每个 Epoch 开始时打乱数据。
        pin_memory (bool): 是否将 Tensor 拷贝到 CUDA 固定内存中，加速 GPU 加载。
        persistent_workers (bool): 训练结束后是否保留子进程，能减少每个 Epoch 开始时的初始化耗时。
        drop_last (bool): 如果数据集大小不能被 batch_size 整除，是否丢弃最后一个不完整的 Batch。
    """
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool