import logging
from typing import Any, TypedDict
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from .dataset import OwlDataloader


class GlobalConfig(TypedDict):
    """全局训练配置字典，定义了训练任务的核心参数。

    Attributes:
        epochs: 总训练轮数 (Total Epochs)。
        device: 训练使用的计算设备 (CPU 或 CUDA)，决定模型和数据加载的位置。
        log_name: 日志文件名的基础名称（无需文件后缀）。
            系统会自动基于此名称生成 `{log_name}_train.log` 和 `{log_name}_valid.log`。
        checkpoint_dir: 模型权重文件 (.pth) 的保存目录路径。
        checkpoint_autosave: 是否开启自动保存权重。
            如果为 True，通常会在每一轮 Epoch 结束时保存一次权重。
        dataloader_train: 训练集的数据加载器配置封装对象。
        dataloader_valid: 验证集的数据加载器配置封装对象。
    """
    epochs: int  # 总 epoch 数
    device: torch.device
    log_name: str  # log 的文件名，不需要后缀
    checkpoint_dir: str  # 权重保存路径
    checkpoint_autosave: bool  #
    dataloader_train: OwlDataloader
    dataloader_valid: OwlDataloader


class _Config(TypedDict):
    current_epoch: int
    epochs: int
    device: torch.device
    logger_train: logging.Logger
    logger_valid: logging.Logger
    checkpoint_dir: str
    checkpoint_autosave: bool
    dataloader_train: DataLoader
    dataloader_valid: dict[str, DataLoader]
    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: LRScheduler


class TrainBatchConfig(TypedDict):
    """定义训练过程中计算损失函数所需的批次上下文信息

    Attributes:
        current_epoch: 当前训练轮次的索引（从 0 开始）。
        current_batch: 当前轮次内的批次索引（从 0 开始）。
        gt: 真实标签（Ground Truth）张量，直接来自 DataLoader。
        model_output: 模型的前向传播输出结果，即 model(input)。
    """
    current_epoch: int  # 从 0 开始
    current_batch: int  # 从 0 开始
    gt: torch.Tensor
    model_output: Any


class CheckpointState(TypedDict):
    """定义保存到 .pth 文件中的权重字典结构。

    Attributes:
        epoch: 当前保存时的 epoch 索引，用于断点续训时恢复进度（例如 finished_epoch）。
        model_state: 模型的权重参数，对应 model.state_dict()。
        optimizer_state: 优化器内部状态，包含动量、梯度缓存等。
        scheduler_state: 学习率调度器状态。
    """
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: Any

if __name__ == "__main__":
    c:CheckpointState = CheckpointState(epoch=0, model_state=dict(), optimizer_state=dict(), scheduler_state=dict())
    var = c["epoch"]