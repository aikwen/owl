import logging
from typing import Any, TypedDict
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from .dataset import OwlDataloader

class GlobalConfig(TypedDict):
    epochs: int
    device: torch.device
    log_name: str
    checkpoint_dir: str
    checkpoint_autosave: bool
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
    current_epoch: int # 从 0 开始
    current_batch: int # 从 0 开始
    gt: torch.Tensor
    model_output: Any


class CheckpointState(TypedDict):
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: Any
