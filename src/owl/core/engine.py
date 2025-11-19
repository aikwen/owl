from torch.utils.data import DataLoader
from torch import nn
import torch
import logging
from pathlib import Path

from typing import  Optional, Any, Dict
from ..utils import types
from ..utils import file_io
from ..utils.types import TrainMode
from .status import Status


class OwlEngine:
    def __init__(self):
        # 全局上下文
        self.status = Status()
        # 训练 epochs 数
        self.epochs: int = 0
        # 损失函数
        self.criterion: Optional[nn.Module] = None
        # 模式
        self.train_mode:types.TrainMode = TrainMode.TRAIN
        self.pre_checkpoint:Optional[Dict[str, Any]] = None
        # 输出权重文件夹名和日志名
        self.checkpoint_dir:str = ""
        self.log_name: str = ""
        # 是否在每个 epoch 保存 checkpoint
        self.autosave: bool = True
        # 是否调用了 build
        self._is_built:bool = False

    def config_model(self, model:nn.Module) -> 'OwlEngine':
        """
        配置模型
        :param model:
        :return:
        """
        self.status.model = model
        return self

    def config_dataloader(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> 'OwlEngine':
        """
        配置数据加载器
        :param train_loader:
        :param val_loader:
        :return:
        """
        self.status.train_loader = train_loader
        self.status.val_loader = val_loader
        return self

    def config_epochs(self, epochs:int) -> 'OwlEngine':
        self.epochs = epochs
        return self

    def config_optimizer(self, optimizer:torch.optim.Optimizer) -> 'OwlEngine':
        """
        配置优化器
        :param optimizer:
        :return:
        """
        self.status.optimizer = optimizer
        return self

    def config_scheduler(self, scheduler: Any) -> 'OwlEngine':
        """配置学习率调度器"""
        self.status.scheduler = scheduler
        return self

    def config_loss(self, criterion: nn.Module) -> 'OwlEngine':
        """配置损失函数"""
        self.criterion = criterion
        return self

    def config_device(self, device:torch.device) -> 'OwlEngine':
        """
        配置 device
        :param device:
        :return:
        """
        self.status.device = device
        return self

    def config_train_mode(self, mode:TrainMode, pre_checkpoint: Optional[Dict[str, Any]] = None) -> 'OwlEngine':
        """
        配置训练模式
        :param mode:
        :param pre_checkpoint:
        :return:
        """
        self.train_mode = mode
        if pre_checkpoint is not None:
            self.pre_checkpoint = pre_checkpoint
        return self

    def config_output(self, checkpoint_dir:str, log_name:str) -> 'OwlEngine':
        """
        设置权重输出文件夹 和 日志文件名
        :param checkpoint_dir:
        :param log_name:
        :return:
        """
        self.checkpoint_dir = checkpoint_dir
        self.log_name = log_name
        return self


    def build(self) -> 'OwlEngine':
        """
        构造整个 engine
        :return:
        """
        # 检查模型
        if self.status.model is None:
            raise RuntimeError("❌ Engine Build Error: Model 未配置。")

        # 检查数据集
        if self.status.train_loader is None:
            raise RuntimeError("❌ Engine Build Error: TrainLoader 未配置。")


        if self.epochs <= 0:
            raise RuntimeError("❌ Engine Build Error: epochs 必须大于等于 0")

        # 检查优化器
        if self.status.optimizer is None:
            raise RuntimeError("❌ Engine Build Error: Optimizer 未配置。")

        # 检查学习率优化器
        if self.status.scheduler is None:
            # 可以没有学习率优化器，暂时先 pass， 后续写到日志里面
            pass

        # 检查损失函数
        if self.criterion is None:
            raise RuntimeError("❌ Engine Build Error: criterion 未配置。")

        # 当训练模式是断点续训或微调的时候，检查是否提供权重文件
        if self.train_mode != TrainMode.TRAIN and self.pre_checkpoint is None:
            raise RuntimeError("❌ Engine Build Error: pre_checkpoint 未配置。")

        # 检查是否设置了日志输出和权重输出文件
        if self.checkpoint_dir == "" or self.log_name == "":
            raise RuntimeError("❌ Engine Build Error: 日志和权重文件夹未配置。")

        # 设置 device
        self.status.model.to(self.status.device)
        self.criterion.to(self.status.device)

        # 断点续训
        if self.train_mode == TrainMode.RESUME:
            self.status.load_state_dict(self.pre_checkpoint, only_model=False)
        # 迁移学习
        elif self.train_mode == TrainMode.FINETUNE:
            self.status.load_state_dict(self.pre_checkpoint, only_model=True)

        # 创建权重输出文件夹
        file_io.create_dir(self.checkpoint_dir)

        # build 结束
        self._is_built = True
        return self

    def save_checkpoint(self):
        filename = f"epoch_{self.status.epoch:03d}.pth"
        save_path = Path(self.checkpoint_dir) / filename
        torch.save(self.status.state_dict(), save_path)


    def train_one_epoch(self):
        self.status.model.train()
        loader = self.status.train_loader

        for i, batch in enumerate(loader):
            tp, gt, _, _ = batch
            tp = tp.to(self.status.device)
            gt = gt.to(self.status.device)
            # 清空梯度
            self.status.optimizer.zero_grad()
            # 获取输出
            outputs = self.status.model(tp)
            loss = self.criterion(outputs, gt)
            loss.backward()

            self.status.optimizer.step()
            if self.status.scheduler is not None:
                self.status.scheduler.step()

    def run(self):
        if not self._is_built:
            self.build()

        # 训练
        for epoch in range(self.status.epoch, self.epochs):
            self.status.epoch = epoch
            self.train_one_epoch()

            if self.autosave:
                self.save_checkpoint()

        # 暂时不验证

