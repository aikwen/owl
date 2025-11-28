from torch.utils.data import DataLoader
from torch import nn
import torch
from pathlib import Path

from typing import  Optional, Any, Dict
from ..utils import types, file_io, console
from ..utils.types import TrainMode, OwlFactory
from ..utils import validator
from .status import Status

_welcome_is_print:bool = False

class OwlEngine:
    def __init__(self, log_name:str):
        # 全局上下文
        self.status:Status = Status()
        # 训练 epochs 数
        self.epochs: int = 0
        # 损失函数
        self.criterion: Optional[nn.Module] = None
        # 训练模式
        self.train_mode:types.TrainMode = TrainMode.TRAIN
        self.pre_checkpoint:Optional[Dict[str, Any]] = None
        # 输出权重文件夹名和日志名
        self.checkpoint_dir:str = ""
        self.log_name: str = log_name
        # 是否在每个 epoch 保存 checkpoint
        self.autosave: bool = True
        # 是否调用了 build
        self._is_built:bool = False
        # 验证函数
        self._is_val: bool = False
        # 工厂类
        self.factory: Optional[OwlFactory] = None
        # 优化设置
        self.cudnn_benchmark:bool = True

    def config_val(self, val_loader: Dict[str, DataLoader]) -> 'OwlEngine':
        """
        配置验证集数据加载器
        :param val_loader:
        :return:
        """
        self.status.val_loader = val_loader
        return self

    def config_cudnn_benchmark(self, o: bool=True) -> 'OwlEngine':
        """
        为整个网络的每个卷积层搜索最适合它的卷积实现算法，实现网络的加速。
        :param o: True 表示打开，默认值是 True
        :return:
        """
        self.cudnn_benchmark = o
        return self

    def config_factory(self, factory: OwlFactory) -> 'OwlEngine':
        """
        工厂
        :param factory:
        :return:
        """
        self.factory = factory
        return self

    def config_epochs(self, epochs:int) -> 'OwlEngine':
        """
        配置 epoch 数量
        :param epochs:
        :return:
        """
        self.epochs = epochs
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

    def config_checkpoints(self, checkpoint_dir:str, autosave:bool=True) -> 'OwlEngine':
        """
        设置权重输出文件夹名称和是否自动保存
        :param checkpoint_dir: 权重输出文件夹名称，不存在会自动创建
        :param autosave: 是否在每个 epoch 后自动保存
        :return:
        """
        self.checkpoint_dir = checkpoint_dir
        self.autosave = autosave
        return self

    def build(self) -> 'OwlEngine':
        """
        构造整个 engine
        :return:
        """
        # 打印欢迎词
        global _welcome_is_print
        if not _welcome_is_print:
            console.welcome()
            _welcome_is_print = True

        # 检查是否设置了日志输出文件
        validator.check(self.log_name != "", "❌ Engine Build Error: 日志文件名未配置。")
        logger = file_io.create_logger(self.log_name, "train")
        # 打印当前设备
        logger.info(f"✅ 当前训练设备：{self.status.device}")
        # ========= 组装模型 ========
        validator.check(self.factory is not None, "❌ Engine Build Error: 未配置 OwlFactory")
        # 创建模型
        self.status.model = self.factory.create_model()
        validator.check(self.status.model is not None, "❌ Engine Build Error: Model 未正确初始化。")
        logger.info("✅ 初始化模型")

        # 创建损失函数
        self.criterion = self.factory.create_criterion()
        validator.check(self.criterion is not None, "❌ Engine Build Error: criterion 未正确初始化。")
        logger.info("✅ 初始化损失函数")

        # ⚠️⚠️⚠️ 模型和损失函数都移动到 device 中
        self.status.model.to(self.status.device)
        self.criterion.to(self.status.device)

        # 检查训练数据集
        self.status.train_loader = self.factory.create_train_dataloader()
        validator.check(self.status.train_loader is not None, "❌ Engine Build Error: TrainLoader 未配置。")
        logger.info(f"✅ 初始化训练数据集, batch_size: {self.status.train_loader.batch_size}")

        # 检查 epoch
        validator.check(self.epochs > 0, "❌ Engine Build Error: epochs 必须大于等于 0")
        logger.info(f"✅ 初始化epoch: {self.epochs}")

        # 创建优化器
        self.status.optimizer = self.factory.create_optimizer(self.status.model)
        validator.check(self.status.optimizer is not None, "❌ Engine Build Error: Optimizer 未正确初始化。")
        logger.info(f"✅ 初始化优化器")

        # 创建学习率调整器
        self.status.scheduler = self.factory.create_scheduler(optimizer=self.status.optimizer,
                                                              epochs=self.epochs,
                                                              batches=len(self.status.train_loader))
        if self.status.scheduler is None:
            logger.warning("⚠️ 学习率调整器未设置")
        else:
            logger.info(f"✅ 初始化学习率调整器")

        # 检查训练模式：当训练模式是断点续训或微调的时候，检查是否提供权重文件
        logger.info(f"✅ 当前训练模式：{self.train_mode}")
        if self.train_mode != TrainMode.TRAIN and self.pre_checkpoint is None:
            raise RuntimeError("❌ Engine Build Error: pre_checkpoint 未配置。")

        # 加载预训练或者断点续训权重
        # 断点续训
        if self.train_mode == TrainMode.RESUME:
            self.status.load_state_dict(self.pre_checkpoint, only_model=False)
            logger.info(f"✅ 加载权重成功！")
        # 迁移学习
        elif self.train_mode == TrainMode.FINETUNE:
            self.status.load_state_dict(self.pre_checkpoint, only_model=True)
            logger.info(f"✅ 加载权重成功！")

        # 检查验证数据集
        self.status.val_loader = self.factory.create_val_dataloader()
        if self.status.val_loader is None or len(self.status.val_loader) == 0:
            logger.warning("⚠️ 未添加验证数据集")
        else:
            logger.info(f"✅ 初始化验证数据集: {[name for name in self.status.val_loader.keys()]}")

        # ========= 组装结束 ========

        # 创建权重输出文件夹
        if self.autosave:
            validator.check(self.checkpoint_dir != "", "❌ Engine Build Error: 权重文件夹未配置。")
            file_io.create_dir(self.checkpoint_dir)
            logger.info(f"✅ 权重保存目录：{self.checkpoint_dir}")
        logger.info(f"{'✅' if self.autosave else '⚠️'} 权重自动保存：{self.autosave}")

        # build 结束
        self._is_built = True
        logger.info("✅ 构建结束!")
        return self

    def save_checkpoint(self):
        filename = f"epoch_{self.status.epoch:03d}.pth"
        save_path = Path(self.checkpoint_dir) / filename
        torch.save(self.status.state_dict(), save_path)

    def train_one_epoch(self):
        logger = file_io.create_logger(self.log_name, "train")
        # 转为训练模式
        self.status.model.train()
        loader = self.status.train_loader

        interval_loss = 0.0
        total_batches = len(loader)
        batches_width = len(str(total_batches))
        pre_record = 0
        for i, batch in enumerate(loader, start=1):
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

            interval_loss += loss.item()
            # 每 10 轮打印平均损失和当前的学习率
            if i % 10 == 0 or i == total_batches:
                cnt = i - pre_record
                avg_loss = interval_loss / cnt
                cur_lr = f"{self.status.optimizer.param_groups[0]['lr']:.8f}"
                logger.info(f"Epoch [{self.status.epoch:03d}/{self.epochs}] | "
                            f"Batch [{i:>{batches_width}}/{total_batches}] | "
                            f"Loss {avg_loss:.6f} | LR {cur_lr}")

                # 清空这10轮的损失
                interval_loss = 0.0
                pre_record = i

    def train(self):
        if not self._is_built:
            self.build()

        logger = file_io.create_logger(self.log_name, "train")
        # 训练之前的优化
        if torch.cuda.is_available() and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        logger.info("✅ 开始训练")
        # 训练
        for epoch in range(self.status.epoch, self.epochs):
            self.status.epoch = epoch
            self.train_one_epoch()

            if self.autosave:
                self.save_checkpoint()

            # 验证
            if self._is_val:
                logger_val = file_io.create_logger(self.log_name, "val")
                self.status.model.eval()

