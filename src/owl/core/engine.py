from dataclasses import asdict
import torch
from ..utils import io
from abc import ABC, abstractmethod
from collections.abc import Callable
from . import context, schemas, routines


class BaseEngine(ABC):
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = io.load_yaml(config_path)

    @abstractmethod
    def run(self):
        pass


class OwlTrainer(BaseEngine):
    """

    1. 构建训练上下文 (OwlTrainContext)。
    2. 管理训练状态 (Resume/Finetune)。
    3. 控制 Epoch 循环和模型保存。
    """

    def __init__(self, config_path: str, train_func: Callable):
        super().__init__(config_path)
        self.train_func = train_func
        self.ctx: context.OwlTrainContext = context.build_owlTrainContext(
            self.config,
            train_func=self.train_func
        )
        self.start_epoch = 0
        self._load_state()

    def _load_state(self):
        """根据 mode (resume/finetune) 加载权重文件并恢复状态。"""
        mode = self.ctx.main.mode
        ckpt_path = self.ctx.main.checkpoint_path

        # 第一种情况：从头训练
        if mode == "train":
            return

        # 第二种情况：需要加载权重 (Resume / Finetune)
        # 加载字典并转换为对象
        raw_dict = io.load_checkpoint(ckpt_path, map_location=self.ctx.main.device)
        try:
            checkpoint = schemas.Checkpoint(**raw_dict)
        except TypeError as e:
            raise ValueError(f"Checkpoint Error: {e}")

        # 加载模型权重
        self.ctx.model.load_state_dict(checkpoint.model_state)

        # 根据模式处理差异
        if mode == "resume":
            self.start_epoch = checkpoint.epoch + 1
            self.ctx.optimizer.load_state_dict(checkpoint.optimizer_state)
            if self.ctx.scheduler and checkpoint.scheduler_state:
                self.ctx.scheduler.load_state_dict(checkpoint.scheduler_state)

        elif mode == "finetune":
            # Finetune: 仅保留模型权重，Epoch/Optimizer 重置
            pass

    def _save_checkpoint(self, epoch: int):
        """保存当前 Epoch 的 Checkpoint。"""
        if not self.ctx.checkpoint_config.autosave:
            return

        ckpt = schemas.Checkpoint(
            epoch=epoch,
            model_state=self.ctx.model.state_dict(),
            optimizer_state=self.ctx.optimizer.state_dict(),
            scheduler_state=self.ctx.scheduler.state_dict() if self.ctx.scheduler else None
        )

        save_dict = asdict(ckpt)
        save_dir = self.ctx.checkpoint_config.save_dir

        # 文件名示例: epoch_01.pth
        # epoch 从 1 开始
        filename = f"epoch_{epoch+1:02d}.pth"
        io.save_checkpoint(save_dict, save_dir, filename)

    def run(self):
        if torch.cuda.is_available() and self.ctx.main.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        for epoch in range(self.start_epoch, self.ctx.epochs):
            # 设置为训练模式
            self.ctx.model.train()
            self._run_epoch(epoch)
            # 设置为评估模式
            self.ctx.model.eval()
            with torch.no_grad():
                self._run_validate(epoch)
            # 保存权重字典
            self._save_checkpoint(epoch)


    def _run_epoch(self, epoch: int):
        num_batches = len(self.ctx.dataloader_train)
        for batch_idx, batch_data in enumerate(self.ctx.dataloader_train, start=1):
            batch_data: schemas.DataSetBatch
            loss_val = self.train_func(
                batch_data=batch_data,
                ctx=self.ctx,
                current_epoch=epoch,
                current_batch=batch_idx
            )

    def _run_validate(self, epoch: int):
        for name, dataloader in self.ctx.validate_context.dataloader_validate.items():
            self.ctx.validate_context.validate_func(dataloader,
                                                    self.ctx.model,
                                                    self.ctx.main.device)


class OwlValidator(BaseEngine):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def run(self):
        pass


class OwlVisualizer(BaseEngine):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def run(self):
        pass

