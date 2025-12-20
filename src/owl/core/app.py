import logging
from abc import ABC, abstractmethod
from pathlib import Path
from prettytable import PrettyTable
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override, final, List, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from .config import GlobalConfig, _Config, CheckpointState, TrainBatchConfig
from .dataset import DataSetBatch
from ..utils import file_io, console

_welcome_is_print:bool = False

class OwlCriterion(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch:TrainBatchConfig) -> torch.Tensor:
        ...


class _AppBase(ABC):
    @abstractmethod
    def model(self) -> nn.Module:
        ...

    @abstractmethod
    def criterion(self) -> OwlCriterion:
        ...

    @abstractmethod
    def optimizer(self, model: nn.Module) -> optim.Optimizer:
        ...

    @abstractmethod
    def scheduler(self, optimizer: optim.Optimizer, epochs: int, batches: int)-> LRScheduler:
        ...

    @abstractmethod
    def _run(self, checkpoint_state: CheckpointState | None, checkpoint_only:bool=False, cudnn_benchmark:bool=False):
        ...

    @abstractmethod
    def validate(self, model: nn.Module, dataloader:DataLoader) -> dict[str, float]:
        ...

    def run_train(self, cudnn_benchmark:bool=False):
        """开始一个新的训练任务（从头训练）。

        Args:
            cudnn_benchmark: 是否开启 cuDNN 的 benchmark 模式。
                如果输入图像尺寸固定，开启它可以加速训练（会自动寻找最优卷积算法）。
        """
        self._run(checkpoint_state=None, cudnn_benchmark=cudnn_benchmark)

    def run_resume(self, checkpoint_state: CheckpointState, cudnn_benchmark:bool=False):
        """从检查点恢复训练（断点续训）。

        不仅加载模型权重，还会恢复优化器状态、学习率调度器状态和 Epoch 进度，
        确保训练状态与保存时完全一致。

        Args:
            checkpoint_state: 加载的权重字典 (Checkpoint)。
            cudnn_benchmark: 是否开启 cuDNN 的 benchmark 模式。
                如果输入图像尺寸固定，开启它可以加速训练（会自动寻找最优卷积算法）。
        """
        self._run(checkpoint_state, checkpoint_only=False, cudnn_benchmark=cudnn_benchmark)

    def run_finetune(self, checkpoint_state: CheckpointState, cudnn_benchmark:bool=False):
        """基于检查点进行微调（Finetune）。

        仅加载模型的权重参数。优化器、学习率调度器和 Epoch 计数器将重新初始化，
        适用于迁移学习或在预训练模型基础上调整。

        Args:
            checkpoint_state: 加载的权重字典 (Checkpoint)。
            cudnn_benchmark: 是否开启 cuDNN 的 benchmark 模式。
                如果输入图像尺寸固定，开启它可以加速训练（会自动寻找最优卷积算法）。
        """
        self._run(checkpoint_state, checkpoint_only=True, cudnn_benchmark=cudnn_benchmark)


class _AppBuild(_AppBase, ABC):
    def __init__(self, global_config: GlobalConfig):
        super(_AppBuild, self).__init__()
        self._build_log: PrettyTable = PrettyTable(["global config", "value"])
        self._config: _Config = self._build_config(global_config)

    def _build_config(self, global_config:GlobalConfig)-> _Config:
        logger_train: logging.Logger = file_io.get_logger(log_file=global_config["log_name"],
                                                          mode="train",
                                                          is_format=True)
        logger_valid: logging.Logger = file_io.get_logger(log_file=global_config["log_name"],
                                                          mode="valid",
                                                          is_format=False)
        dataloader_train: DataLoader = global_config["dataloader_train"].build_dataloader_train()
        self._build_log.add_row(["train dataset", list(global_config["dataloader_train"].datasets_map.keys())])
        self._build_log.add_row(["train batch_size", global_config["dataloader_train"].batch_size])
        dataloader_valid: dict[str, DataLoader] = global_config["dataloader_valid"].build_dataloader_valid()
        self._build_log.add_row(["valid dataset", list(global_config["dataloader_valid"].datasets_map.keys())])
        self._build_log.add_row(["valid batch_size", global_config["dataloader_valid"].batch_size])
        current_epoch: int = 0
        epochs: int = global_config["epochs"]
        self._build_log.add_row(["total epoch", epochs])
        batches: int = len(dataloader_train)
        device: torch.device = global_config["device"]
        self._build_log.add_row(["device", device])
        model: nn.Module = self.model()
        criterion: OwlCriterion = self.criterion()
        model.to(device)
        criterion.to(device)
        optimizer: optim.Optimizer = self.optimizer(model)
        scheduler: LRScheduler = self.scheduler(optimizer, epochs, batches)

        return _Config(
            current_epoch=current_epoch,
            epochs = epochs,
            device = device,
            logger_train = logger_train,
            logger_valid = logger_valid,
            dataloader_train = dataloader_train,
            dataloader_valid = dataloader_valid,
            checkpoint_dir = global_config["checkpoint_dir"],
            checkpoint_autosave = global_config["checkpoint_autosave"],
            model= model,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
        )

class OwlApp(_AppBuild, ABC):
    def __init__(self, global_config: GlobalConfig):
        super(OwlApp, self).__init__(global_config)

    @final
    @override
    def _run(self, checkpoint_state: CheckpointState | None, checkpoint_only:bool=False, cudnn_benchmark:bool=False):
        # 打印欢迎词
        global _welcome_is_print
        if not _welcome_is_print:
            console.welcome()
            _welcome_is_print = True

        # 加载权重
        if checkpoint_state is not None:
            self._load_checkpoint(checkpoint_state, checkpoint_only)

        self._build_log.add_row(["start lr", self._config["optimizer"].param_groups[0]["lr"]])
        self._build_log.add_row(["start epoch", self._config["current_epoch"]+1])
        self._config["logger_train"].info(f"\n{self._build_log}")
        if torch.cuda.is_available() and cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        for epoch in range(self._config["current_epoch"], self._config["epochs"]):
            # ==========开始训练=============

            self._config["model"].train()
            self._config["current_epoch"] = epoch
            # 记录一些损失
            interval_loss = 0.0
            total_batches = len(self._config["dataloader_train"])
            batches_width = len(str(total_batches))
            epoch_width = len(str(self._config["epochs"]))
            pre_record = 0
            for batch, value in enumerate(self._config["dataloader_train"], start=1):
                value: DataSetBatch
                tp, gt = value["tp_tensor"], value["gt_tensor"]
                tp = tp.to(self._config["device"])
                gt = gt.to(self._config["device"])
                # 清空梯度
                self._config["optimizer"].zero_grad()
                # 获取输出
                outputs = self._config["model"](tp)
                # 构造损失函数输入
                train_batch:TrainBatchConfig = {
                    "current_epoch": epoch,
                    "current_batch": batch-1,
                    "gt":gt,
                    "model_output":outputs,
                }
                loss = self._config["criterion"](train_batch)
                loss.backward()
                # 计算梯度
                self._config["optimizer"].step()
                self._config["scheduler"].step()

                # 打印日志
                interval_loss += loss.item()
                if batch % 10 == 0 or batch == total_batches:
                    cnt = batch - pre_record
                    avg_loss = interval_loss / cnt
                    cur_lr = f"{self._config['optimizer'].param_groups[0]['lr']:.8f}"
                    self._config['logger_train'].info(f"Epoch [{epoch + 1:>{epoch_width}}/{self._config['epochs']}] | "
                                f"Batch [{batch:>{batches_width}}/{total_batches}] | "
                                f"Loss {avg_loss:.6f} | LR {cur_lr}")

                    # 清空这10轮的损失
                    interval_loss = 0.0
                    pre_record = batch

            # 保存权重
            if self._config["checkpoint_autosave"]:
                self._save_checkpoint()

            # validate
            self._config["model"].eval()
            with torch.no_grad():
                metrics_types = set()
                row_data:List[dict[str, Any]] = []
                for name, dataloader in self._config["dataloader_valid"].items():
                    name:str
                    dataloader: DataLoader
                    metric = self.validate(self._config["model"], dataloader)
                    metrics_types.update(metric.keys())
                    row_data.append({"dataset": name, "metrics": metric})
                self._logger_validate(metrics_types, row_data, epoch)

    @override
    def validate(self, model: nn.Module, dataloader:DataLoader) -> dict[str, float]:
        return {}

    def _load_checkpoint(self, checkpoint_state:CheckpointState, checkpoint_only:bool):
        self._config["model"].load_state_dict(checkpoint_state["model_state"])

        if not checkpoint_only:
            self._config["optimizer"].load_state_dict(checkpoint_state["optimizer_state"])
            self._config["scheduler"].load_state_dict(checkpoint_state["scheduler_state"])
            self._config["current_epoch"] = checkpoint_state["epoch"]+1

    def _save_checkpoint(self):
        current_epoch = self._config["current_epoch"]
        filename = f"epoch_{current_epoch:03d}.pth"
        Path(self._config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        save_path = Path(self._config["checkpoint_dir"]) / filename
        checkpoint_state: CheckpointState = CheckpointState(
            epoch = current_epoch,
            model_state=self._config["model"].state_dict(),
            optimizer_state=self._config["optimizer"].state_dict(),
            scheduler_state=self._config["scheduler"].state_dict(),
        )
        torch.save(checkpoint_state, save_path)

    def _logger_validate(self, metrics_types:set, row_data:List[dict[str, Any]], e:int):
        headers = ["dataset"]
        # 获取所有的指标类型
        metrics_types_list = sorted(metrics_types)
        headers.extend(metrics_types_list)
        tb = PrettyTable(headers)
        for row in row_data:
            li = [row["dataset"]]
            # 添加每个数据
            for m in metrics_types_list:
                value = row['metrics'].get(m, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                li.append(value)
            tb.add_row(li)
        # 打印到日志
        self._config["logger_valid"].info(f"Epoch: {e + 1}")
        self._config["logger_valid"].info(tb)
