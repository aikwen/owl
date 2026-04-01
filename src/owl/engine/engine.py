import pathlib

from statemachine import StateMachine, State
from typing import Optional, Any
import torch

from .state import EngineState, ExecMode
from .pipeline import StepPipeline
from ..toolkits.model.base import OwlModel
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.visual.base import OwlVisualizer
from ..toolkits.data.dataloader import OwlDataLoader
from ..toolkits.data.types import DataSetBatch
from ..toolkits.common import fs
from ..toolkits.common.types import CheckpointDict
from torch.utils.data import DataLoader

class OwlEngine(StateMachine):

    # ==========================================
    #  Engine States
    # ==========================================
    empty = State(EngineState.EMPTY.value, initial=True) # 初始化
    pending = State(EngineState.PENDING.value)           # 注入组件
    inited = State(EngineState.INITED.value)             # 初始化模型
    running = State(EngineState.RUNNING.value)           # 运行
    finished = State(EngineState.FINISHED.value)
    error = State(EngineState.ERROR.value)

    # ==========================================
    # 状态转移
    # ==========================================
    # empty -> pending: 注入组件
    run_setup_components = empty.to(pending)
    # pending -> inited: 加载权重
    run_initialize = pending.to(inited)
    # inited -> running: 开启主循环
    run_start = inited.to(running)
    # running -> finished: 正常结束
    run_complete = running.to(finished)
    # * -> error: 捕获异常
    run_fail = (empty.to(error) | pending.to(error) | inited.to(error) | running.to(error))

    def __init__(self):
        self.model: OwlModel | None = None
        self.criterion: OwlCriterion | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.visualizer: OwlVisualizer|None = None

        # --- dataloader ---
        self.train_loader: DataLoader | None = None
        self.val_loaders: dict[str, DataLoader]  = {}

        self.pipeline: StepPipeline | None = None

        # --- 其它信息 ---
        self.device: torch.device = torch.device("cpu")
        self.current_mode: ExecMode | None = None
        self.current_epoch: int = 0
        self.current_step: int = 0  # 全局 Step
        self.max_epochs: int = 1

        super().__init__()

    # ==========================================
    # Action Hooks
    # ==========================================

    def on_run_setup_components(self,
                            model:OwlModel,
                            criterion:OwlCriterion,
                            optimizer:torch.optim.Optimizer,
                            scheduler:Any | None,
                            train_loader: OwlDataLoader | None,
                            val_loader: OwlDataLoader | None,
                            visualizer: Any | None = None,):
        """初始化组件"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.visualizer = visualizer
        if train_loader:
            self.train_loader: DataLoader = train_loader.get_train_loader()
        if val_loader:
            self.val_loaders: dict[str, DataLoader] = val_loader.get_valid_loaders()

    def on_run_initialize(self, mode: ExecMode,
                          checkpoint_path:str | pathlib.Path="",
                          device: str|torch.device = "cuda"):
        """初始化模型、优化器、学习率调整器"""
        self.device = torch.device(device)
        self.current_mode = mode

        # 将模型搬运到指定硬件
        self.model.to(self.device)
        self.criterion.to(self.device)
        # 加载权重
        if str(checkpoint_path).strip() != "":
            checkpoint:CheckpointDict = fs.load_checkpoint(checkpoint_path, device=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            # 只有训练模式下才会加载 optimizer 和 scheduler
            if mode == ExecMode.TRAIN:
                if "optimizer_state" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                if self.scheduler and "scheduler_state" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.current_epoch = checkpoint.get("epoch", -1) + 1

        # 实例化 pipline
        self.pipeline = StepPipeline(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            non_blocking=True,
        )

    def run(self, max_epochs: int = 1):
        """"""
        self.max_epochs = max_epochs
        self.run_start()

        try:
            if self.current_mode == ExecMode.TRAIN:
                self._run_train_loop(max_epochs)
            elif self.current_mode == ExecMode.VALIDATE:
                self._run_validate_loop()
            elif self.current_mode == ExecMode.VISUALIZE:
                self._run_visualize_loop()

            # 运行结束
            self.run_complete()

        except Exception as e:
            self.run_fail()
            raise e

    def _run_train_loop(self, max_epochs: int):
        """训练"""

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            # 将模型调整为 train 模式
            self.model.train()

            for batch_idx, batch_data in enumerate(self.train_loader):
                batch_data: DataSetBatch
                self.current_step += 1

                self.pipeline.do_step_flow(
                    batch_data=batch_data,
                    current_epoch=self.current_epoch,
                    current_step=self.current_step
                )

            if self.val_loaders:
                self._run_validate_loop()

    def _run_validate_loop(self):
        """验证"""
        # 将模型调整为eval 模式
        self.model.eval()

        with torch.no_grad():
            for dataset_name, dataloader in self.val_loaders.items():
                for batch_data in dataloader:
                    batch_data: DataSetBatch
                    batch_data['tp_tensors'] = batch_data['tp_tensors'].to(self.device, non_blocking=True,)
                    batch_data['gt_tensors'] = batch_data['gt_tensors'].to(self.device, non_blocking=True,)

                    outputs = self.model(
                        batch_data,
                        current_epoch=self.current_epoch,
                        current_step=self.current_step
                    )

                    self.criterion(
                        outputs,
                        batch_data,
                        current_epoch=self.current_epoch,
                        current_step=self.current_step
                    )

    def _run_visualize_loop(self):
        """可视化推理循环"""
        self.model.eval()
        with torch.no_grad():
            for _, dataloader in self.val_loaders.items():
                for batch_data in dataloader:
                    batch_data: DataSetBatch
                    batch_data['tp_tensors'] = batch_data['tp_tensors'].to(self.device, non_blocking=True)
                    batch_data['gt_tensors'] = batch_data['gt_tensors'].to(self.device, non_blocking=True,)

                    outputs = self.model(batch_data, self.current_epoch, self.current_step)

                    if self.visualizer:
                        _pred_masks = self.visualizer(outputs)
