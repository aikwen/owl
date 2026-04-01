import pathlib
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from statemachine import StateMachine, State

from .state import AppState, ExecMode
from .engine import OwlEngine
from ..toolkits.model import MODELS
from ..toolkits.criterion import CRITERIA
from ..toolkits.optimizer import OPTIMIZERS
from ..toolkits.scheduler import SCHEDULERS
from ..toolkits.visual import VISUALIZERS
from ..toolkits.model.base import OwlModel
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.visual.base import OwlVisualizer
from ..toolkits.data.dataloader import OwlDataLoader
from ..toolkits.common import fs
from ..toolkits.common.types import CheckpointDict


class OwlApp(StateMachine):
    """Owl level 1

    组装组件
    """

    # ==========================================
    # AppState
    # ==========================================
    empty = State(AppState.EMPTY.value, initial=True)  # 空状态
    instantiated = State(AppState.INSTANTIATED.value)            # 实例化组件
    mounted = State(AppState.MOUNTED.value)              # 初始化权重，device 之类
    running = State(AppState.RUNNING.value)            # 进入运行
    finished = State(AppState.FINISHED.value)          # 运行结束
    error = State(AppState.ERROR.value)                # 错误

    # ========================================================================
    # 状态转移图
    #
    # +---------+      +--------------+      +---------+      +---------+      +----------+
    # |  empty  |----->| instantiated |----->| mounted |----->| running |----->| finished |
    # +---------+      +--------------+      +---------+      +---------+      +----------+
    #      |                |                   |                 |
    #      | run_fail       | run_fail          | run_fail        | run_fail
    #      v                v                   v                 v
    # +-----------------------------------------------------------------------------+
    # |                                   error                                     |
    # +-----------------------------------------------------------------------------+
    # ========================================================================
    run_instantiate = empty.to(instantiated)
    run_mount = instantiated.to(mounted)
    run_start = mounted.to(running)
    run_complete = running.to(finished)
    run_fail = (empty.to(error) | instantiated.to(error) | mounted.to(error) | running.to(error))

    def __init__(self):
        # --- 存放实例化的组件 ---
        self.model: OwlModel | None = None
        self.criterion: OwlCriterion | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.visualizer: OwlVisualizer | None = None

        self.train_loader: DataLoader | None = None
        self.val_loaders: dict[str, DataLoader] | None = {}

        # --- 运行时信息 ---
        self.device: torch.device = torch.device("cpu")
        self.start_epoch: int = 0

        self.engine: OwlEngine | None = None

        super().__init__()

    def on_run_instantiate(self,
                     max_epochs: int,
                     model_name: str, model_cfg: Dict[str, Any],
                     criterion_name: str, criterion_cfg: Dict[str, Any],
                     optimizer_name: str, optimizer_cfg: Dict[str, Any],
                     owl_train_loader: OwlDataLoader | None = None,
                     owl_val_loaders: OwlDataLoader | None = None,
                     scheduler_name: str | None = None, scheduler_cfg: Dict[str, Any]| None = None,
                     visualizer_name: str | None = None, visualizer_cfg: Dict[str, Any]|None = None,
                     ):
        """Empty -> Instantiated：只用来实例化组件
        """
        self.model = MODELS.build(model_name, **model_cfg)
        self.criterion = CRITERIA.build(criterion_name, **criterion_cfg)

        """实例化 optimizer
         @OPTIMIZERS.register(name="adamw")
         def adamw(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        """
        if optimizer_name:
            injected_kwargs = {
                "model": self.model,
            }
            optimizer_cfg.update(injected_kwargs)
            self.optimizer = OPTIMIZERS.build(optimizer_name, **optimizer_cfg)
        # 数据加载器
        self.train_loader = owl_train_loader.get_train_loader() if owl_train_loader else None
        self.val_loaders = owl_val_loaders.get_valid_loaders() if owl_val_loaders else {}

        """实例化 scheduler
        @SCHEDULERS.register(name="poly")
        def poly(optimizer: optim.Optimizer, power: float, epochs: int, batches: int) -> optim.lr_scheduler.LRScheduler:
            total_iters = epochs * batches
            return optim.lr_scheduler.PolynomialLR(
                optimizer=optimizer,
                total_iters=total_iters,
                power=power
            )
        """
        #
        if scheduler_name:
            injected_kwargs = {
                "optimizer": self.optimizer,
                "epochs": max_epochs,
                "batches": len(self.train_loader) if self.train_loader else 1
            }

            scheduler_cfg = scheduler_cfg or {}
            scheduler_cfg.update(injected_kwargs)
            self.scheduler = SCHEDULERS.build(scheduler_name, **scheduler_cfg)
        if visualizer_name:
            self.visualizer = VISUALIZERS.build(visualizer_name, **(visualizer_cfg or {}))

    def on_run_mount(self, mode: ExecMode, checkpoint_path: str | pathlib.Path, device: str|torch.device):
        """Instantiated -> Mounted：加载 checkpoint 和移动 device

        加载权重和移动 device
        """
        self.device = torch.device(device)

        self.model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

        if str(checkpoint_path).strip():
            ckpt: CheckpointDict = fs.load_checkpoint(checkpoint_path, device=self.device)
            self.model.load_state_dict(ckpt["model_state"])

            if mode == ExecMode.TRAIN:
                if "optimizer_state" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                if self.scheduler and "scheduler_state" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state"])
                self.start_epoch = ckpt.get("epoch", -1) + 1

    def on_run_start(self, mode: ExecMode, max_epochs: int):
        """Mounted -> Running： 运行期

        开始运行
        """
        if mode == ExecMode.VALIDATE or mode == ExecMode.VISUALIZE:
            max_epochs = 1
            self.start_epoch = 0

        self.engine = OwlEngine(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loaders=self.val_loaders,
            visualizer=self.visualizer
        )

        self.engine.run(
            mode=mode,
            max_epochs=max_epochs,
            start_epoch=self.start_epoch,
            device=self.device
        )

    def launch(self,
               # ==========================================
               # 运行控制参数
               # ==========================================
               mode: ExecMode,
               max_epochs: int = 1,
               checkpoint_path: str = "",
               device: str = "cpu",

               # ==========================================
               # 核心组件参数
               # ==========================================
               model_name: str = "",
               model_cfg: Dict[str, Any] = None,
               criterion_name: str = "",
               criterion_cfg: Dict[str, Any] = None,
               optimizer_name: str = "",
               optimizer_cfg: Dict[str, Any] = None,

               # ==========================================
               # 数据与可选组件
               # ==========================================
               scheduler_name: str | None = None,
               scheduler_cfg: Dict[str, Any] | None = None,
               owl_train_loader: OwlDataLoader | None = None,
               owl_val_loaders: OwlDataLoader | None = None,
               visualizer_name: str | None = None,
               visualizer_cfg: Dict[str, Any] | None = None):
        """
        该方法会自动按照状态机的定义，依次触发组件实例化 (instantiated)、硬件分配与权重加载 (instantiated)、
        启动第二层任务 (start)，并最终收尾完成任务 (complete)。。

        Args:
            mode (ExecMode):任务执行模式，可选 `TRAIN`, `VALIDATE`, `VISUALIZE`。
            max_epochs (int, optional): 最大运行轮次。当 mode 为 VALIDATE 或 VISUALIZE 时，内部会强制重置为 1。默认为 1。
            checkpoint_path (str, optional): 断点续训或预训练权重的文件路径（如 '.pth'） 若为空字符串，则模型使用随机初始化权重。默认为 ""。
            device (str, optional): 目标物理设备，例如 "cuda", "cuda:0" 或 "cpu"。默认为 "cpu"。
            model_name (str, optional): 注册在 MODELS 中的模型名称。默认为 ""。
            model_cfg (Dict[str, Any], optional): 传递给模型构造函数的配置字典。默认为 None。
            criterion_name (str, optional): 注册在 CRITERIA 中的损失函数名称。默认为 ""。
            criterion_cfg (Dict[str, Any], optional): 传递给损失函数构造函数的配置字典。默认为 None。
            optimizer_name (str, optional): 注册在 OPTIMIZERS 中的优化器名称。默认为 ""。
            optimizer_cfg (Dict[str, Any], optional): 传递给优化器构造函数的配置字典。默认为 None。
            scheduler_name (str | None, optional): 注册在 SCHEDULERS 中的学习率调度器名称。默认为 None。
            scheduler_cfg (Dict[str, Any] | None, optional): 学习率调度器配置字典。默认为 None。
            visualizer_name (str | None, optional): 注册在 VISUALIZERS 中的可视化器名称。默认为 None。
            visualizer_cfg (Dict[str, Any] | None, optional): 可视化器配置字典。默认为 None。
            owl_train_loader (OwlDataLoader | None, optional): 封装了训练集的加载器对象。默认为 None。
            owl_val_loaders (OwlDataLoader | None, optional): 封装了验证集的加载器对象。默认为 None。

        Raises:
            Exception: 在装配、初始化或运行循环中抛出的任何底层运行时异常。捕获后状态机
                将跳转至 ERROR 态，并将异常重新抛出。

        Examples:
            >>> app = OwlApp()
            >>> app.launch(
            ...     mode=ExecMode.TRAIN,
            ...     max_epochs=30,
            ...     device="cuda:0",
            ...     model_name="MyModel",
            ...     model_cfg={"in_channels": 3},
            ...     optimizer_name="AdamW",
            ...     optimizer_cfg={"lr": 1e-3, "weight_decay": 1e-2},
            ...     owl_train_loader=train_data_loader,
            ...     owl_val_loaders=val_data_loader
            ... )
        """

        model_cfg = model_cfg or {}
        criterion_cfg = criterion_cfg or {}
        optimizer_cfg = optimizer_cfg or {}

        try:
            # empty -> instantiated：实例化组件
            self.run_instantiate(
                max_epochs=max_epochs,
                model_name=model_name,             model_cfg=model_cfg,
                criterion_name=criterion_name,     criterion_cfg=criterion_cfg,
                optimizer_name=optimizer_name,     optimizer_cfg=optimizer_cfg,
                scheduler_name=scheduler_name,     scheduler_cfg=scheduler_cfg,
                visualizer_name=visualizer_name,   visualizer_cfg=visualizer_cfg,
                owl_train_loader=owl_train_loader, owl_val_loaders=owl_val_loaders,
            )

            # instantiated -> mounted： 加载权重、移动 device 之类的
            self.run_mount(mode=mode, checkpoint_path=checkpoint_path, device=device)

            # mounted -> RUNNING： 开始运行
            self.run_start(mode=mode, max_epochs=max_epochs)

            # RUNNING -> FINISHED： 结束
            self.run_complete()

        except Exception as e:
            self.run_fail()
            raise e