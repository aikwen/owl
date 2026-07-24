import pathlib
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from statemachine import StateMachine, State

from ..toolkits.common.ckpt import load_checkpoint
from .state import AppState, ExecMode
from .engine import OwlEngine
from ..toolkits.evaluator import EVALUATORS
from ..toolkits.evaluator.base import OwlEvaluator
from ..toolkits.model import MODELS
from ..toolkits.criterion import CRITERIA
from ..toolkits.optimizer import OPTIMIZERS
from ..toolkits.scheduler import SCHEDULERS
from ..toolkits.visualizer import VISUALIZERS
from ..toolkits.model.base import OwlModel
from ..toolkits.criterion.base import OwlCriterion
from ..toolkits.visualizer.base import OwlVisualizer
from ..toolkits.data.dataloader import OwlDataLoader
from ..toolkits.common.ckpt import CheckpointDict
from .._monitor.ring import MonitorRing
from .._monitor.server import (
    MonitorServerHandle,
    start_monitor_server,
    stop_monitor_server,
)
from ._launch.normalize import normalize_launch_kwargs
from ._launch.types import (
    LaunchConfig,
    TrainLaunchConfig,
    ValidateLaunchConfig,
    VisualizeLaunchConfig,
)
from ._launch.dump import dump_launch_config


class OwlApp(StateMachine):
    """Owl level 1

    组装组件
    """

    # ==========================================
    # AppState
    # ==========================================
    empty_state = State(AppState.EMPTY.value, initial=True)                # 空状态
    instantiated_state = State(AppState.INSTANTIATED.value)                # 实例化组件
    mounted_state = State(AppState.MOUNTED.value)                          # 初始化权重，device 之类
    running_state = State(AppState.RUNNING.value)                          # 进入运行
    finished_state = State(AppState.FINISHED.value, final=True)            # 运行结束
    error_state = State(AppState.ERROR.value, final=True)                  # 错误

    # ========================================================================
    # 状态转移图
    #
    # +-------------+      +--------------------+      +---------------+      +---------------+      +----------------+
    # | empty_state |----->| instantiated_state |----->| mounted_state |----->| running_state |----->| finished_state |
    # +-------------+      +--------------------+      +---------------+      +---------------+      +----------------+
    #      |                        |                         |                       |
    #      | event_fail             | event_fail              | event_fail            | event_fail
    #      v                        v                         v                       v
    # +---------------------------------------------------------------------------------------------------------------+
    # |                                                  error_state                                                  |
    # +---------------------------------------------------------------------------------------------------------------+
    # ========================================================================
    event_instantiate = empty_state.to(instantiated_state)
    event_mount = instantiated_state.to(mounted_state)
    event_start = mounted_state.to(running_state)
    event_complete = running_state.to(finished_state)
    event_fail = (empty_state.to(error_state) | instantiated_state.to(error_state) | mounted_state.to(error_state) | running_state.to(error_state))

    def __init__(self):
        # --- 存放实例化的组件 ---
        self.nn_model: OwlModel | None = None
        self.criterion: OwlCriterion | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None  = None
        self.visualizer: OwlVisualizer | None = None
        self.evaluator: OwlEvaluator | None = None
        self.work_dir: pathlib.Path | None = None
        self.ckpt_autosave: bool = False

        self.train_loader: DataLoader | None = None
        self.val_loaders: dict[str, DataLoader] | None = {}

        # --- 运行时信息 ---
        self.device: torch.device = torch.device("cpu")
        self.start_epoch: int = 0

        self.engine: OwlEngine | None = None

        # --- monitor ---
        self.monitor_ring: MonitorRing | None = None
        self.monitor_handle: MonitorServerHandle | None = None

        # --- clean ---
        self._cleaned: bool = False
        super().__init__()

    def on_event_mount(self, config: LaunchConfig) -> None:
        """Instantiated -> Mounted：加载 checkpoint 和移动 device。"""
        mode = config["mode"]
        checkpoint_path = config["checkpoint_path"]
        finetune = config["finetune"]

        self.device = torch.device(config["device"])

        self.nn_model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

        # 检查权重是否存在，存在的话就加载权重
        if str(checkpoint_path).strip():
            ckpt: CheckpointDict = load_checkpoint(checkpoint_path, device=self.device)
            self.nn_model.load_state_dict(ckpt["model_state"])

            from .._internal.logger import logger

            logger.info(f"成功加载模型权重: {checkpoint_path}")

            if mode == ExecMode.TRAIN and not finetune:
                if self.optimizer and "optimizer_state" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                    logger.info("断点续训：加载恢复优化器状态")

                if self.scheduler and "scheduler_state" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state"])
                    logger.info("断点续训：加载恢复学习率优化器状态")

                self.start_epoch = ckpt.get("epoch", -1) + 1
                logger.info(f"断点续训：加载epoch，从 Epoch {self.start_epoch} 开始")

    def on_event_start(self, config: LaunchConfig) -> None:
        """Mounted -> Running：创建 Engine 并开始运行。"""
        mode = config["mode"]
        max_epochs = config["max_epochs"]

        if mode in (ExecMode.VALIDATE, ExecMode.VISUALIZE):
            self.start_epoch = 0

        # 创建 monitor
        if mode == ExecMode.TRAIN:
            train_config = cast(TrainLaunchConfig, config)
            monitor = train_config["monitor"]
        else:
            monitor = False

        if monitor:
            self.monitor_ring = MonitorRing()
            self.monitor_handle = start_monitor_server(self.monitor_ring)

            from .._internal.logger import logger

            logger.info(f"monitor server started: {self.monitor_handle.address}")
        else:
            self.monitor_ring = None
            self.monitor_handle = None

        # 创建下一层 Engine
        self.engine = OwlEngine(
            model=self.nn_model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loaders=self.val_loaders,
            visualizer=self.visualizer,
            evaluator=self.evaluator,
            work_dir=self.work_dir,
            ckpt_autosave=self.ckpt_autosave,
            monitor_ring=self.monitor_ring,
        )

        self.engine.run(
            mode=mode,
            max_epochs=max_epochs,
            start_epoch=self.start_epoch,
            device=self.device,
        )

    def launch(
        self,
        # 运行模式
        mode: ExecMode,
        # 模型名称
        model_name: str,
        ckpt_autosave: bool = False,
        # monitor
        monitor: bool = False,
        # 损失函数
        criterion_name: str = "",
        # 优化器
        optimizer_name: str = "",
        # 学习率优化器
        scheduler_name: str = "",
        # 数据集
        owl_train_loader: OwlDataLoader | None = None,
        owl_val_loaders: OwlDataLoader | None = None,
        # 训练轮次
        max_epochs: int = 1,
        # 预先加载的权重
        checkpoint_path: str = "",
        # 微调模式
        finetune: bool=False,
        # device 设备
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        # 相关配置
        model_cfg: dict[str, Any] | None = None,
        criterion_cfg: dict[str, Any] | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,

         # 可视化模式
        visualizer_name: str | None = None,
        visualizer_cfg: dict[str, Any] | None = None,
        # 评估模式
        evaluator_name: str | None = None,
        evaluator_cfg: dict[str, Any] | None = None,
    ):
        """
        该方法会自动按照状态机的定义，依次触发组件实例化 (instantiated_state)、硬件分配与权重加载 (mounted_state)、启动第二层任务 (running_state)，并最终收尾完成任务 (finished_state)

        Args:
            mode (ExecMode):任务执行模式，可选 `TRAIN`, `VALIDATE`, `VISUALIZE`。
            ckpt_autosave (bool): 是否在每一轮自动保存权重
            max_epochs (int, optional): 最大运行轮次。当 mode 为 VALIDATE 或 VISUALIZE 时，内部会强制重置为 1。默认为 1。
            checkpoint_path (str, optional): 断点续训或预训练权重的文件路径（如 '.pth'） 若为空字符串，则模型使用随机初始化权重。默认为 ""。
            monitor (bool): 是否开启训练监控服务。仅 TRAIN 模式生效。
            finetune (bool): 微调模式，only model ckpt， 只会加载模型权重
            device (str, optional): 目标物理设备，例如 "cuda", "cuda:0" 或 "cpu"。默认为 "cpu"。
            model_name (str, optional): 注册在 MODELS 中的模型名称。默认为 ""。
            model_cfg (dict[str, Any], optional): 传递给模型构造函数的配置字典。默认为 None。
            criterion_name (str, optional): 注册在 CRITERIA 中的损失函数名称。默认为 ""。
            criterion_cfg (dict[str, Any], optional): 传递给损失函数构造函数的配置字典。默认为 None。
            optimizer_name (str, optional): 注册在 OPTIMIZERS 中的优化器名称。默认为 ""。
            optimizer_cfg (dict[str, Any], optional): 传递给优化器构造函数的配置字典。默认为 None。
            scheduler_name (str | None, optional): 注册在 SCHEDULERS 中的学习率调度器名称。默认为 None。
            scheduler_cfg (dict[str, Any] | None, optional): 学习率调度器配置字典。默认为 None。
            visualizer_name (str | None, optional): 注册在 VISUALIZERS 中的可视化器名称。默认为 None。
            visualizer_cfg (dict[str, Any] | None, optional): 可视化器配置字典。默认为 None。
            evaluator_name (str | None): 注册的评估器名称。
            evaluator_cfg (dict[str, Any] | None): 评估器配置。
            owl_train_loader (OwlDataLoader | None, optional): 封装了训练集的加载器对象。默认为 None。
            owl_val_loaders (OwlDataLoader | None, optional): 封装了验证集的加载器对象。默认为 None。

        Raises:
            Exception: 在装配、初始化或运行循环中抛出的任何底层运行时异常。捕获后状态机
                将跳转至 ERROR 态，并将异常重新抛出。
        """

        try:
            # 规范化 launch 参数
            raw_kwargs = locals().copy()
            raw_kwargs.pop("self", None)
            config = normalize_launch_kwargs(**raw_kwargs)
            # 写入规范化后的启动配置
            dump_launch_config(config)
            # empty -> instantiated：实例化组件
            self.event_instantiate(
                config=config,
            )

            # instantiated -> mounted： 加载权重、移动 device...
            self.event_mount(
                config=config,
            )

            if config["mode"] == ExecMode.TRAIN:
                train_config = cast(TrainLaunchConfig, config)

                if not train_config["finetune"] and self.start_epoch >= train_config["max_epochs"]:
                    raise ValueError(
                        f"\n[Error] 断点续训冲突：\n"
                        f"当前权重已完成 {self.start_epoch} 轮，目标 'max_epochs' 为 {train_config['max_epochs']}。\n"
                        f"若要继续训练，请使用 'finetune=True' 开启新的微调阶段。"
                    )

            # mounted -> running：开始运行
            self.event_start(
                config=config,
            )

            # running -> finished：结束
            self.event_complete()

        except KeyboardInterrupt:
            from .._internal.logger import logger
            logger.warning("received Ctrl+C, owl stopped by user")
            self._fail_and_cleanup()
            return
        except Exception:
            self._fail_and_cleanup()
            raise

    def on_event_complete(self):
        self._cleanup()

    def on_event_instantiate(
        self,
        config: LaunchConfig
    ) -> None:
        """Empty -> Instantiated：实例化当前运行模式需要的组件。"""
        mode = config["mode"]
        self.work_dir = config["work_dir"]
        self.ckpt_autosave = False
        if mode == ExecMode.TRAIN:
            train_config = cast(TrainLaunchConfig, config)
            self.ckpt_autosave = train_config["ckpt_autosave"]
        # 打印日志
        from .._internal.logger import OwlLogger
        OwlLogger.setup(work_dir=self.work_dir)
        OwlLogger.welcome()

        # 实例化模型
        self.nn_model = MODELS.build(
            config["model_name"],
            **config["model_cfg"],
        )

        # 加载验证数据
        self.val_loaders = (
            config["owl_val_loaders"].get_valid_loaders()
            if config["owl_val_loaders"]
            else {}
        )

        if mode == ExecMode.TRAIN:
            train_config = cast(TrainLaunchConfig, config)

            # 实例化损失函数
            self.criterion = CRITERIA.build(
                train_config["criterion_name"],
                **train_config["criterion_cfg"],
            )

            # 实例化 optimizer，自动注入 model
            if train_config["optimizer_name"]:
                optimizer_cfg = dict(train_config["optimizer_cfg"])
                # 注入的 model
                optimizer_cfg["model"] = self.nn_model

                self.optimizer = OPTIMIZERS.build(
                    train_config["optimizer_name"],
                    **optimizer_cfg,
                )

            # 加载训练数据
            self.train_loader = train_config["owl_train_loader"].get_train_loader()

            # 实例化 scheduler，自动注入 optimizer、epochs、batches
            if train_config["scheduler_name"]:
                scheduler_cfg = dict(train_config["scheduler_cfg"])
                # 注入的参数
                scheduler_cfg.update(
                    {
                        "optimizer": self.optimizer,
                        "epochs": train_config["max_epochs"],
                        "batches": len(self.train_loader) if self.train_loader else 1,
                    }
                )

                self.scheduler = SCHEDULERS.build(
                    train_config["scheduler_name"],
                    **scheduler_cfg,
                )

            # 实例化 evaluator
            if train_config["evaluator_name"]:
                self.evaluator = EVALUATORS.build(
                    train_config["evaluator_name"],
                    **train_config["evaluator_cfg"],
                )

            return

        if mode == ExecMode.VALIDATE:
            validate_config = cast(ValidateLaunchConfig, config)

            self.evaluator = EVALUATORS.build(
                validate_config["evaluator_name"],
                **validate_config["evaluator_cfg"],
            )
            return

        if mode == ExecMode.VISUALIZE:
            visualize_config = cast(VisualizeLaunchConfig, config)

            self.visualizer = VISUALIZERS.build(
                visualize_config["visualizer_name"],
                **visualize_config["visualizer_cfg"],
            )
            return

        raise ValueError(f"不支持的运行模式：{mode}")

    def _shutdown_monitor(self):
        if self.monitor_handle is not None:
            stop_monitor_server(self.monitor_handle)
            self.monitor_handle = None

    def _cleanup(self):
        if self._cleaned:
            return

        self._cleaned = True
        self._shutdown_monitor()

        from .._internal.logger import OwlLogger
        if OwlLogger.is_initialized():
            OwlLogger.stop()

    def _fail_and_cleanup(self):
        try:
            self.event_fail()
        except Exception:
            pass
        finally:
            self._cleanup()
