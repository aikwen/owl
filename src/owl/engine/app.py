import pathlib
from typing import Any

import torch
from torch.utils.data import DataLoader
from statemachine import StateMachine, State
from datetime import datetime

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
from .._monitor import (
    MonitorConfig,
    MonitorState,
    MonitorServerHandle,
    start_monitor_server,
    stop_monitor_server,
)

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
        self.monitor_config: MonitorConfig | None = None
        self.monitor_state: MonitorState | None = None
        self.monitor_handle: MonitorServerHandle | None = None

        # --- clean ---
        self._cleaned: bool = False
        super().__init__()

    def on_event_mount(
        self,
        mode: ExecMode,
       checkpoint_path: str | pathlib.Path,
       finetune: bool,
       device: str|torch.device,
    ):
        """Instantiated -> Mounted：加载 checkpoint 和移动 device

        加载权重和移动 device
        """
        self.device = torch.device(device)

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
                    logger.info(f"断点续训：加载恢复优化器状态")
                if self.scheduler and "scheduler_state" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state"])
                    logger.info(f"断点续训：加载恢复学习率优化器状态")
                self.start_epoch = ckpt.get("epoch", -1) + 1
                logger.info(f"断点续训：加载epoch，从 Epoch {self.start_epoch} 开始")

    def on_event_start(
        self,
        mode: ExecMode,
        max_epochs: int,
        monitor: bool = False,
        monitor_cfg: dict[str, Any] | None = None,
    ):
        """Mounted -> Running： 运行期

        开始运行
        """
        if mode == ExecMode.VALIDATE or mode == ExecMode.VISUALIZE:
            max_epochs = 1
            self.start_epoch = 0

        # 创建 monitor
        self.monitor_config = MonitorConfig(
            enabled=bool(monitor),
            **(monitor_cfg or {}),
        )

        if mode == ExecMode.TRAIN and self.monitor_config.enabled:
            self.monitor_state = MonitorState(
                buffer_size=self.monitor_config.buffer_size,
            )
            self.monitor_handle = start_monitor_server(
                state=self.monitor_state,
                config=self.monitor_config,
                work_dir=self.work_dir,
            )
            from .._internal.logger import logger
            logger.bind(mode="train").info(
                f"monitor server started: {self.monitor_handle.address}"
            )
        else:
            self.monitor_state = None
            self.monitor_handle = None
        # 创建下一层
        self.engine = OwlEngine(
            model=self.nn_model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loaders=self.val_loaders,
            visualizer=self.visualizer,
            evaluator = self.evaluator,
            work_dir=self.work_dir,
            ckpt_autosave=self.ckpt_autosave,
            monitor_state=self.monitor_state,
        )

        self.engine.run(
            mode=mode,
            max_epochs=max_epochs,
            start_epoch=self.start_epoch,
            device=self.device
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
        monitor_cfg: dict[str, Any] | None = None,
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
            monitor (bool): 是否开启训练监控。仅 TRAIN 模式生效。
            monitor_cfg (dict[str, Any] | None): 训练监控配置，例如 buffer_size、transport、
                unix_socket_name、http_host、http_port。
            max_epochs (int, optional): 最大运行轮次。当 mode 为 VALIDATE 或 VISUALIZE 时，内部会强制重置为 1。默认为 1。
            checkpoint_path (str, optional): 断点续训或预训练权重的文件路径（如 '.pth'） 若为空字符串，则模型使用随机初始化权重。默认为 ""。
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
            # empty -> instantiated：实例化组件
            kwargs = locals().copy()
            kwargs.pop("self", None)
            safe_kwargs = self._default(**kwargs)
            self.event_instantiate(
                **safe_kwargs,
            )

            # instantiated -> mounted： 加载权重、移动 device...
            self.event_mount(
                mode=safe_kwargs["mode"],
                checkpoint_path=safe_kwargs["checkpoint_path"],
                finetune=safe_kwargs["finetune"],
                device=safe_kwargs["device"]
            )

            if safe_kwargs["mode"] == ExecMode.TRAIN and not safe_kwargs["finetune"]:
                if self.start_epoch >= safe_kwargs["max_epochs"]:
                    raise ValueError(
                        f"\n[Error] 断点续训冲突：\n"
                        f"当前权重已完成 {self.start_epoch} 轮，目标 'max_epochs' 为 {safe_kwargs['max_epochs']}。\n"
                        f"若要继续训练，请使用 'finetune=True' 开启新的微调阶段。"
                    )

            # mounted -> RUNNING： 开始运行
            self.event_start(
                mode=safe_kwargs["mode"],
                max_epochs=safe_kwargs["max_epochs"],
                monitor=safe_kwargs["monitor"],
                monitor_cfg=safe_kwargs["monitor_cfg"],
            )

            # RUNNING -> FINISHED： 结束
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

    def _default(self, **kwargs):
        """event_instantiate hook

        对输入的参数进行检查和设置默认值
        """
        mode = kwargs.get("mode")
        components = [
            ("model_name", "model_cfg"),
            ("criterion_name", "criterion_cfg"),
            ("optimizer_name", "optimizer_cfg"),
            ("scheduler_name", "scheduler_cfg"),
            ("evaluator_name", "evaluator_cfg"),
            ("visualizer_name", "visualizer_cfg"),
        ]

        for name_key, cfg_key in components:
            name_val = kwargs.get(name_key)
            cfg_val = kwargs.get(cfg_key)

            # 如果传了名字没传配置，或者传了配置没写名字，直接报错
            if (name_val and cfg_val is None) or (not name_val and cfg_val is not None):
                raise ValueError(
                    f"参数不匹配：'{name_key}' 和 '{cfg_key}' 必须成对提供，"
                    f"或全不提供以使用默认值。当前状态: {name_key}={name_val}, {cfg_key}={cfg_val}"
                )

            # 类型校验
            if name_val and not isinstance(name_val, str):
                raise TypeError(f"类型错误：'{name_key}' 必须是 str 类型，当前为 {type(name_val).__name__}")
            if cfg_val and not isinstance(cfg_val, dict):
                raise TypeError(f"类型错误：'{cfg_key}' 必须是 dict 类型，当前为 {type(cfg_val).__name__}")

        train_loader = kwargs.get("owl_train_loader")
        val_loaders = kwargs.get("owl_val_loaders")

        # 运行前检查
        if mode == ExecMode.TRAIN:
            # 检查数据集
            if train_loader is None:
                raise ValueError("参数错误: TRAIN 模式必须提供 'owl_train_loader'。")
            # 检查损失函数
            criterion_name = kwargs.get("criterion_name")
            if not criterion_name or not criterion_name.strip():
                raise ValueError(
                    "参数错误: TRAIN 模式下必须显式指定 'criterion_name'，框架不提供默认损失函数。"
                )
        elif mode in (ExecMode.VALIDATE, ExecMode.VISUALIZE):
            # 评估和可视化模式必须有验证数据集
            if not val_loaders:
                raise ValueError(f"参数错误: {mode.value} 模式下 'owl_val_loaders' 不能为空。")

        # 设置默认值
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        current_work_dir = pathlib.Path(f"./{timestamp}")
        kwargs["work_dir"] = current_work_dir

        # monitor 默认值
        monitor = kwargs.get("monitor")
        if not isinstance(monitor, bool):
            raise TypeError(
                f"类型错误：'monitor' 必须是 bool 类型，当前为 {type(monitor).__name__}"
            )

        monitor_cfg = kwargs.get("monitor_cfg")
        if monitor_cfg is None:
            kwargs["monitor_cfg"] = {}
        elif not isinstance(monitor_cfg, dict):
            raise TypeError(
                f"类型错误：'monitor_cfg' 必须是 dict 类型，当前为 {type(monitor_cfg).__name__}"
            )
        if kwargs.get("mode") != ExecMode.TRAIN:
            kwargs["monitor"] = False

        if mode == ExecMode.TRAIN:
            # 默认优化器注入
            if not kwargs.get("optimizer_name"):
                kwargs["optimizer_name"] = "adamw"
                kwargs["optimizer_cfg"] = {"lr": 1e-3, "weight_decay": 1e-2}

            # 默认学习率策略注入
            if not kwargs.get("scheduler_name"):
                kwargs["scheduler_name"] = "poly"
                kwargs["scheduler_cfg"] = {"power": 0.9}

        # 评估器默认值，非可视化模式下默认开启
        if not kwargs.get("evaluator_name") and mode != ExecMode.VISUALIZE:
            kwargs["evaluator_name"] = "default_auc_f1"
            kwargs["evaluator_cfg"] = {"threshold": 0.5}

        # visualizer 默认值
        if mode == ExecMode.VISUALIZE:
            if not kwargs.get("visualizer_name"):
                kwargs["visualizer_name"] = "default_mask"

        if kwargs.get("visualizer_name"):
            v_cfg = kwargs.get("visualizer_cfg") or {}
            user_save_dir = v_cfg.get("save_dir")
            if not user_save_dir:
                # 用户没传，使用默认的 `工作区/visual`
                v_cfg["save_dir"] = str(current_work_dir.joinpath("visual"))
            else:
                user_path = pathlib.Path(user_save_dir)
                if not user_path.is_absolute():
                    # 用户传了相对路径 (如 "custom_vis")，强行锚定到当前工作区
                    v_cfg["save_dir"] = str(current_work_dir.joinpath(user_path))
                # 用户传了绝对路径 (如 "/mnt/data/vis")，原样保留；

            # 如果用户没传阈值，给个默认 None
            if "threshold" not in v_cfg and mode == ExecMode.VISUALIZE:
                v_cfg["threshold"] = None

            kwargs["visualizer_cfg"] = v_cfg

        for _, cfg_key in components:
            if kwargs.get(cfg_key) is None:
                kwargs[cfg_key] = {}

        return kwargs

    def on_event_instantiate(
        self,
        mode: ExecMode,
        work_dir: pathlib.Path,
        max_epochs: int,
        ckpt_autosave: bool,
        model_name: str, model_cfg: dict[str, Any],
        criterion_name: str, criterion_cfg: dict[str, Any],
        optimizer_name: str, optimizer_cfg: dict[str, Any],
        owl_train_loader: OwlDataLoader | None,
        owl_val_loaders: OwlDataLoader | None,
        scheduler_name: str | None, scheduler_cfg: dict[str, Any]| None ,
        visualizer_name: str | None, visualizer_cfg: dict[str, Any]|None,
        evaluator_name: str | None, evaluator_cfg: dict[str, Any] | None,
        **kwargs,
    ):
        """Empty -> Instantiated：只用来实例化组件
        """
        self.work_dir = work_dir
        self.ckpt_autosave = ckpt_autosave
        # 打印日志
        from .._internal.logger import OwlLogger
        OwlLogger.setup(work_dir=self.work_dir)
        OwlLogger.welcome()

        self.nn_model = MODELS.build(model_name, **model_cfg)
        # 加载测试数据
        self.val_loaders = owl_val_loaders.get_valid_loaders() if owl_val_loaders else {}
        if mode == ExecMode.TRAIN:
            self.criterion = CRITERIA.build(criterion_name, **criterion_cfg)

            """实例化 optimizer, 自动注入model
             @OPTIMIZERS.register(name="adamw")
             def adamw(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
                return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            """
            if optimizer_name:
                injected_kwargs = {
                    "model": self.nn_model,
                }
                optimizer_cfg.update(injected_kwargs)
                self.optimizer = OPTIMIZERS.build(optimizer_name, **optimizer_cfg)
            # 加载训练数据
            self.train_loader = owl_train_loader.get_train_loader() if owl_train_loader else None

            """实例化 scheduler， 自动注入optimizer，epochs，batches
            @SCHEDULERS.register(name="poly")
            def poly(optimizer: optim.Optimizer, power: float, epochs: int, batches: int) -> optim.lr_scheduler.LRScheduler:
                total_iters = epochs * batches
                return optim.lr_scheduler.PolynomialLR(
                    optimizer=optimizer,
                    total_iters=total_iters,
                    power=power
                )
            """
            if scheduler_name:
                injected_kwargs = {
                    "optimizer": self.optimizer,
                    "epochs": max_epochs,
                    "batches": len(self.train_loader) if self.train_loader else 1
                }

                scheduler_cfg = scheduler_cfg or {}
                scheduler_cfg.update(injected_kwargs)
                self.scheduler = SCHEDULERS.build(scheduler_name, **scheduler_cfg)

        if mode == ExecMode.VISUALIZE:
            if visualizer_name:
                self.visualizer = VISUALIZERS.build(visualizer_name, **(visualizer_cfg or {}))
            else:
                raise ValueError(
                    "参数错误: VISUALIZE 模式下必须指定 'visualizer_name'，"
                )
        if mode in (ExecMode.TRAIN, ExecMode.VALIDATE):
            if evaluator_name:
                self.evaluator = EVALUATORS.build(evaluator_name, **(evaluator_cfg or {}))
            elif mode == ExecMode.VALIDATE:
                raise ValueError(
                    "参数错误: VALIDATE 模式下必须指定 'evaluator_name'"
                )

    def _shutdown_monitor(self):
        if self.monitor_handle is not None:
            stop_monitor_server(self.monitor_handle)
            self.monitor_handle = None

    def _cleanup(self):
        # 关闭监控器
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