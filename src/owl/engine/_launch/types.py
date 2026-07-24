from __future__ import annotations

import pathlib
from typing import Any, TypeAlias, TypedDict

import torch

from ..state import ExecMode
from ...toolkits.data.dataloader import OwlDataLoader


ComponentName: TypeAlias = str
"""组件注册名称。

例如:
    "dummy_model"
    "adamw"
    "poly"
    "default_auc_f1"
"""


ComponentConfig: TypeAlias = dict[str, Any]
"""组件构造参数字典。

该字典会被展开传入组件构造函数或注册函数，例如::

    MODELS.build(model_name, **model_cfg)
    CRITERIA.build(criterion_name, **criterion_cfg)
"""


class RawLaunchKwargs(TypedDict, total=False):
    """OwlApp.launch() 接收到的原始参数。

    该类型描述的是用户传入 launch() 后、尚未经过 normalize 处理的参数。

    特点:
        1. 部分字段可能缺失或为 None。
        2. 组件 name/cfg 可能尚未补默认值。
        3. work_dir 尚未生成。
        4. 非当前运行模式需要的参数可能仍然存在。
        5. 该类型主要用于 normalize_launch_kwargs() 的输入阶段。
    """

    # 运行模式
    mode: ExecMode

    # 训练轮次
    max_epochs: int

    # checkpoint / finetune
    checkpoint_path: str
    finetune: bool
    ckpt_autosave: bool

    # device
    device: str | torch.device

    # monitor
    monitor: bool

    # model
    model_name: ComponentName
    model_cfg: ComponentConfig | None

    # criterion
    criterion_name: ComponentName
    criterion_cfg: ComponentConfig | None

    # optimizer
    optimizer_name: ComponentName
    optimizer_cfg: ComponentConfig | None

    # scheduler
    scheduler_name: ComponentName
    scheduler_cfg: ComponentConfig | None

    # evaluator
    evaluator_name: ComponentName | None
    evaluator_cfg: ComponentConfig | None

    # visualizer
    visualizer_name: ComponentName | None
    visualizer_cfg: ComponentConfig | None

    # data
    owl_train_loader: OwlDataLoader | None
    owl_val_loaders: OwlDataLoader | None


class BaseLaunchConfig(TypedDict):
    """规范化后的基础启动配置。

    BaseLaunchConfig 表示 TRAIN / VALIDATE / VISUALIZE 三种模式都会使用的字段。

    该类型描述的是 normalize 之后的配置，因此:
        1. work_dir 已经生成。
        2. model_cfg 已经被规范化为 dict。
        3. max_epochs 已经根据 mode 做过必要处理。
        4. device / checkpoint_path / finetune 等通用运行参数已经保留。
    """

    # 运行模式
    mode: ExecMode

    # 工作目录，用于日志、可视化结果、checkpoint 等输出
    work_dir: pathlib.Path

    # 运行轮次。
    # TRAIN 使用用户传入值；
    # VALIDATE / VISUALIZE 会被 normalize 为 1。
    max_epochs: int

    # checkpoint 路径。为空字符串表示不加载权重。
    checkpoint_path: str

    # 是否以 finetune 方式加载 checkpoint。
    # finetune=True 时通常只加载模型权重，不恢复 optimizer / scheduler。
    finetune: bool

    # 运行设备，例如 "cuda" / "cuda:0" / "cpu" 或 torch.device。
    device: str | torch.device

    # 模型注册名称与构造参数
    model_name: ComponentName
    model_cfg: ComponentConfig

    # 验证数据加载器封装。
    # TRAIN 模式下可以为空；
    # VALIDATE / VISUALIZE 模式下必须提供。
    owl_val_loaders: OwlDataLoader | None


class TrainLaunchConfig(BaseLaunchConfig):
    """规范化后的训练模式配置。

    TRAIN 模式会额外需要:
        1. 训练数据；
        2. 损失函数；
        3. 优化器；
        4. 学习率调度器；
        5. 可选评估器；
        6. 可选 monitor。
    """

    # 是否自动保存 checkpoint
    ckpt_autosave: bool

    # 是否启动本地 gRPC monitor server。
    # 仅 TRAIN 模式生效。
    monitor: bool

    # 训练数据加载器封装。TRAIN 模式必须提供。
    owl_train_loader: OwlDataLoader

    # 损失函数注册名称与构造参数。TRAIN 模式必须提供 criterion_name。
    criterion_name: ComponentName
    criterion_cfg: ComponentConfig

    # 优化器注册名称与构造参数。
    # 如果用户未传，normalize 会注入默认 optimizer。
    optimizer_name: ComponentName
    optimizer_cfg: ComponentConfig

    # 学习率调度器注册名称与构造参数。
    # 如果用户未传，normalize 会注入默认 scheduler。
    scheduler_name: ComponentName
    scheduler_cfg: ComponentConfig

    # 评估器注册名称与构造参数。
    # TRAIN 模式下默认注入 default evaluator，用于每轮验证。
    evaluator_name: ComponentName
    evaluator_cfg: ComponentConfig


class ValidateLaunchConfig(BaseLaunchConfig):
    """规范化后的验证模式配置。

    VALIDATE 模式只负责加载模型并在验证集上计算评估指标。

    特点:
        1. max_epochs 会被 normalize 为 1。
        2. monitor 会被关闭。
        3. 不需要 criterion / optimizer / scheduler。
        4. 必须提供 owl_val_loaders。
        5. 需要 evaluator。
    """

    # 验证模式必须提供验证数据。
    owl_val_loaders: OwlDataLoader

    # 评估器注册名称与构造参数。
    evaluator_name: ComponentName
    evaluator_cfg: ComponentConfig


class VisualizeLaunchConfig(BaseLaunchConfig):
    """规范化后的可视化模式配置。

    VISUALIZE 模式只负责加载模型并在验证集上生成可视化结果。

    特点:
        1. max_epochs 会被 normalize 为 1。
        2. monitor 会被关闭。
        3. 不需要 criterion / optimizer / scheduler / evaluator。
        4. 必须提供 owl_val_loaders。
        5. 需要 visualizer。
    """

    # 可视化模式必须提供验证数据。
    owl_val_loaders: OwlDataLoader

    # 可视化器注册名称与构造参数。
    # 如果用户未传，normalize 会注入默认 visualizer。
    visualizer_name: ComponentName
    visualizer_cfg: ComponentConfig


LaunchConfig: TypeAlias = (
    TrainLaunchConfig
    | ValidateLaunchConfig
    | VisualizeLaunchConfig
)
"""规范化后的启动配置联合类型。
"""