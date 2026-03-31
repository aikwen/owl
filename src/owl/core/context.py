from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from .criterion import OwlCriterion
from . import schemas, dataset, criterion, optimizer, scheduler
from . import model as m
from ..utils import reflect

@dataclass
class BaseContext:
    """所有上下文对象的基类，封装了运行时通用的基础组件。

    Attributes:
        model (torch.nn.Module | None): 核心神经网络模型实例。
            通常在训练或推理时必须存在，但在某些仅涉及数据处理的上下文中可能为 None。
        main (schemas.Main): 全局运行环境配置对象。
            包含设备（Device）、运行模式（Mode）、日志名称等全局共享的配置信息。
    """
    model: torch.nn.Module | None
    main: schemas.Main

@dataclass
class OwlValidateContext(BaseContext):
    """验证阶段的上下文环境容器。

    继承自 BaseContext，额外封装了验证所需的数据加载器和自定义验证逻辑。

    Attributes:
        dataloader_validate (dict[str, DataLoader]): 验证集数据加载器字典。
            Key 为数据集名称（例如 "nist16", "coverage"），Value 为对应的 PyTorch DataLoader。
            允许同时在多个不同的数据集上评估模型性能。
        validate_func (Callable): 自定义的验证执行函数。
            该函数将被 Engine 调用以执行具体的评估逻辑（如计算 F1, AUC 等指标）。
    """
    dataloader_validate: dict[str, DataLoader]
    validate_func: Callable

@dataclass
class OwlVisualizationContext(BaseContext):
    """可视化阶段的上下文环境容器。

    用于在模型推理阶段生成和保存可视化结果。

    Attributes:
        dataloader_validate (dict[str, DataLoader]): 用于可视化的数据加载器字典。
            通常包含需要进行推理展示的验证集或测试集数据。
        visual_func (Callable): 自定义的可视化执行函数。
            该函数将被 Engine 调用，负责接收模型输出并将其转化为可视化的图像文件保存。
    """
    dataloader_validate: dict[str, DataLoader]
    visual_func: Callable

@dataclass
class OwlTrainContext(BaseContext):
    """训练阶段的完整上下文环境容器。

    该类聚合了训练循环所需的所有核心组件，作为协议对象传递给训练引擎 (Engine)。

    Attributes:
        epochs (int): 训练的总轮数 (Total Epochs)。
        dataloader_train (DataLoader): 训练集数据加载器。
            已经配置好 Batch Size、Shuffle 和增强流水线。
        criterion (OwlCriterion): 损失函数实例。
            负责计算预测值与真实标签之间的差异。
        optimizer (optim.Optimizer): 优化器实例 (如 AdamW, SGD)。
            负责根据梯度更新模型参数。
        scheduler (LRScheduler): 学习率调度器。
            负责在训练过程中动态调整优化器的学习率。
        validate_context (OwlValidateContext): 内嵌的验证上下文。
            用于在训练过程中（例如每个 Epoch 结束后）执行模型评估。
        checkpoint_config (schemas.CheckpointConfig): 检查点保存策略配置。
            定义了自动保存、保存路径等行为。
        train_func (Callable): 自定义的训练执行函数。
            定义了单个 Epoch 内的训练逻辑（前向传播、反向传播、梯度更新等）。
    """
    epochs: int  # 总 epoch 数
    dataloader_train: DataLoader
    criterion: OwlCriterion
    optimizer: optim.Optimizer
    scheduler: LRScheduler
    validate_context: OwlValidateContext
    checkpoint_config: schemas.CheckpointConfig
    train_func: Callable

def build_owlValidateContext(validate_config:dict) -> OwlValidateContext:
    """根据配置字典构建验证上下文 OwlValidateContext 实例。

    该函数解析配置，加载验证所需的数据集、模型类（无参实例化）以及自定义验证函数，
    最终打包成统一的上下文对象供 Engine 使用。

    Args:
        validate_config (dict): 验证阶段的配置字典。通常包含：
            - 'datasets_validate': 验证数据集配置（将被传给 build_owlDataloader）。
            - 'model': 模型类的全限定路径字符串（用于反射加载）。
            - 'main': 全局运行环境配置。

    Returns:
        OwlValidateContext: 初始化完成的验证上下文对象。
    """
    owl_dataloader = dataset.build_owlDataloader(validate_config["datasets_validate"])
    main = schemas.build_main(validate_config["main"])
    validate_func = reflect.load_validate_func(validate_config["datasets_validate"]["custom_validate_func"])
    model_instance = m.build_model(class_path=validate_config["model"])
    model_instance.to(main.device)

    return OwlValidateContext(
        dataloader_validate=owl_dataloader.build_dataloader_valid(),
        validate_func=validate_func,
        model = model_instance,
        main = main
    )

def build_owlVisualizationContext(validate_config:dict,
                                  visual_func:Callable) -> OwlVisualizationContext:
    """根据配置构建可视化上下文 OwlVisualizationContext 实例。

    用于准备模型推理和可视化所需的环境。与验证上下文构建不同，
    可视化执行函数 (visual_func) 需要作为参数直接传入。

    Args:
        validate_config (dict): 包含数据集和模型配置的字典。
            通常复用验证配置的结构，需包含 'datasets_validate', 'model', 'main' 等字段。
        visual_func (Callable): 执行可视化的具体函数。
            该函数将被注入到上下文中，负责接收模型输出并进行后续处理。

    Returns:
        OwlVisualizationContext: 初始化完成的可视化上下文对象。
    """
    owl_dataloader = dataset.build_owlDataloader(validate_config["datasets_validate"])
    main = schemas.build_main(validate_config["main"])
    model_instance = m.build_model(class_path=validate_config["model"])
    model_instance.to(main.device)
    return OwlVisualizationContext(
        dataloader_validate=owl_dataloader.build_dataloader_valid(),
        visual_func=visual_func,
        model = model_instance,
        main=main
    )


def build_owlTrainContext(train_config:dict,
                          train_func:Callable) -> OwlTrainContext:
    """构建训练上下文 OwlTrainContext 实例。

    训练流程的核心工厂函数，负责将配置字典转化为可执行的对象图。
    它会自动处理模型和 Loss 的设备迁移（CPU -> GPU），并计算 Steps 用于调度器配置。

    Args:
        train_config (dict): 包含完整训练参数的配置字典。必须包含以下关键字段::

            - 'epochs' (int): 总训练轮数。
            - 'main' (dict): 全局环境配置。
            - 'model' (str): 模型类路径。
            - 'criterion' (str): 损失函数类路径。
            - 'optimizer' (dict): 优化器配置。
            - 'schedule' (dict): 学习率调度器配置。
            - 'datasets_train' (dict): 训练数据集配置。
            - 'datasets_validate' (dict): 验证数据集配置。
            - 'checkpoints' (dict): 权重保存策略配置。
        train_func (Callable): 定义单个 Epoch 训练逻辑的执行函数。

    Returns:
        OwlTrainContext: 包含所有训练所需组件（模型、数据、优化器等）的上下文对象。
    """
    try:
        # 字段提取
        model_path = train_config["model"]
        criterion_path = train_config["criterion"]
        optimizer_cfg = train_config["optimizer"]
        scheduler_cfg = train_config["schedule"]
        datasets_train_cfg = train_config["datasets_train"]
        datasets_validate_cfg = train_config["datasets_validate"]
        checkpoints_cfg = train_config["checkpoints"]
        main_cfg = train_config["main"]

        # 关键子字段提取
        # epochs 定义在 datasets_train 下
        epochs = datasets_train_cfg["epochs"]
        # 自定义验证函数定义在 datasets_validate 下
        custom_validate_func_path = datasets_validate_cfg["custom_validate_func"]

    except KeyError as e:
        raise KeyError(f"构建训练上下文失败，配置文件缺少必要字段: {e}。请检查 train.yaml 结构。")
    # 全局环境
    main = schemas.build_main(main_cfg)

    # 构建模型 (Model)
    model_instance = m.build_model(class_path=model_path)
    model_instance.to(main.device)

    # 构建损失函数 (Criterion)
    loss = criterion.build_criterion(class_path=criterion_path)
    loss.to(main.device)

    # 构建训练数据加载器 (Train DataLoader)
    # build_owlDataloader 会处理 path, batchsize, transform 等字段
    owl_dataloader_train = dataset.build_owlDataloader(datasets_train_cfg)
    dataloader_train = owl_dataloader_train.build_dataloader_train()

    # 构建优化器
    # 依赖模型参数
    op = optimizer.build_optimizer(config=optimizer_cfg, model=model_instance)

    # 构建学习率优化器
    # 依赖优化器、总 Epochs 和每个 Epoch 的 Batch 数
    sc = scheduler.build_scheduler(
        config=scheduler_cfg,
        optimizer=op,
        epochs=epochs,
        batches=len(dataloader_train)
    )

    # 构建验证组件 (Validation Context)
    owl_dataloader_val = dataset.build_owlDataloader(datasets_validate_cfg)
    dataloader_validate = owl_dataloader_val.build_dataloader_valid()
    validate_func = reflect.load_validate_func(custom_validate_func_path)

    # 构建内部的验证上下文
    val_context = OwlValidateContext(
        dataloader_validate=dataloader_validate,
        validate_func=validate_func,
        model=model_instance,  # 共享同一个模型实例
        main=main,  # 共享同一个全局配置
    )

    # 构建 Checkpoint 配置
    ckpt_config = schemas.build_checkpointConfig(checkpoints_cfg)

    return OwlTrainContext(
        epochs=epochs,
        dataloader_train=dataloader_train,
        criterion=loss,
        optimizer=op,
        scheduler=sc,
        main=main,
        train_func=train_func,
        model=model_instance,
        checkpoint_config=ckpt_config,
        validate_context=val_context,
    )