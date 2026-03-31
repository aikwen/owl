import pathlib
from typing_extensions import TypedDict

import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointConfig:
    """Checkpoint 策略配置类，对应 YAML 文件中的 checkpoint 节点。

    Attributes:
        autosave (bool): 是否开启自动保存检查点功能。
        save_dir (str | pathlib.Path): 检查点（权重文件）的保存目录路径。
    """
    autosave: bool
    save_dir: str | pathlib.Path


@dataclass
class Checkpoint:
    """定义保存到 .pth 文件中的权重字典结构。

    Attributes:
        epoch: 当前保存时的 epoch 索引，用于断点续训时恢复进度（例如 finished_epoch）。
        model_state: 模型的权重参数，对应 model.state_dict()。
        optimizer_state: 优化器内部状态，包含动量、梯度缓存等。
        scheduler_state: 学习率调度器状态。
    """
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: Any


@dataclass
class CriterionBundle:
    """定义训练过程中计算损失函数所需的上下文信息容器。

    封装了当前 Step 的所有关键数据，作为协议传递给 Loss 函数 (OwlCriterion)。

    Attributes:
        current_epoch (int): 当前训练轮次的索引（从 0 开始）。
        current_batch (int): 当前轮次内的批次索引（从 0 开始）。
        model_output (Any): 模型的前向传播输出结果。
        gt (torch.Tensor): 真实标签（Ground Truth），直接来自 DataLoader。
        inputs (Any | None): 模型原本的输入数据。
    """
    current_epoch: int
    current_batch: int
    model_output: Any
    gt: torch.Tensor
    inputs: Any | None = None


@dataclass
class Main:
    """全局运行环境配置，对应 YAML 配置文件中的 main 节点。

    Attributes:
        log_name (str): 日志名称，用于生成日志文件名。
        device (torch.device): 计算设备对象（如 cpu, cuda:0）。
        mode (str): 运行模式。必须是以下值之一：
            'train', 'resume', 'finetune', 'validate', 'visualization'。
        checkpoint_path (str): 权重文件路径。用于断点恢复 (resume)、微调 (finetune)
            或验证 (validate)。
        cudnn_benchmark (bool): 是否启用 torch.backends.cudnn.benchmark。
            设为 True 可加速固定输入尺寸网络的训练，但会增加初始化时间。
    """
    log_name: str
    device: torch.device
    mode: str
    checkpoint_path: str
    cudnn_benchmark: bool

    def __post_init__(self):
        modes = ["train", "resume", "finetune", "validate", "visualization"]
        if self.mode not in modes:
            raise ValueError(f"Invalid mode: {self.mode}, must be in {modes}")


class DataSetBatch(TypedDict):
    """定义一个 Batch 的数据字典结构。

    通常作为 DataLoader 迭代出的对象，由 collate_fn 堆叠 DataSetItem 而成。

    Attributes:
        tp_tensor (torch.Tensor): 批次级篡改图像张量。
            Shape: [B, 3, H, W]，其中 B 为 Batch Size。
        gt_tensor (torch.Tensor): 批次级真实标签张量。
            Shape: [B, 1, H, W]。
        tp_name (list[str]): 当前 Batch 中所有样本的图像文件名列表。
            列表长度等于 Batch Size。
        gt_name (list[str]): 当前 Batch 中所有样本的标签文件名列表。
    """
    tp_tensor: torch.Tensor  # [B, 3, H, W]
    gt_tensor: torch.Tensor  # [B, 1, H, W]
    tp_name: list[str]
    gt_name: list[str]


class DataSetItem(TypedDict):
    """定义单条数据样本的字典结构。

    通常作为 Dataset.__getitem__ 的返回值。

    Attributes:
        tp_tensor (torch.Tensor): 篡改图像（Tampered Image）张量。
            Shape: [3, H, W] (RGB格式)
        gt_tensor (torch.Tensor): 真实标签（Ground Truth）张量。
            Shape: [1, H, W] (Mask格式)
        tp_name (str): 篡改图像的文件名（例如 "001_t.png"）。
        gt_name (str): 对应标签的文件名（例如 "001_mask.png"）。
    """
    tp_tensor: torch.Tensor
    gt_tensor: torch.Tensor
    tp_name: str
    gt_name: str

def build_checkpointConfig(checkpoint_config) -> CheckpointConfig:
    """根据 YAML 配置字典构建 CheckpointConfig 实例。

    Args:
        checkpoint_config (dict): 包含 'autosave' 和 'save_dir' 字段的原始配置字典。

    Returns:
        CheckpointConfig: 实例化后的检查点配置对象。
    """
    return CheckpointConfig(
        autosave=checkpoint_config["autosave"],
        save_dir=checkpoint_config["save_dir"],
    )


def build_main(main_config) -> Main:
    """根据配置字典构建全局运行环境配置 Main 实例。

    该函数负责解析设备配置（支持 "auto" 自动检测），并从嵌套的配置结构中
    提取运行模式、Checkpoint 路径等信息。

    Args:
        main_config (dict): 原始配置字典。通常包含以下键::

            - 'device': 设备字符串（"auto", "cpu", "cuda:0" 等）。
            - 'log_name': 日志标识符。
            - 'run': 包含 'mode', 'checkpoint_path', 'cudnn_benchmark' 的子字典。

    Returns:
        Main: 解析并初始化后的全局环境配置对象。
    """
    device: torch.device
    dev_str = main_config["device"]
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # 新增：检查 Mac MPS 加速
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        #  "cpu", "cuda:0", "mps" 等格式
        device = torch.device(dev_str)

    return Main(
        log_name=main_config["log_name"],
        device=device,
        mode=main_config["run"]["mode"],
        checkpoint_path=main_config["run"]["checkpoint_path"],
        cudnn_benchmark=main_config["run"]["cudnn_benchmark"],
    )
