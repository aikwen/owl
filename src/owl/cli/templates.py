TRAIN_TEMPLATE = """
from typing import Optional, Any
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from pathlib import Path

from owl.core import engine, dataset
from owl.utils import types, img_aug

# 模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

# 定义损失函数
class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, outputs, target):
        return self.loss(outputs, target)


class Factory(types.OwlFactory):
    def __init__(self):
        super().__init__()

    def create_model(self) -> nn.Module:
        # -------------------------------------------------------------------------
        # 配置模型
        # -------------------------------------------------------------------------
        return SimpleModel()

    def create_criterion(self) -> nn.Module:
        # -------------------------------------------------------------------------
        # 配置损失函数
        # -------------------------------------------------------------------------
        return Criterion()

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        # -------------------------------------------------------------------------
        # 配置优化器
        # -------------------------------------------------------------------------
        lr = 0.001
        weight_decay = 0.01
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def create_scheduler(self, optimizer: optim.Optimizer, epochs: int, batches: int) -> Optional[Any]:
        # -------------------------------------------------------------------------
        # 配置学习率优化器
        # -------------------------------------------------------------------------
        power = 0.9
        total_iters = epochs * batches
        poly_scheduler = optim \\
            .lr_scheduler \\
            .PolynomialLR(optimizer=optimizer,
                          total_iters=total_iters,
                          power=power)
        return poly_scheduler


    def create_train_dataloader(self) -> DataLoader:
        # -------------------------------------------------------------------------
        # 配置训练数据集
        # -------------------------------------------------------------------------
        # 数据增强 pipeline
        batch_size = 2
        num_workers = 1
        shuffle = True
        transform_pipeline_train = [
            types.RotateConfig(p=0.5),
            types.VFlipConfig(p=0.5),
            types.HFlipConfig(p=0.5),
            types.JpegConfig(quality_low=70, quality_high=100, p=0.3),
            types.GblurConfig(kernel_low=3, kernel_high=15, p=0.3),
            types.ResizeConfig(width=512, height=512, p=1),
        ]

        # 数据集列表
        datasets_train = [
            Path('example'),
        ]

        dataloader_train = dataset.create_dataloader(
            dataset_list=datasets_train,
            transform=img_aug.aug_compose(transform_pipeline_train),
            batchsize=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return dataloader_train



def main():
    # -------------------------------------------------------------------------
    # 运行, 初始化引擎
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    e = (engine.OwlEngine(log_name="model_v1")
        # 配置工厂
        .config_factory(factory=Factory())
        # 配置训练 epochs
        .config_epochs(epochs)
        # 配置权重保存目录和保存策略
        .config_checkpoints(checkpoint_dir="checkpoints", autosave=True)
        # 配置训练模式 TRAIN 从0开始, RESUME 断点续训, 
        .config_train_mode(types.TrainMode.TRAIN) 
        # 配置训练设备
        .config_device(device)
        # 构建
        .build())
    e.train()

if __name__ == '__main__':
    main()
"""