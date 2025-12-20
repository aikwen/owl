TRAIN_TEMPLATE = r"""
from pathlib import Path

from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import override
from owl.core.app import OwlCriterion, OwlApp
from owl.core.config import TrainBatchConfig, GlobalConfig
from torch import nn, optim
import torch

from owl.core.dataset import OwlDataloader
from owl.utils import types


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Criterion(OwlCriterion):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    @override
    def forward(self, batch: TrainBatchConfig) -> torch.Tensor:
        pred = batch["model_output"]  # [B, 1, H, W]，来自 model(tp)
        gt = batch["gt"]  # [B, 1, H, W]，来自 dataloader
        gt = gt.to(dtype=pred.dtype)
        loss = self.bce(pred, gt)
        return loss


class App(OwlApp):
    def __init__(self, global_config: GlobalConfig):
        super().__init__(global_config=global_config)

    def model(self) -> nn.Module:
        return SimpleModel()

    def criterion(self) -> OwlCriterion:
        return Criterion()

    def optimizer(self, model: nn.Module) -> optim.Optimizer:
        lr = 0.001
        weight_decay = 0.01
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def scheduler(self, optimizer: optim.Optimizer, epochs: int, batches: int) -> LRScheduler:
        power = 0.9
        total_iters = epochs * batches
        poly_scheduler = optim \
            .lr_scheduler \
            .PolynomialLR(optimizer=optimizer,
                          total_iters=total_iters,
                          power=power)
        return poly_scheduler
    
    @override
    def validate(self, model: nn.Module, dataloader:DataLoader) -> dict[str, float]:
        return {"f1":0.99}


dataloader_train:OwlDataloader = OwlDataloader(
    datasets_map={"example":Path("example")},
    batch_size=12,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    transform_pipeline=[
        types.RotateConfig(p=0.5),
        types.VFlipConfig(p=0.5),
        types.HFlipConfig(p=0.5),
        types.JpegConfig(quality_low=70, quality_high=100, p=0.3),
        types.GblurConfig(kernel_low=3, kernel_high=15, p=0.3),
        types.ResizeConfig(width=512, height=512, p=1),
    ]
)

dataloader_valid:OwlDataloader = OwlDataloader(
    datasets_map={"example":Path("example")},
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    transform_pipeline=[
        types.ResizeConfig(width=512, height=512, p=1),
    ]
)

GLOBAL_CONFIG:GlobalConfig = GlobalConfig(
    epochs=30,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    log_name="2025-12-1",
    checkpoint_dir="checkpoint",
    checkpoint_autosave=True,
    dataloader_train=dataloader_train,
    dataloader_valid=dataloader_valid,
)

if __name__ == "__main__":
    app = App(global_config=GLOBAL_CONFIG)
    # 预训练
    app.run_train(cudnn_benchmark=True)
    
    # 断点续训
    # checkpoint_path = ""
    # checkpoint_state = torch.load(checkpoint_path, map_location= GLOBAL_CONFIG["device"])
    # app.run_resume(checkpoint_state = checkpoint_state, cudnn_benchmark=True)
    
    # 微调
    # checkpoint_path = ""
    # checkpoint_state = torch.load(checkpoint_path, map_location= GLOBAL_CONFIG["device"])
    # app.run_finetune(checkpoint_state = checkpoint_state, cudnn_benchmark=True)
"""