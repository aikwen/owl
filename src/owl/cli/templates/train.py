import copy

import torch
import torch.nn as nn
from owl.engine.app import OwlApp
from owl.engine.state import ExecMode
from owl.toolkits.data.dataloader import OwlDataLoader
from owl.toolkits.data.types import DataLoaderConfig
from owl.toolkits.model.base import OwlModel
from owl.toolkits.criterion.base import OwlCriterion
from owl.toolkits.model import MODELS
from owl.toolkits.criterion import CRITERIA
from owl.toolkits.data.augment.types import ResizeConfig


# ==========================================
# 模型
# ==========================================
@MODELS.register("dummy_model")
class DummyModel(OwlModel):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, batch_data, current_epoch=0, current_step=0, **kwargs):
        x = batch_data["tp_tensor"]  # 篡改图输入: [B, 3, H, W]
        logits = self.conv(x)         # 输出 Logits: [B, 1, H, W]
        return {"logits": logits}


# ==========================================
# 损失函数
# ==========================================
@CRITERIA.register("dummy_loss")
class DummyLoss(OwlCriterion):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, batch_data, current_epoch=0, current_step=0, **kwargs):
        logits = model_outputs["logits"]
        gt = batch_data["gt_tensor"]
        return self.loss_fn(logits, gt)


def main():
    # ==========================================
    # 数据集配置
    # ==========================================
    # 指向你项目中的测试数据集目录 (根据运行位置调整)
    dataset_paths = ["example"]
    common_aug = [
        ResizeConfig(height=256, width=256, p=1.0)
    ]

    # DataLoader 基础参数字典
    base_dl_config = DataLoaderConfig(
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    # 训练集 Loader
    train_dl_config = copy.deepcopy(base_dl_config)
    train_dl_config["shuffle"] = True
    train_loader = OwlDataLoader(
        dataset_paths=dataset_paths,
        dataloader_config=train_dl_config,
        aug_configs=common_aug
    )

    # 验证集 Loader (关闭打乱)
    val_dl_config =copy.deepcopy(base_dl_config)
    val_dl_config["shuffle"] = False
    val_loader = OwlDataLoader(
        dataset_paths=dataset_paths,
        dataloader_config=val_dl_config,
        aug_configs=common_aug
    )

    # ==========================================
    # 启动 Owl Engine
    # ==========================================
    app = OwlApp()

    app.launch(
        mode=ExecMode.TRAIN,  # 训练模式
        max_epochs=3,
        ckpt_autosave=True,   # 自动保存
        device="cuda" if torch.cuda.is_available() else "cpu",

        # 挂载组件
        model_name="dummy_model",
        model_cfg={"in_channels": 3},
        criterion_name="dummy_loss",
        criterion_cfg={},

        # 数据集
        owl_train_loader=train_loader,
        owl_val_loaders=val_loader,
    )


if __name__ == "__main__":
    main()