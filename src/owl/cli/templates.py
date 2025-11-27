TRAIN_TEMPLATE = """
import torch
from torch import optim
from torch import nn
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

def main():
    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 12
    epochs = 30
    num_workers = 1
    shuffle = True
    
    # -------------------------------------------------------------------------
    # 配置数据集
    # -------------------------------------------------------------------------
    
    # 数据增强 pipeline
    transform_pipeline_train = [
        types.RotateConfig(p=0.5),
        types.VFlipConfig(p=0.5),
        types.HFlipConfig(p=0.5),
        types.JpegConfig(quality_low=70, quality_high=100, p=0.3),
        types.GblurConfig(kernel_low=3, kernel_high=15, p=0.3),
        types.ResizeConfig(width=512, height=512, p = 1),
    ]
    
    # 数据集列表
    datasets_train = [
        Path('example1'), 
        Path('example2')
    ]
    
    dataloader_train = dataset.create_dataloader(
                        dataset_list=datasets_train,
                        transform=img_aug.aug_compose(transform_pipeline_train),
                        batchsize=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        )
    
    # -------------------------------------------------------------------------
    # 配置优化器
    # lr 3e-4 ~ 1e-3
    # weight_decay 1e-4 ~ 0.1
    # -------------------------------------------------------------------------
    lr = 0.001
    weight_decay = 0.01
    model = SimpleModel()
    criterion = Criterion()
    adamw = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    power = 0.9
    total_iters = len(dataloader_train) * epochs
    poly_scheduler = optim \\
            .lr_scheduler \\
            .PolynomialLR(optimizer= adamw,
                              total_iters=total_iters,
                              power=power)
    
    
    # -------------------------------------------------------------------------
    # 运行
    # -------------------------------------------------------------------------

    e = (engine.OwlEngine()
        # 配置模型
        .config_model(model)
        # 配置数据集
        .config_dataloader(train_loader=dataloader_train)
        # 配置优化器
        .config_optimizer(adamw)
        .config_scheduler(poly_scheduler)
        # 配置损失函数
        .config_loss(criterion)
        # 配置训练 epochs
        .config_epochs(epochs)
        # 配置日志和权重保存策略和保存目录
        .config_output(checkpoint_dir="checkpoints", log_name="model_v1")
        .config_autosave(autosave=True) 
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