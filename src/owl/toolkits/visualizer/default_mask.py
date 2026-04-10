import pathlib
from torchvision.utils import save_image

from .base import OwlVisualizer
from . import VISUALIZERS
from ..model.types import ModelOutput
from ..data.types import DataSetBatch

@VISUALIZERS.register(name="default_mask")
class DefaultMaskVisualizer(OwlVisualizer):
    """默认的二值化/概率掩码可视化器"""

    def __call__(self, batch_data: DataSetBatch, outputs: ModelOutput, dataset_name: str):
        # 获取模型原始输出并进行后处理 -> Shape: [B, 1, H, W]
        logits = outputs["logits"]
        pred_masks = self._process_logits(logits)

        # 从 DataSetBatch 中提取文件名列表
        img_names = batch_data["tp_names"]

        # 当前数据集保存目录: save_dir / dataset_name
        current_save_dir = self.save_dir.joinpath(dataset_name)
        current_save_dir.mkdir(parents=True, exist_ok=True)

        # 遍历 Batch，逐张保存图片
        for i in range(pred_masks.shape[0]):
            # 提取原图的文件名主体 (例如 "image_001.png" -> "image_001")
            img_stem = pathlib.Path(img_names[i]).stem

            # 构造保存路径 (例如: val_dataset_image_001_mask.png)
            save_path = current_save_dir.joinpath(f"{img_stem}_mask.png")
            # 保存单张图片
            save_image(pred_masks[i], str(save_path))