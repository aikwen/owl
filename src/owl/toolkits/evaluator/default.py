import torch

from . import EVALUATORS
from .base import OwlEvaluator
from ..model.types import ModelOutput
from ..data.types import DataSetBatch
from ..common.metrics import (
    f1_score,
    to_binary_by_threshold,
    calculate_confusion_matrix,
    auc_single,
    auc_batch
)

@EVALUATORS.register(name="default_auc_f1")
class DefaultEvaluator(OwlEvaluator):
    """默认评估器：计算 AUC 和 F1-Score"""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold (float): 将预测概率二值化为 0/1 的阈值，默认 0.5。
        """
        self.threshold = threshold

        self.auc_list: list[float] = []
        self.f1_list: list[float] = []

    def reset(self):
        """清空上一轮的缓存"""
        self.auc_list.clear()
        self.f1_list.clear()

    def update(self, outputs: ModelOutput, batch_data: DataSetBatch):
        """收集预测值和真实标签"""
        # 将 logits 并转为概率
        logits_tensor = outputs["logits"]
        predict_prob = torch.sigmoid(logits_tensor)
        gt_prob = batch_data['gt_tensor']

        # ==========================================
        # 计算图像级 AUC
        # ==========================================
        batch_aucs = auc_batch(gt_prob, predict_prob, auc_single)
        self.auc_list.extend(batch_aucs)

        # ==========================================
        # 计算图像级 F1-Score
        # ==========================================
        # 阈值二值化
        binary_pred = to_binary_by_threshold(predict_prob, threshold=self.threshold)

        # 获取当前 Batch 混淆矩阵，TP, TN, FP, FN shape 均为 [N]
        batch_cm = calculate_confusion_matrix(binary_pred, gt_prob)

        # 返回 shape 为 [N] 的 Tensor (每张图的 F1)
        batch_f1s = f1_score(batch_cm)

        # 将张量转换为 Python 浮点数列表，并追加至全局缓存中
        self.f1_list.extend(batch_f1s.cpu().tolist())
        del batch_f1s
        del batch_cm
        del binary_pred

    def compute(self) -> dict[str, float]:
        """计算并返回最终指标字典 (全部采用均值策略)"""

        # 计算全局图像级平均 AUC
        if len(self.auc_list) > 0:
            final_auc = sum(self.auc_list) / len(self.auc_list)
        else:
            final_auc = 0.0

        # 计算全局图像级平均 F1
        if len(self.f1_list) > 0:
            final_f1 = sum(self.f1_list) / len(self.f1_list)
        else:
            final_f1 = 0.0

        return {
            "auc": final_auc,
            "f1_score": final_f1
        }