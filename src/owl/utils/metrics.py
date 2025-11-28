import torch
from dataclasses import dataclass
from typing import List, Protocol
import math

@dataclass
class ConfusionMatrix:
    TP:torch.Tensor
    TN:torch.Tensor
    FP:torch.Tensor
    FN:torch.Tensor

    def __add__(self, other: 'ConfusionMatrix') -> 'ConfusionMatrix':
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented

        return ConfusionMatrix(
            TP=self.TP + other.TP,
            TN=self.TN + other.TN,
            FP=self.FP + other.FP,
            FN=self.FN + other.FN
        )

    def __iadd__(self, other: 'ConfusionMatrix') -> 'ConfusionMatrix':
        """
        mat += other
        :param other:
        :return:
        """
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented

        # 直接修改 self 的属性（Tensor 的加法）
        self.TP += other.TP
        self.TN += other.TN
        self.FP += other.FP
        self.FN += other.FN

        # 必须返回 self
        return self


def f1_score(mat: ConfusionMatrix) -> torch.Tensor:
    """
    计算 F1 Score。
    不进行 reduce 操作，返回维度与输入 mat 保持一致。
    如果 mat.TP 是 [N]，返回 [N] (每张图的 F1)
    :param mat:
    :return:
    """
    if mat is None:
        return torch.tensor(0.0)
    precision = mat.TP / (mat.TP + mat.FP + 1e-8)
    recall = mat.TP / (mat.TP + mat.FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def to_binary(tensor: torch.Tensor, threshold: float = 0.5, inplace: bool=False) -> torch.Tensor:
    """
    将一个 tensor 二值化，
    大于 threshold 的值会被调整为 1， 小于等于 threshold 的值会被调整为 0
    :param tensor:
    :param threshold:
    :param inplace:
    :return:
    """
    if not inplace:
        tensor = tensor.clone()
    tensor[tensor > threshold] = 1
    tensor[tensor <= threshold] = 0
    return tensor

def calculate_confusion_matrix(y_pred:torch.Tensor, y_true: torch.Tensor):
    """
    计算两个 tensor 之间的混淆矩阵
    注意：输入 y_pred 和 y_true 必须已经是 0/1 的二值矩阵
    :param y_pred: 里面的元素只能是 0 或者 1， shape [C, H, W], [N,C,H,W]
    :param y_true: 里面的元素只能是 0 或者 1， shape [C, H, W], [N,C,H,W]
    :return:
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"计算混淆矩阵的两个矩阵形状不一致，original shape: {y_pred.shape}, target shape: {y_true.shape}")


    y_pred = y_pred.clone()
    y_true = y_true.clone()

    if len(y_pred.shape) == 3:
        y_pred.unsqueeze_(0)
        y_true.unsqueeze_(0)

    y_pred = y_pred.float()
    y_true = y_true.float()

    return ConfusionMatrix(
        TP=torch.sum(y_pred * y_true, dim=(1, 2, 3)),
        TN=torch.sum((1 - y_pred) * (1 - y_true), dim=(1, 2, 3)),
        FP=torch.sum(y_pred * (1 - y_true), dim=(1, 2, 3)),
        FN=torch.sum((1 - y_pred) * y_true, dim=(1, 2, 3))
    )

class AUCSingleProtocol(Protocol):
    """
    计算单张图片的 auc 函数签名
    """
    def __call__(self, gt_prob: torch.Tensor, predict_prob: torch.Tensor) -> float:
        """
        Calculate AUC score for a single image.
        :param gt_prob: gt 概率图，值范围是 0 和 1; shape: [1, H, W] 或者 [H, W]
        :param predict_prob: 预测概率图， 值的范围是 [0, 1] 之间，不需要二值化; shape: [1, H, W] 或者 [H, W]
        """
        ...


def auc_single(gt_prob: torch.Tensor, predict_prob: torch.Tensor) -> float:
        """
        Calculate AUC score for a single image.
        from https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/evaluation/AUC.py
        :param gt_prob: gt 概率图，值范围是 0 和 1; shape: [1, H, W] 或者 [H, W]
        :param predict_prob: 预测概率图， 值的范围是 [0, 1] 之间，不需要二值化; shape: [1, H, W] 或者 [H, W]
        """
        y_true = gt_prob.flatten().to(torch.float32)
        y_scores = predict_prob.flatten().to(torch.float32)

        # Check if the mask has only one class
        if len(y_true.unique()) < 2:
            return float('nan')

        # Sort scores and corresponding true labels
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        # Calculate the number of positive and negative samples
        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        # Calculate cumulative true positives and false positives
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
        tpr = tps / n_pos
        fpr = fps / n_neg

        # Calculate AUC using the trapezoidal rule
        res = torch.trapz(tpr, fpr)

        return res.item()

def auc_batch(gt_prob: torch.Tensor, predict_prob: torch.Tensor, auc_single_func:AUCSingleProtocol) -> List[float]:
    """
    计算一个 batch 的 AUC
    :param gt_prob:
    :param predict_prob:
    :param auc_single_func:
    :return: type List[float] 会忽略 nan 的值
    """
    batch_size = gt_prob.shape[0]
    res:List[float] = []
    for i in range(batch_size):
        v = auc_single_func(gt_prob[i], predict_prob[i])
        if not math.isnan(v):
            res.append(v)
    return res
