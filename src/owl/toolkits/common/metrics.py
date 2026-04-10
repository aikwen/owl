import torch
from dataclasses import dataclass
from typing import Protocol
import math

@dataclass
class ConfusionMatrix:
    """混淆矩阵数据容器。

    用于存储和累加像素级或图像级的分类统计指标（TP, TN, FP, FN）。

    Attributes:
        TP (torch.Tensor): 真正例数量。
        TN (torch.Tensor): 真负例数量。
        FP (torch.Tensor): 假正例数量。
        FN (torch.Tensor): 假负例数量。

    Examples:
        >>> import torch
        >>> mat1 = ConfusionMatrix(TP=torch.tensor([10]), TN=torch.tensor([80]),
        ...                        FP=torch.tensor([5]), FN=torch.tensor([5]))
        >>> mat2 = ConfusionMatrix(TP=torch.tensor([20]), TN=torch.tensor([70]),
        ...                        FP=torch.tensor([5]), FN=torch.tensor([5]))
        >>> total = mat1 + mat2
        >>> print(total.TP)
        tensor([30])
    """
    TP:torch.Tensor
    TN:torch.Tensor
    FP:torch.Tensor
    FN:torch.Tensor

    def __add__(self, other: 'ConfusionMatrix') -> 'ConfusionMatrix':
        """实现两个混淆矩阵的加法运算（mat1 + mat2）。

        Args:
            other (ConfusionMatrix): 另一个混淆矩阵实例。

        Returns:
            ConfusionMatrix: 包含属性相加结果的新实例。
        """
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented

        return ConfusionMatrix(
            TP=self.TP + other.TP,
            TN=self.TN + other.TN,
            FP=self.FP + other.FP,
            FN=self.FN + other.FN
        )

    def __iadd__(self, other: 'ConfusionMatrix') -> 'ConfusionMatrix':
        """实现混淆矩阵的原地加法运算（mat += other）。

        Args:
            other (ConfusionMatrix): 另一个混淆矩阵实例。

        Returns:
            ConfusionMatrix: 修改后的自身实例。
        """
        if not isinstance(other, ConfusionMatrix):
            return NotImplemented

        # 直接修改 self 的属性（Tensor 的加法）
        self.TP += other.TP
        self.TN += other.TN
        self.FP += other.FP
        self.FN += other.FN

        return self


def f1_score(mat: ConfusionMatrix) -> torch.Tensor:
    """计算 F1 Score 指标。

    该函数不进行 reduce 操作，返回的维度与输入混淆矩阵的 Tensor 维度保持一致。

    Args:
        mat (ConfusionMatrix): 包含统计值的混淆矩阵实例。如果为 None 则返回 0.0。

    Returns:
        torch.Tensor: 计算得到的 F1 Score 结果。

    Examples:
        >>> mat = ConfusionMatrix(TP=torch.tensor([1.0, 2.0]), TN=torch.tensor([0, 0]),
        ...                        FP=torch.tensor([0.0, 1.0]), FN=torch.tensor([0.0, 0.0]))
        >>> scores = f1_score(mat)
        >>> print(scores)
        tensor([1.0000, 0.8000])
    """
    if mat is None:
        return torch.tensor(0.0)
    precision = mat.TP / (mat.TP + mat.FP + 1e-8)
    recall = mat.TP / (mat.TP + mat.FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def to_binary_by_threshold(tensor: torch.Tensor, threshold: float = 0.5, inplace: bool=False) -> torch.Tensor:
    """基于阈值对 Tensor 进行二值化处理。

    Args:
        tensor (torch.Tensor): 需要处理的输入张量。
        threshold (float, optional): 二值化阈值。默认为 0.5。
        inplace (bool, optional): 是否进行原地修改。默认为 False。

    Returns:
        torch.Tensor: 二值化后的张量（0 或 1）。

    Examples:
        >>> x = torch.tensor([0.2, 0.6, 0.8])
        >>> binary_x = to_binary_by_threshold(x, threshold=0.5)
        >>> print(binary_x)
        tensor([0., 1., 1.])
    """
    if not inplace:
        tensor = tensor.clone()
    tensor[tensor > threshold] = 1
    tensor[tensor <= threshold] = 0
    return tensor

def calculate_confusion_matrix(y_pred:torch.Tensor, y_true: torch.Tensor) -> ConfusionMatrix:
    """计算预测图与真值图之间的混淆矩阵。

    该函数支持单张图像 [C, H, W] 或批量图像 [N, C, H, W] 输入。
    计算时会保留 Batch 维度（即在 dim=(1, 2, 3) 上求和），返回的混淆矩阵中
    TP, TN, FP, FN 的形状均为 [N]，对应 Batch 中每张图的独立统计结果。

    注意：
        输入的 y_pred 和 y_true 必须是已经完成二值化（0 或 1）的张量。

    Args:
        y_pred (torch.Tensor): 模型的预测二值掩码。
            支持形状: [C, H, W] 或 [N, C, H, W]。
        y_true (torch.Tensor): 真实的二值掩码（Ground Truth）。
            形状必须与 y_pred 完全一致。

    Returns:
        ConfusionMatrix: 包含批次统计结果的混淆矩阵实例。
            其成员 TP, TN, FP, FN 均为形状为 [N] 的 torch.Tensor。

    Raises:
        ValueError: 当 y_pred 与 y_true 的形状不匹配时抛出。

    Examples:
        >>> import torch
        >>> # 构造一个 batch_size=2, channel=1, 2x2 的数据
        >>> # 第一张图：完全匹配 (TP=4)
        >>> # 第二张图：一半匹配 (TP=2, FP=2)
        >>> pred = torch.tensor([
        ...     [[[1, 1], [1, 1]]],
        ...     [[[1, 1], [1, 1]]]
        ... ]).float()
        >>> gt = torch.tensor([
        ...     [[[1, 1], [1, 1]]],
        ...     [[[1, 1], [0, 0]]]
        ... ]).float()
        >>> cm = calculate_confusion_matrix(pred, gt)
        >>> print(cm.TP)
        tensor([4., 2.])
        >>> print(cm.FP)
        tensor([0., 2.])
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
        """计算单张图片的 AUC 分数。

        Args:
            gt_prob (torch.Tensor): 标签概率图，元素为 0 或 1。
            predict_prob (torch.Tensor): 预测概率图，值范围在 [0, 1] 之间。

        Returns:
            float: 计算得到的 AUC 分数。
        """
        ...

def auc_single(gt_prob: torch.Tensor, predict_prob: torch.Tensor) -> float:
    """计算单张图像的 AUC 分数。

    该函数会自动将输入的二维或三维概率图（Mask）展平（Flatten）为一维向量，
    并将所有像素点视为独立的样本点进行排序和面积计算。

    算法实现参考自 IMDLBenCo 评测库。当真值图中仅包含单一类别（全 0 或全 1）时，
    由于无法定义二分类边界，函数将返回 float('nan')。

    Args:
        gt_prob (torch.Tensor): 标签概率图，元素为 0 或 1。
            支持形状: [H, W], [1, H, W] 或已展平的 [L]。
        predict_prob (torch.Tensor): 预测概率图，值范围为 [0, 1]。
            形状必须与 gt_prob 一致。

    Returns:
        float: 计算得到的 AUC 标量值。如果无法计算（单类别）则返回 NaN。

    Examples:
        >>> gt = torch.tensor([0, 0, 1, 1])
        >>> pred = torch.tensor([0.1, 0.4, 0.3, 0.8])
        >>> score = auc_single(gt, pred)
        >>> print(round(score, 4))
        0.75
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

def auc_batch(gt_prob: torch.Tensor, predict_prob: torch.Tensor, auc_single_func:AUCSingleProtocol) -> list[float]:
    """批量计算一个批次（Batch）数据的图像级 AUC 分数。

    该函数会沿着第 0 维（Batch 维度）遍历数据。对于图像篡改定位任务，
    输入通常为 [N, 1, H, W] 的掩码，或者是已经展平后的 [N, L] 矩阵。
    计算过程中会自动过滤掉返回值为 NaN 的图像（即过滤掉纯负样本或纯正样本图）。

    Args:
        gt_prob (torch.Tensor): 批次真值张量，形状通常为 [N, 1, H, W] 或 [N, L]。
        predict_prob (torch.Tensor): 批次预测张量，形状与 gt_prob 一致。
        auc_single_func (AUCSingleProtocol): 用于计算单图 AUC 的函数或协议实现。

    Returns:
        list[float]: 包含该批次所有有效 AUC 分数的列表。

    Examples:
        >>> # 构造一个 batch_size=2 的已展平数据 [N, L]
        >>> # 第一张图：完美预测；第二张图：全负样本（预期返回 NaN 并被过滤）
        >>> gt_b = torch.tensor([
        ...     [0, 1, 0, 1],
        ...     [0, 0, 0, 0]
        ... ])
        >>> pred_b = torch.tensor([
        ...     [0.1, 0.9, 0.2, 0.8],
        ...     [0.1, 0.2, 0.1, 0.2]
        ... ])
        >>> results = auc_batch(gt_b, pred_b, auc_single)
        >>> # 结果列表里应该只剩下一个 1.0，因为第二张图被过滤了
        >>> print([round(v, 2) for v in results])
        [1.0]
    """
    batch_size = gt_prob.shape[0]
    res:list[float] = []
    for i in range(batch_size):
        v = auc_single_func(gt_prob[i], predict_prob[i])
        if not math.isnan(v):
            res.append(v)
    return res
