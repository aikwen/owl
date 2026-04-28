from typing import TypedDict, Any

import torch


class ModelOutput(TypedDict, total=False):
    """Owl 模型输出的标准约定格式。

    模型 forward() 仅需输出未经过激活函数（如 Sigmoid/Softmax）处理的原始张量。
    因为原始 Logits 可以无损转换为概率值，而概率值转换为 Logits 会存在极大的数值精度丢失。

    下游组件使用规则:
        1. 损失函数 (Criterion)：直接获取 logits，配合如 `BCEWithLogitsLoss` 等损失函数，
           利用底层 C++ 的 LogSumExp 优化，确保反向传播时的数值绝对稳定性，避免梯度爆炸/消失。
        2. 评估/可视化 (Evaluator/Visualizer)：在组件内部明确当前的任务类型，
           手动对 logits 调用 `torch.sigmoid()`（二分类/多标签）或 `torch.softmax()`（多分类），将转化为 [0, 1] 的概率值后再进行指标计算。

    Attributes:
        logits (torch.Tensor): 模型的原始、未经过激活函数处理的预测输出。
            值域为 (-inf, +inf)。
        extra: 模型自定义输出，例如辅助监督 logits、中间特征、可视化信息等。
    """
    logits: torch.Tensor
    extra: dict[str, Any]