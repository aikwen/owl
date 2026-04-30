from __future__ import annotations

import random


def seed_everything(
    seed: int = 3407,
    deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = False,
    warn_only: bool = True,
) -> int:
    """
    固定随机种子，尽可能提高实验的可复现性。

    注意：
        完全可复现通常无法跨 PyTorch 版本、CUDA 版本、硬件平台保证。
        本函数主要控制 Python、NumPy、PyTorch 的随机数状态。
        如果需要控制 PYTHONHASHSEED，建议在启动 Python 前通过环境变量设置：

            PYTHONHASHSEED=3407 python train.py

    Args:
        seed (int): 随机种子。
        deterministic (bool): 是否开启 cuDNN 确定性模式。
        benchmark (bool): 是否开启 cuDNN benchmark。
        use_deterministic_algorithms (bool): 是否强制 PyTorch 使用确定性算法。
        warn_only (bool): 当强制确定性算法遇到问题时，是否只发出警告。

    Returns:
        int: 当前设置的随机种子。
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int 类型，当前为 {type(seed).__name__}")

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = benchmark

        if use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

    except ImportError:
        pass

    return seed