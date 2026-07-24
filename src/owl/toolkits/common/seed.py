from __future__ import annotations

import random


def seed_everything(
    seed: int = 3407,
    deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = False,
    warn_only: bool = True,
) -> int:
    """固定常见随机源，尽可能提高实验可复现性。

    本函数会设置：

    1. Python 内置 random 的随机种子；
    2. NumPy 的随机种子，如果当前环境安装了 NumPy；
    3. PyTorch CPU 随机种子；
    4. PyTorch CUDA 随机种子，如果当前环境可用 CUDA；
    5. cuDNN 的 deterministic / benchmark 行为；
    6. 可选的 PyTorch 确定性算法模式。

    注意：
        完全可复现通常无法跨 PyTorch 版本、CUDA 版本、cuDNN 版本、
        操作系统和硬件平台保证。

        本函数主要用于减少同一环境、同一代码、同一数据配置下的随机性。

    关于 PYTHONHASHSEED：
        PYTHONHASHSEED 会影响 Python 哈希随机化，例如部分 dict / set
        相关行为。它最好在 Python 进程启动前通过环境变量设置。

        不建议在本函数内部强行设置 os.environ["PYTHONHASHSEED"]，
        因为函数被调用时 Python 进程已经启动，部分哈希行为可能已经发生。

        Linux / macOS 示例：
            PYTHONHASHSEED=3407 python train.py

        Windows PowerShell 示例：
            $env:PYTHONHASHSEED="3407"
            python train.py

    关于 deterministic 和 benchmark：
        deterministic=True 会要求 cuDNN 尽量使用确定性算法，
        有利于实验复现。

        benchmark=True 会让 cuDNN 根据输入尺寸自动搜索更快的卷积算法，
        有利于性能，但可能降低复现稳定性。

        如果目标是复现，推荐：
            deterministic=True
            benchmark=False

        如果目标是性能，且不强求严格复现，可以考虑：
            deterministic=False
            benchmark=True

    Args:
        seed:
            随机种子。默认使用 3407。
        deterministic:
            是否开启 cuDNN 确定性模式。
        benchmark:
            是否开启 cuDNN benchmark 自动算法搜索。
        use_deterministic_algorithms:
            是否调用 torch.use_deterministic_algorithms(True)。
            该选项比 cuDNN deterministic 更严格。
            如果某些 PyTorch 算子没有确定性实现，可能触发警告或异常。
        warn_only:
            当 use_deterministic_algorithms=True 时，如果遇到非确定性算子，
            是否只发出警告而不是直接抛出异常。

    Returns:
        当前设置的随机种子。
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
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = benchmark

        if use_deterministic_algorithms:
            torch.use_deterministic_algorithms(
                True,
                warn_only=warn_only,
            )

    except ImportError:
        pass

    return seed