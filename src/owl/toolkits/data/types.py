from typing import TypedDict, TypeAlias, Any

BatchData: TypeAlias = dict[str, Any]
"""DataLoader 输出的通用 batch 类型。"""

class DataLoaderConfig(TypedDict):
    """DataLoader 的配置参数字典。

    Attributes:
        batch_size (int): 每个 Batch 的样本数。默认值为 1。
        num_workers (int): 用于数据加载的子进程数。0 表示在主进程中加载。
        shuffle (bool): 是否在每个 Epoch 开始时打乱数据。
        pin_memory (bool): 是否将 Tensor 拷贝到 CUDA 固定内存中，加速 GPU 加载。
        persistent_workers (bool): 训练结束后是否保留子进程，能减少每个 Epoch 开始时的初始化耗时。
        drop_last (bool): 如果数据集大小不能被 batch_size 整除，是否丢弃最后一个不完整的 Batch。
    """
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool