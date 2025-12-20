from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from typing import Union, List, Optional
import albumentations as albu
import pathlib
import torch
from typing_extensions import TypedDict
from ..utils import validator, file_io, img_op, img_aug, types


class DataSetItem(TypedDict):
    """定义单条数据样本的字典结构。

    通常作为 Dataset.__getitem__ 的返回值。

    Attributes:
        tp_tensor (torch.Tensor): 篡改图像（Tampered Image）张量。
            Shape: [3, H, W] (RGB格式)
        gt_tensor (torch.Tensor): 真实标签（Ground Truth）张量。
            Shape: [1, H, W] (Mask格式)
        tp_name (str): 篡改图像的文件名（例如 "001_t.png"）。
        gt_name (str): 对应标签的文件名（例如 "001_mask.png"）。
    """
    tp_tensor: torch.Tensor
    gt_tensor: torch.Tensor
    tp_name: str
    gt_name: str


class DataSetBatch(TypedDict):
    """定义一个 Batch 的数据字典结构。

    通常作为 DataLoader 迭代出的对象，由 collate_fn 堆叠 DataSetItem 而成。

    Attributes:
        tp_tensor (torch.Tensor): 批次级篡改图像张量。
            Shape: [B, 3, H, W]，其中 B 为 Batch Size。
        gt_tensor (torch.Tensor): 批次级真实标签张量。
            Shape: [B, 1, H, W]。
        tp_name (list[str]): 当前 Batch 中所有样本的图像文件名列表。
            列表长度等于 Batch Size。
        gt_name (list[str]): 当前 Batch 中所有样本的标签文件名列表。
    """
    tp_tensor: torch.Tensor  # [B, 3, H, W]
    gt_tensor: torch.Tensor  # [B, 1, H, W]
    tp_name: list[str]
    gt_name: list[str]


class ImageDataset(Dataset):
    """
    加载 tp 和 gt 图像
    """

    def __init__(self, path: Union[str, pathlib.Path],
                 transform: Optional[albu.Compose] = None):
        """ImageDataset

        Args:
            path (Union[str, pathlib.Path]): 数据集路径
            transform (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
                A function to post-process the images and masks after loading.
        """
        # 数据集路径
        self.path = validator.data_protocol(path)
        # 数据集列表
        self.dataset_list: List[dict] = file_io.load_json(self.path / f"{self.path.name}.json")
        # 数据集要做的变换
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx) -> DataSetItem:
        """
        Args:
            idx (int): 对应的下标

        Returns:
            DataSetItem: 包含图像张量和文件名的样本字典。
                具体字段定义请参考 `types.DataSetItem`。
        """
        # 加载对应 idx 的 tp 图像路径
        tp_img_path = self.path.joinpath("tp", self.dataset_list[idx]["tp"])
        gt_img_path = None
        if self.dataset_list[idx]["gt"] != "":
            gt_img_path = self.path.joinpath("gt", self.dataset_list[idx]["gt"])

        # 加载tp 和 gt 图像
        tp_image: Image.Image = file_io.load_image(tp_img_path)
        gt_image: Image.Image

        if gt_img_path is None:
            gt_image = Image.new("L", tp_image.size, 0)
        else:
            gt_image = file_io.load_image(gt_img_path).convert("L")

        # 获取图像矩阵
        tp_array = img_op.to_numpy(tp_image, ensure_rgb=True)
        gt_array = img_op.to_numpy(gt_image, ensure_rgb=False)

        # 对 tp， gt 矩阵进行增强
        if self.transform is not None:
            transformed = self.transform(image=tp_array, mask=gt_array)
            tp_array = transformed['image']
            gt_array = transformed['mask']

        # 将 gt 矩阵变成概率矩阵
        gt_array = img_op.normalize_binary_mask(gt_array, 0.5)

        # to tensor
        tp_tensor = torch.from_numpy(tp_array).to(torch.float32).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt_array).to(torch.float32).unsqueeze(0)
        tp_name = tp_img_path.name
        gt_name = gt_img_path.name if gt_img_path is not None else ""
        return DataSetItem(
            tp_tensor=tp_tensor,
            gt_tensor=gt_tensor,
            tp_name=tp_name,
            gt_name=gt_name,
        )


def create_dataloader(dataset_list: List[pathlib.Path],
                      transform: Optional[albu.Compose],
                      batchsize: int,
                      num_workers: int = 0,
                      shuffle: bool = True,
                      pin_memory: bool = True,
                      persistent_workers: bool = True,
                      ) -> DataLoader:
    """创建包含多个数据集的 DataLoader。

    该函数会自动遍历 `dataset_list` 中的所有路径，为每个路径创建一个 ImageDataset，
    然后使用 ConcatDataset 将它们合并为一个大的数据集。

    Args:
        dataset_list: 包含数据集根目录路径的列表。
        transform: Albumentations 数据增强流水线对象（可选）。
        batchsize: 批次大小。
        num_workers: 数据加载子进程数量。默认为 0（主进程加载）。
        shuffle: 是否在每个 Epoch 开始时打乱数据。默认为 True。
        pin_memory: 是否使用 CUDA 锁页内存。
            注意：代码会自动检测 `torch.cuda.is_available()`，如果无 GPU，
            即使设为 True 也会强制使用 False。
        persistent_workers: 是否保持子进程存活。
            注意：仅当 `num_workers > 0` 时生效。如果 `num_workers=0`，
            即使设为 True 也会强制使用 False。

    Returns:
        DataLoader: 包装了所有合并数据集的 PyTorch DataLoader 实例。
    """
    datasets = []
    for path in dataset_list:
        datasets.append(ImageDataset(path, transform=transform))
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset,
                      batch_size=batchsize,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      persistent_workers=(num_workers > 0) and persistent_workers,
                      pin_memory=(pin_memory if torch.cuda.is_available() else False), )


class OwlDataloader:
    def __init__(self, datasets_map: dict[str, pathlib.Path],
                 transform_pipeline: List[types.BaseAugConfig],
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool,
                 pin_memory: bool,
                 persistent_workers: bool):
        """初始化 OwlDataloader 配置封装。

        Args:
            datasets_map: 数据集名称到路径的映射字典。
                例如: {"nist16": Path("data/nist16"), "coverage": Path("data/coverage")}。
            transform_pipeline: 数据增强配置列表，定义了图像预处理和增强的流水线。
            batch_size: 每个批次的样本数量。
            num_workers: 用于数据加载的子进程数量。
                0 表示在主进程中加载数据；大于 0 表示使用多进程加载。
            shuffle: 是否打乱数据顺序。
                通常训练集设为 True，验证/测试集设为 False。
            pin_memory: 是否将 Tensor 放入 CUDA 锁页内存 (Pinned Memory)。
                如果你使用 GPU 训练，设为 True 可以加速 CPU 到 GPU 的数据传输。
            persistent_workers: 是否在一个 epoch 结束后保持 worker 进程存活。
                如果设为 True，可以减少每个 epoch 重新创建子进程的开销，但会占用更多内存。

        Examples:
            transform_pipeline 的配置示例::

                [
                    types.RotateConfig(p=0.5),
                    types.VFlipConfig(p=0.5),
                    types.HFlipConfig(p=0.5),
                    types.JpegConfig(quality_low=70, quality_high=100, p=0.3),
                    types.GblurConfig(kernel_low=3, kernel_high=15, p=0.3),
                    types.ResizeConfig(width=512, height=512, p=1),
                ]
        """
        self.datasets_map: dict[str, pathlib.Path] = datasets_map
        self.transform_pipeline: List[types.BaseAugConfig] = transform_pipeline
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers

    def build_dataloader_train(self) -> DataLoader:
        """构建训练用的 DataLoader。

        将 `datasets_map` 中的所有数据集路径合并为一个大的 ConcatDataset，
        并应用统一的增强流水线。

        Returns:
            配置好的 PyTorch DataLoader 对象，包含合并后的所有训练数据。
        """
        paths = list(self.datasets_map.values())
        dataloader_train = create_dataloader(
            dataset_list=paths,
            transform=img_aug.aug_compose(self.transform_pipeline),
            batchsize=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader_train

    def build_dataloader_valid(self) -> dict[str, DataLoader]:
        """构建验证用的 DataLoader 字典。

        为 `datasets_map` 中的每一个数据集单独创建一个 DataLoader，
        以便能够独立评估模型在不同数据集上的性能指标。

        Returns:
            一个字典，Key 为数据集名称 (与 datasets_map 的 Key 一致)，
            Value 为对应的 PyTorch DataLoader 对象。
        """
        dataloader_test: dict[str, DataLoader] = {}
        for k, v in self.datasets_map.items():
            dataloader_test[k] = create_dataloader(
                dataset_list=[v],
                transform=img_aug.aug_compose(self.transform_pipeline),
                batchsize=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        return dataloader_test
