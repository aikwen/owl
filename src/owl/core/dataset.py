from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from typing import Union, List, Optional, Any
import albumentations as A
import pathlib
import torch

from .schemas import DataSetItem
from ..utils import validator, io, img_op
from ..augment import transforms, types


class ImageDataset(Dataset):
    """
    加载 tp 和 gt 图像
    """

    def __init__(self, path: Union[str, pathlib.Path],
                 transform: Optional[A.Compose] = None):
        """ImageDataset

        Args:
            path (Union[str, pathlib.Path]): 数据集路径
            transform (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
                A function to post-process the images and masks after loading.
        """
        # 数据集路径
        self.path = validator.data_protocol(path)
        # 数据集列表
        self.dataset_list: List[dict] = io.load_json(self.path / f"{self.path.name}.json")
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
        tp_image: Image.Image = io.load_image(tp_img_path)
        gt_image: Image.Image

        if gt_img_path is None:
            gt_image = Image.new("L", tp_image.size, 0)
        else:
            gt_image = io.load_image(gt_img_path).convert("L")

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
                      transform: Optional[A.Compose],
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
            transform=transforms.aug_compose(self.transform_pipeline),
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
                transform=transforms.aug_compose(self.transform_pipeline),
                batchsize=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        return dataloader_test


def build_owlDataloader(config: dict[str, Any]) -> OwlDataloader:
    """根据嵌套配置字典构建数据加载器实例。
    适配的 YAML 结构::
        datasets_train:
          path:
            - "example"
          batchsize: 12
          epochs: 30
          transform_pipeline:
            - {type: rotate, param: {p: 0.5}}
            - {type: vflip,  param: {p: 0.5}}
            - {type: hflip,  param: {p: 0.5}}
            - {type: jpeg,   param: {p: 0.3, quality_low: 70, quality_high: 100}}
            - {type: gblur,  param: {p: 0.3, kernel_low: 3, kernel_high: 15}}
            - {type: resize, param: {width: 512, height: 512}}
          other:
            num_workers: 4
            shuffle: true
            pin_memory: true
            persistent_workers: true

        datasets_validate:
          path:
            - "example"
          batchsize: 1
          transform_pipeline:
            - {type: resize, param: {p: 1, width: 512, height: 512}}
          custom_validate_func: "model.model.validate"
          other:
            num_workers: 1
            shuffle: false
            pin_memory: true
            persistent_workers: true
    Args:
        config (dict[str, Any]): 配置字典，必须包含 path, batchsize, other, transform_pipeline。

    Returns:
        OwlDataloader: 配置好的数据加载器封装对象。

    Raises:
        KeyError: 如果配置中缺少必要的字段。
    """
    # 获取所有字段
    try:
        path_list = config["path"]
        batch_size = config["batchsize"]
        transform_pipeline_config = config["transform_pipeline"]
        other_config = config["other"]

        # 二级严格校验
        num_workers = other_config["num_workers"]
        shuffle = other_config["shuffle"]
        pin_memory = other_config["pin_memory"]
        persistent_workers = other_config["persistent_workers"]
    except KeyError as e:
        raise KeyError(f"配置文件校验失败: 缺少必填字段 {e}。请在 YAML 中显式定义该字段，如不需要可设为 null 或空列表。")

    # 加载数据集路径，支持重复
    datasets_map: dict[str, pathlib.Path] = {}
    for p_str in path_list:
        p = pathlib.Path(p_str).resolve()

        # 初始 Key 为文件夹名
        key_name = p.name

        # 冲突检测与重命名逻辑
        # 如果 datasets_map 中已经有了这个 key (说明用户在 list 中写了多次，或者不同路径同文件夹名)
        # 我们给它加上后缀 _1, _2, _3 ... 直到唯一
        if key_name in datasets_map:
            original_name = key_name
            counter = 1
            while key_name in datasets_map:
                key_name = f"{original_name}_{counter}"
                counter += 1
            # 此时 key_name 已经是唯一的了，例如 "nist16_1"

        # 存入字典
        datasets_map[key_name] = p

    # 构建增强流水线
    transform_pipeline: list[types.BaseAugConfig] = []
    # 即使是空列表，代码也能正常运行，但必须要是 List 类型
    if not isinstance(transform_pipeline_config, list):
        raise TypeError(f"transform_pipeline 必须是列表格式，当前为: {type(transform_pipeline_config)}")

    for ts in transform_pipeline_config:
        transform_pipeline.append(transforms.dict2config(ts))

    return OwlDataloader(
        datasets_map=datasets_map,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        transform_pipeline=transform_pipeline,
    )
