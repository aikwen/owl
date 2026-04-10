import pathlib
from typing import Union
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .types import DataLoaderConfig
from .dataset import ImageDataset
from .collectors.owl import from_owl_json
from .augment.transforms import aug_compose
from .augment.types import BaseAugConfig


class OwlDataLoader:
    """
    统一的数据加载管理器。
    """

    def __init__(self,
                 dataset_paths: list[Union[str, pathlib.Path]],
                 dataloader_config: DataLoaderConfig,
                 aug_configs: list[BaseAugConfig] | None,
                 ):

        self.config = dataloader_config
        self.transform = aug_compose(aug_configs) if aug_configs else None

        self.datasets_map: dict[str, ImageDataset] = {}
        self._init_datasets(dataset_paths)

    def _init_datasets(self, dataset_paths: list[Union[str, pathlib.Path]]):
        for p in dataset_paths:
            path = pathlib.Path(p).resolve()

            key_name = path.name
            original_name = key_name
            counter = 1
            while key_name in self.datasets_map:
                key_name = f"{original_name}_{counter}"
                counter += 1

            dataset = ImageDataset(
                root_dir=path,
                collector_fn=from_owl_json,
                transform=self.transform
            )
            self.datasets_map[key_name] = dataset

    @property
    def num_datasets(self) -> int:
        """获取传入的数据集文件夹数量"""
        return len(self.datasets_map)

    @property
    def total_samples(self) -> int:
        """获取所有数据集包含的图片样本总数"""
        return sum(len(ds) for ds in self.datasets_map.values())

    @property
    def train_batches(self) -> int:
        """合并后训练集的总 batch 数量"""
        batch_size = self.config.get("batch_size", 1)
        drop_last = self.config.get("drop_last", False)

        if drop_last:
            return self.total_samples // batch_size
        else:
            return (self.total_samples + batch_size - 1) // batch_size

    def get_dataset_info(self) -> dict[str, int]:
        """返回各个数据集的具体样本数量"""
        return {name: len(ds) for name, ds in self.datasets_map.items()}


    def get_train_loader(self) -> DataLoader:
        """合并所有已实例化的 Dataset 为一个大 DataLoader
        """
        combined_dataset = ConcatDataset(list(self.datasets_map.values()))

        num_workers = self.config.get("num_workers", 0)
        batch_size = self.config.get("batch_size", 1)
        use_pin_memory = self.config.get("pin_memory", False) if torch.cuda.is_available() else False
        use_persistent = self.config.get("persistent_workers", False) if num_workers > 0 else False
        use_shuffle = self.config.get("shuffle", False)
        use_drop_last = self.config.get("drop_last", False)

        return DataLoader(
            dataset=combined_dataset,
            batch_size=batch_size,
            shuffle=use_shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
            drop_last=use_drop_last,
        )

    def get_valid_loaders(self) -> dict[str, DataLoader]:
        """
        验证模式：保持数据集独立，返回 Dict[str, DataLoader]
        """
        loaders = {}
        num_workers = self.config.get("num_workers", 0)
        batch_size = self.config.get("batch_size", 1)
        use_pin_memory = self.config.get("pin_memory", False) if torch.cuda.is_available() else False
        use_persistent = self.config.get("persistent_workers", False) if num_workers > 0 else False
        use_shuffle = False
        use_drop_last = False

        for name, dataset in self.datasets_map.items():
            loaders[name] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=use_shuffle,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                persistent_workers=use_persistent,
                drop_last=use_drop_last
            )
        return loaders