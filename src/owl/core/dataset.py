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
    tp_tensor: torch.Tensor
    gt_tensor: torch.Tensor
    tp_name: str
    gt_name: str

class DataSetBatch(TypedDict):
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
        """
        Args:
            path (Union[str, pathlib.Path]): 数据集路径
            postprocess (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
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
            tp (tensor) shape [channel, height, width], channel = 3
            gt (tensor) shape [channel, height, width], channel = 1
            tp_name (str) 图像名, 比如 tp.png
            gt_name (str)图像名， 比如 gt.png
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
                 transform_pipline: List[types.BaseAugConfig],
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool,
                 pin_memory: bool,
                 persistent_workers: bool):
        """
        :param datasets_map: 数据集字典，e.g., {"nist16":"path to nist16", "coverage":"path to coverage"}
        :param transform_pipline:
        :param batch_size:
        :param num_workers:
        :param shuffle:
        :param pin_memory:
        :param persistent_workers:
        """
        self.datasets_map: dict[str, pathlib.Path] = datasets_map
        self.transform_pipline: List[types.BaseAugConfig] = transform_pipline
        self.batch_size: int = batch_size
        self.num_worker: int = num_workers
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers

    def build_dataloader_train(self) -> DataLoader:
        paths = list(self.datasets_map.values())
        dataloader_train = create_dataloader(
            dataset_list=paths,
            transform=img_aug.aug_compose(self.transform_pipline),
            batchsize=self.batch_size,
            num_workers=self.num_worker,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader_train

    def build_dataloader_valid(self) -> dict[str, DataLoader]:
        dataloader_test: dict[str, DataLoader] = {}
        for k, v in self.datasets_map.items():
            dataloader_test[k] = create_dataloader(
                dataset_list=[v],
                transform=img_aug.aug_compose(self.transform_pipline),
                batchsize=self.batch_size,
                num_workers=self.num_worker,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        return dataloader_test
