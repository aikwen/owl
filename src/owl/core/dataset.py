from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from typing import Union, Dict, List, Optional
import albumentations as albu
import pathlib
import torch
from ..utils import validator
from ..utils import file_io
from ..utils import img_op

class ImageDataset(Dataset):
    """
    加载 tp 和 gt 图像
    """
    def __init__(self, path:Union[str, pathlib.Path],
                 transform:Optional[albu.Compose]=None):
        """
        Args:
            path (Union[str, pathlib.Path]): 数据集路径
            postprocess (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
                A function to post-process the images and masks after loading.
        """
        # 数据集路径
        self.path = validator.data_protocol(path)
        # 数据集列表
        self.dataset_list:List[Dict] = file_io.load_json(self.path / f"{self.path.name}.json")
        # 数据集要做的变换
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
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
        tp_image:Image.Image = file_io.load_image(tp_img_path)
        if gt_img_path is  None:
            gt_image:Image.Image = Image.new("L", tp_image.size, 0)
        else:
            gt_image:Image.Image = file_io.load_image(gt_img_path).convert("L")

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
        return (tp_tensor,
                gt_tensor,
                tp_name,
                gt_name)

def create_dataloader(dataset_list:List[pathlib.Path],
                    transform:Optional[albu.Compose],
                    batchsize:int,
                    num_workers:int=0,
                    shuffle:bool=True,
                    )-> DataLoader:
    datasets = []
    for path in dataset_list:
        datasets.append(ImageDataset(path, transform=transform))
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset,
                      batch_size=batchsize,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      persistent_workers=(num_workers > 0),)