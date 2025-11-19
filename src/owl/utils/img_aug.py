from dataclasses import dataclass
import albumentations as albu
from typing import List, Optional, Union


@dataclass
class BaseAugConfig:
    """
    p 概率值, 0-1之间
    """
    p: float

@dataclass
class RotateConfig(BaseAugConfig):
    """
    随机旋转
    """
    pass

@dataclass
class VFlipConfig(BaseAugConfig):
    """
    垂直反转
    """
    pass

@dataclass
class HFlipConfig(BaseAugConfig):
    """
    水平反转
    """
    pass

@dataclass
class ResizeConfig(BaseAugConfig):
    """
    改变尺寸
    """
    height: int
    width: int

@dataclass
class JpegConfig(BaseAugConfig):
    """
    jpeg 压缩
    随机在 [quality_low, quality_high] 区间之间选择一个质量因子进行压缩
    """
    quality_low: int
    quality_high: int

@dataclass
class GblurConfig(BaseAugConfig):
    """
    高斯模糊
    随机在 [kernel_low, kernel_high] 区间之间选择一个 kernal size
    """
    kernel_low: int
    kernel_high: int

@dataclass
class GNoiseConfig(BaseAugConfig):
    """
    高斯噪声
    """
    std_low: float
    std_high: float

@dataclass
class ScaleConfig(BaseAugConfig):
    """
    随机在 (1+scale1, 1+scale2) 之间旋转一个值进行缩放
    """
    scale1: float
    scale2: float

def config2transform(cfg: BaseAugConfig) -> albu.BasicTransform:
    """
    将 config 转化成对于的具体转换
    :param cfg:
    :return:
    """
    if isinstance(cfg, RotateConfig):
        return albu.RandomRotate90(p=cfg.p)

    elif isinstance(cfg, VFlipConfig):
        return albu.VerticalFlip(p=cfg.p)

    elif isinstance(cfg, HFlipConfig):
        return albu.HorizontalFlip(p=cfg.p)

    elif isinstance(cfg, ResizeConfig):
        return albu.Resize(height=cfg.height, width=cfg.width, p=cfg.p)

    elif isinstance(cfg, JpegConfig):
        return albu.ImageCompression(quality_range=(cfg.quality_low,
                                                    cfg.quality_high), p=cfg.p)
    elif isinstance(cfg, GblurConfig):
        return albu.GaussianBlur(blur_limit=(cfg.kernel_low,
                                             cfg.kernel_high), p=cfg.p)
    elif isinstance(cfg, GNoiseConfig):
        return albu.GaussNoise(std_range=(cfg.std_low,
                                          cfg.std_high), p=cfg.p)
    elif isinstance(cfg, ScaleConfig):
        return albu.RandomScale(scale_limit=(cfg.scale1,
                                             cfg.scale2), p=cfg.p)
    else:
        return albu.NoOp(p=1)

AugmentItemType = Union[albu.BasicTransform, BaseAugConfig]


def aug_compose(aug_list: List[AugmentItemType]) -> Optional[albu.Compose]:
    """
    获取转换组合
    :param aug_list:
    :return:
    """
    if aug_list is None or len(aug_list) <= 0:
        return None

    l : List[albu.BasicTransform] = []

    for aug in aug_list:
        if isinstance(aug, albu.BasicTransform):
            l.append(aug)
        elif isinstance(aug, BaseAugConfig):
            l.append(config2transform(aug))
        else:
            pass
    return albu.Compose([item for item in l])

