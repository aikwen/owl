import albumentations as albu
from typing import List, Optional, Union
from .types import (BaseAugConfig,
                    RotateConfig,
                    VFlipConfig,
                    HFlipConfig,
                    ResizeConfig,
                    JpegConfig,
                    GblurConfig,
                    GNoiseConfig,
                    ScaleConfig)

def dict2config(t: dict) -> BaseAugConfig:
    config_type = t['type']
    param = t['param']

    if config_type == "rotate":
        return RotateConfig(**param)
    elif config_type == "vflip":
        return VFlipConfig(**param)
    elif config_type == "hflip":
        return HFlipConfig(**param)
    elif config_type == "jpeg":
        return JpegConfig(**param)
    elif config_type == "gblur":
        return GblurConfig(**param)
    elif config_type == "gnoise":
        return GNoiseConfig(**param)
    elif config_type == "scale":
        return ScaleConfig(**param)
    elif config_type == "resize":
        param.pop('p', None)
        return ResizeConfig(p=1.0, **param)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

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