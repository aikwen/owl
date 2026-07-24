from dataclasses import dataclass

@dataclass
class BaseAugConfig:
    """
    p 概率值, 0-1之间
    """
    p: float

    def __post_init__(self):
        if self.p < 0 or self.p > 1:
            raise ValueError(f"{self.p} 取值必须是 [0,1] 之间")

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

    def __post_init__(self):
        super().__post_init__()
        if self.quality_low > self.quality_high:
            raise ValueError("quality_low 应该小于 quality_high")

        if self.quality_high > 100 or self.quality_low < 0:
            raise ValueError("quality_high 和 quality_low 范围是[0, 100]")

@dataclass
class GblurConfig(BaseAugConfig):
    """
    高斯模糊
    随机在 [kernel_low, kernel_high] 区间之间选择一个 kernal size
    """
    kernel_low: int
    kernel_high: int

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_low > self.kernel_high:
            raise ValueError(
                f"GblurConfig Error: kernel_low ({self.kernel_low}) 不能大于 kernel_high ({self.kernel_high})")

        if self.kernel_low % 2 != 1 or self.kernel_high % 2 != 1:
            raise ValueError("kernel_low 和 kernel_high 必须是奇数")


@dataclass
class GNoiseConfig(BaseAugConfig):
    """
    高斯噪声 (Gaussian Noise)

    配置参数使用绝对像素标准差 (sigma)，如 3, 5, 7, 11, 15。

    计算说明 (基于官方文档)::

        Albumentations 要求传入 [0, 1] 的比例。对于 uint8 图像，内部会乘以 255。
        即: 实际噪声 sigma = 传入比例 * 255。
        因此，如果在此配置 sigma=15，底层会自动换算为 15/255 ≈ 0.058 传给框架。

    Attributes:
        sigma_low: 噪声标准差下限 (基于 0-255)。例如: 3.0
        sigma_high: 噪声标准差上限 (基于 0-255)。例如: 15.0
    """
    sigma_low: float = 3.0
    sigma_high: float = 15.0

    def __post_init__(self):
        super().__post_init__()

        if self.sigma_low < 0 or self.sigma_high < 0:
            raise ValueError("GNoise Error: sigma 必须大于等于 0")

        if self.sigma_low > self.sigma_high:
            raise ValueError(f"GNoise Error: sigma_low ({self.sigma_low}) 不能大于 sigma_high ({self.sigma_high})")

@dataclass
class ScaleConfig(BaseAugConfig):
    """
    随机在 (1+scale1, 1+scale2) 之间旋转一个值进行缩放
    """
    scale1: float
    scale2: float
