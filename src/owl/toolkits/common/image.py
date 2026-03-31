import numpy as np
from PIL import Image

def rgba2rgb(img: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
    """
    将 rgba 类型图转换成 rgb 图
    :param img:
    :param background_color:
    :return:
    """
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, background_color)
        background.paste(img, mask=img.split()[3])
        return background
    else:
        return img.convert('RGB')


def to_numpy(img: Image.Image, ensure_rgb: bool = False) -> np.ndarray:
    """
    将 PIL Image 转换为 Numpy Array。
    Args:
        img: PIL 图像
        ensure_rgb: 如果为 True，会强制检查并处理 RGBA/透明背景，转换为 RGB 格式。
    Returns:
        np.ndarray: 图像矩阵
    """
    if ensure_rgb:
        img = rgba2rgb(img)

    return np.array(img)


def normalize_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将 GT 掩码处理为标准的 0/1 概率矩阵 (float32)。

    1. 如果 > 1.0，说明是 0-255 格式，除以 255 归一化。
    2. 强制二值化：大于阈值的设为 1.0，其余为 0.0 (消除插值产生的中间值)。

    Args:
        mask: 输入掩码 (uint8 或 float)
        threshold: 二值化阈值，默认 0.5
    Returns:
        np.ndarray: float32 类型的 0/1 矩阵
    """
    # 转 float32，避免整数运算丢失精度
    mask = mask.astype(np.float32)

    # 归一化 只有当数值明显超出 0-1 范围时才除以 255
    if mask.max() > 1.0:
        mask /= 255.0

    # 3. 二值化 (Binarization)
    return (mask > threshold).astype(np.float32)