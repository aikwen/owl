from torch import nn
from ..utils import reflect

def build_model(class_path: str) -> nn.Module:
    """反射构建神经网络模型实例。

    仅支持字符串路径配置，且要求 Model 类必须支持无参构造。

    Example:
        model: "projects.my_project.models.SimpleNet"

    Args:
        class_path (str): 类的完整点分路径。

    Returns:
        nn.Module: 初始化后的 PyTorch 模型实例。
    """
    if not isinstance(class_path, str):
        raise TypeError(f"Model 配置错误：必须是字符串路径，实际收到 {type(class_path)}。")

    # 反射加载类
    # 强制要求加载的类必须是 torch.nn.Module 的子类
    cls = reflect.load_class(class_path, base_class=nn.Module)

    # 无参实例化
    try:
        instance = cls()
    except TypeError as e:
        # 捕获用户定义的类如果包含必填参数导致的错误
        raise ValueError(
            f"实例化 Model 类 '{class_path}' 失败。\n"
            f"Owl 框架当前要求 Model 类必须支持无参构造 (__init__ 不应有必填参数)。\n"
            f"底层错误: {e}"
        )

    return instance