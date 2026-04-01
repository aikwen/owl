from typing import Any, Callable, Dict, Generic, TypeVar

# 定义一个泛型类型 T
T = TypeVar('T')

class Registry(Generic[T]):
    """泛型组件注册器。

    支持直接注册类（Class），也支持注册构建函数（Factory Function）。
    """

    def __init__(self, name: str):
        self._name = name
        # 存储名称到可调用对象（类或函数）的映射
        self._obj_map: Dict[str, Callable[..., T]] = {}

    def register(self, name: str|None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """装饰器：将类或构建函数注册到工厂中。

        把类或者函数放到 map 里面
        """

        def _register(obj: Callable[..., T]) -> Callable[..., T]:
            key = name if name is not None else obj.__name__
            if key in self._obj_map:
                raise ValueError(f"组件 '{key}' 已经在注册器 '{self._name}' 中被注册过了！")

            self._obj_map[key] = obj
            return obj

        return _register

    def get(self, name: str) -> Callable[..., T]:
        """获取已注册的类或构建函数"""
        if name not in self._obj_map:
            valid_list = list(self._obj_map.keys())
            raise ValueError(f"未知的组件类型: '{name}'。在 '{self._name}' 中的支持列表: {valid_list}")
        return self._obj_map[name]

    def build(self, obj_type: str, **kwargs) -> T:
        """组件构建

        Args:
            obj_type (str): 注册的组件名称（例如 "poly" 或 "adamw"）。
            **kwargs: 目标类或函数所需的任意参数。

        Returns:
            T: 实例化后的具体对象。
        """
        # 获取已注册的类或工厂函数
        build_func = self.get(obj_type)

        # kwargs 解包
        try:
            return build_func(**kwargs)
        except TypeError as e:
            raise ValueError(f"构建组件 '{obj_type}' 失败，参数不匹配: {e}")

    def __contains__(self, key: str) -> bool:
        return key in self._obj_map

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._obj_map.keys())})"