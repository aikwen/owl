try:
    from importlib.metadata import version
    __version__ = version("owl-imdl")
except ImportError:
    # 为了兼容旧版 Python 或未安装的情况，提供一个回退值
    __version__ = "unknown"