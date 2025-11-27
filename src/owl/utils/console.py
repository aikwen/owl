import datetime
from colorama import Fore, Style, init

init(autoreset=True)

LOGO_TEXT = r"""
                 __
  ____ _      __/ /
 / __ \ | /| / / /
/ /_/ / |/ |/ / /
\____/|__/|__/_/
"""

def generate_prefix() -> str:
    """
    生成带颜色的时间戳和 INFO 标签
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 亮黑色时间 - 亮绿色 INFO
    return (
        f"{Fore.LIGHTBLACK_EX}{current_time}{Style.RESET_ALL} - "
        f"{Fore.LIGHTGREEN_EX}INFO{Style.RESET_ALL} - "
    )


def highlight(s_format: str, *words: str, with_prefix: bool = False):
    """
    高亮关键字并打印
    :param s_format: 格式化字符串，如 "Loading {}..."
    :param words: 要填入并高亮的词
    :param with_prefix: 是否显示时间前缀
    """
    colored_words = [f"{Fore.CYAN}{w}{Style.RESET_ALL}" for w in words]
    s = s_format.format(*colored_words)
    if with_prefix:
        s = generate_prefix() + s
    print(s)


def welcome():
    """
    打印欢迎词
    """
    from .. import __version__
    print(f"{Fore.CYAN}{LOGO_TEXT}{Style.RESET_ALL}")
    highlight("🚀 {} v{} is starting...", "owl engine", __version__, with_prefix=True)


def stop():
    highlight("{} is stopped!", "owl engine", with_prefix=True)