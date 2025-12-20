from pathlib import Path
from .templates import TRAIN_TEMPLATE

def init_command(args):
    """执行 init 命令逻辑，生成标准训练脚本模板。

    核心流程:
        1. **后缀补全**：如果用户输入的文件名没有 `.py`，自动补全。
        2. **冲突检测**：检查文件是否存在。
           - 如果存在，发起交互式询问 (y/n) 确认是否覆盖。
           - 如果用户拒绝，直接返回。
        3. **文件写入**：将预定义的 `TRAIN_TEMPLATE` 写入目标路径。

    Args:
        args: 命令行参数命名空间 (通常来自 argparse)。
            必须包含 `filename` (str) 属性，表示目标文件路径。

    Side Effects:
        - 在磁盘上创建新文件或覆盖旧文件。
        - 向 stdout 打印进度和错误信息。
        - 若文件存在，会阻塞程序等待用户输入 (stdin)。
    """
    target_path = Path(args.filename)

    # 自动补全后缀
    if target_path.suffix != ".py":
        target_path = target_path.with_suffix(".py")

    if target_path.exists():
        print(f"⚠️ 文件已存在: {target_path}")
        overwrite = input("❓ 是否覆盖? (y/[n]): ").lower().strip()
        if overwrite != 'y':
            print("❌ 操作已取消。")
            return

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(TRAIN_TEMPLATE.strip())
        print(f"✅ 成功生成训练脚本: {target_path}")
    except Exception as e:
        print(f"❌ 生成失败: {e}")

def version_command(args):
    from .. import __version__
    print(__version__)