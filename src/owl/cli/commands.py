from pathlib import Path
from .templates import TRAIN_TEMPLATE

def init_command(args):
    """处理 init 命令的具体逻辑"""
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