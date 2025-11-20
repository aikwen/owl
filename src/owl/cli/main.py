import argparse
from .commands import init_command

def main():
    parser = argparse.ArgumentParser(
        description="Owl - Deep Learning Training Framework CLI",
        prog="owl"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="可用命令")

    # === 子命令: init ===
    parser_init = subparsers.add_parser("init", help="生成标准训练脚本模板")
    parser_init.add_argument(
        "filename",
        nargs="?",
        default="train.py",
        help="生成的脚本文件名 (默认: train.py)"
    )
    parser_init.set_defaults(func=init_command)

    # 解析并执行
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()