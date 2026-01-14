import argparse
from . import cmd_version, cmd_init, cmd_run

def main():
    parser = argparse.ArgumentParser(
        description="Owl - IMDL Deep Learning Training Framework CLI",
        prog="owl"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="可用命令")

    # ==========================================
    # 子命令: init (初始化)
    # ==========================================
    parser_init = subparsers.add_parser(
        name="init",
        help="初始化项目配置模板"
    )
    # 绑定处理函数
    parser_init.set_defaults(func=cmd_init.func)

    # ==========================================
    # 子命令: version (版本)
    # ==========================================
    parser_version = subparsers.add_parser(
        name="version",
        help="查看版本号")
    parser_version.set_defaults(func=cmd_version.func)

    # ==========================================
    # 子命令: run (运行配置文件)
    # ==========================================
    parser_run = subparsers.add_parser(
        name="run",
        help="运行配置文件")

    # 添加一个位置参数
    parser_run.add_argument(
        "file",  # 参数名
        type=str,  # 类型
        help="配置文件路径 (例如: train.yaml)"
    )

    parser_run.set_defaults(func=cmd_run.func)

    # ==========================================
    # 解析并执行
    # ==========================================
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()