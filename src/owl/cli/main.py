import argparse
from . import cmd_version, cmd_init, cmd_stats

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
    # 子命令: stats (训练监控)
    # ==========================================
    parser_stats = subparsers.add_parser(
        name="stats",
        help="连接训练监控服务并实时查看状态",
    )

    parser_stats.add_argument(
        "address",
        help='monitor 地址或端口号，例如 "127.0.0.1:39125" 或 "39125"',
    )

    parser_stats.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="stream 输出间隔，单位秒；0 表示每条都打印",
    )

    parser_stats.add_argument(
        "-r",
        "--retry",
        type=int,
        default=5,
        help="连接失败后的重试次数，每秒重试一次；0 表示不重试",
    )
    parser_stats.set_defaults(func=cmd_stats.func)

    # ==========================================
    # 解析参数并执行对应的函数
    # ==========================================
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()