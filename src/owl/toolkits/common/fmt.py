from prettytable import PrettyTable


def format_metrics_table(all_metrics: dict[str, dict[str, float]], current_epoch: int) -> str:
    """使用 PrettyTable 生成纯 ASCII 指标表格，确保日志文件无乱码。"""
    if not all_metrics:
        return ""

    table = PrettyTable()

    # 获取指标名称 (例如: AUC, F1)
    first_ds = list(all_metrics.keys())[0]
    metric_keys = list(all_metrics[first_ds].keys())

    # 设置表头列名
    table.field_names = ["DATASET"] + [k.upper() for k in metric_keys]

    for ds_name, metrics in all_metrics.items():
        row_values = [ds_name]
        for k in metric_keys:
            val = metrics.get(k, 0.0)
            # 保持 4 位小数格式化
            row_values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        table.add_row(row_values)

    table.align = "c"

    lines = [
        f"\nEpoch [{current_epoch}] Summary",
        table.get_string(),
        ""
    ]

    return "\n".join(lines) + "\n"

def format_zero_pad(value: int, max_val: int) -> str:
    """
    对数字进行前导补 0 格式化。
    对齐宽度由 max_val 的位数决定。

    Args:
        value (int): 当前值。
        max_val (int): 允许达到的最大值, 用于确定宽度。

    Returns:
        str: 格式化后的字符串。例如：value=1, max_val=100 -> "001"
    """
    # 计算最大值的位数作为宽度
    width = len(str(max_val))
    return f"{value:0{width}d}"