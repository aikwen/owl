from io import StringIO
from rich.table import Table
from rich.console import Console
from rich import box


def format_metrics_table(all_metrics: dict[str, dict[str, float]], current_epoch: int) -> str:
    """Format evaluation metrics into an ASCII table with center alignment."""
    if not all_metrics:
        return ""

    table = Table(
        box=box.SQUARE,
        header_style="bold",
        show_edge=True
    )

    # 1. 第一列（数据集名称）居中对齐
    table.add_column("DATASET", justify="center", style="cyan", no_wrap=True)

    first_ds = list(all_metrics.keys())[0]
    metric_keys = list(all_metrics[first_ds].keys())
    for k in metric_keys:
        # 2. 所有指标列（AUC, F1 等）也全部居中对齐
        table.add_column(k.upper(), justify="center", style="green")

    for ds_name, metrics in all_metrics.items():
        row_values = [ds_name]
        for k in metric_keys:
            val = metrics.get(k, 0.0)
            row_values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        table.add_row(*row_values)

    string_io = StringIO()
    capture_console = Console(file=string_io, force_terminal=True)

    capture_console.print(f"[bold yellow]Epoch [{current_epoch}] Summary [/bold yellow]")
    capture_console.print(table)

    return f"\n{string_io.getvalue()}"