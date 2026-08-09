"""Inspect the current status of an Owl workspace."""

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..app import app


_POLL_INTERVAL_SECONDS = 0.5
_SPEED_SAMPLE_WINDOW = 3
_TITLE_MARGIN = 8
_TERMINAL_STATUSES = {"completed", "interrupted"}

_STATUS_STYLES: dict[str, tuple[str, str]] = {
    "running": ("● Running", "bold green"),
    "completed": ("● Completed", "bold cyan"),
    "interrupted": ("● Interrupted", "bold red"),
}

_READ_FAILED = object()


def _read_snapshot(
    path: Path,
) -> dict[str, Any] | object:
    """Read a snapshot file.

    A missing snapshot is represented by an empty dictionary because optional
    snapshots such as ``train.json`` may not exist.

    Read or JSON decoding failures are represented separately so the caller can
    preserve the currently displayed UI and retry on the next polling cycle.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return _READ_FAILED


def _compact_path(
    path: Path,
    max_width: int,
) -> str:
    """Compact a path from the left while preserving complete components."""
    resolved = path.resolve()
    full_path = str(resolved)

    if cell_len(full_path) <= max_width:
        return full_path

    parts = resolved.parts

    for index in range(1, len(parts)):
        suffix = os.sep.join(parts[index:])
        compact = f"...{os.sep}{suffix}"

        if cell_len(compact) <= max_width:
            return compact

    # Always preserve the workspace directory name as the final component.
    return f"...{os.sep}{resolved.name}"


def _status_text(status: str) -> Text:
    """Build styled lifecycle status text."""
    text, style = _STATUS_STYLES.get(
        status,
        ("-", "dim"),
    )
    return Text(text, style=style)


def _progress(
    completed: float,
    total: int,
    progress_text: str,
) -> Table:
    """Build a compact styled progress display."""
    completed = max(0.0, min(completed, total))
    percentage = completed / total * 100

    table = Table.grid(
        expand=True,
        padding=0,
    )
    table.add_column(width=1)
    table.add_column(ratio=1)
    table.add_column(width=1)
    table.add_column(width=16, justify="right")
    table.add_column(width=8, justify="right")

    table.add_row(
        Text("[", style="dim"),
        ProgressBar(
            total=total,
            completed=completed,
            style="grey37",
            complete_style="bright_blue",
            finished_style="green",
        ),
        Text("]", style="dim"),
        f"  {progress_text}",
        f"{percentage:5.1f}%",
    )

    return table


def _epoch_progress(
    epoch: dict[str, Any],
    batch: dict[str, Any],
) -> Table | Text:
    """Build epoch progress using the current batch position."""
    current_epoch = epoch.get("current")
    total_epoch = epoch.get("total")
    current_batch = batch.get("current")
    total_batch = batch.get("total")

    if (
        not isinstance(current_epoch, int)
        or not isinstance(total_epoch, int)
        or not isinstance(current_batch, int)
        or not isinstance(total_batch, int)
        or total_epoch <= 0
        or total_batch <= 0
    ):
        return Text("-", style="dim")

    completed = (
        current_epoch - 1
        + current_batch / total_batch
    )

    return _progress(
        completed=completed,
        total=total_epoch,
        progress_text=f"{current_epoch} of {total_epoch}",
    )


def _batch_progress(
    batch: dict[str, Any],
) -> Table | Text:
    """Build progress for the current training batch."""
    current = batch.get("current")
    total = batch.get("total")

    if (
        not isinstance(current, int)
        or not isinstance(total, int)
        or total <= 0
    ):
        return Text("-", style="dim")

    return _progress(
        completed=current,
        total=total,
        progress_text=f"{current} / {total}",
    )


def _training_position(
    train: dict[str, Any],
) -> tuple[int, float] | None:
    """Return the global training batch position and update timestamp."""
    epoch = train.get("epoch", {})
    batch = train.get("batch", {})

    if not isinstance(epoch, dict) or not isinstance(batch, dict):
        return None

    current_epoch = epoch.get("current")
    current_batch = batch.get("current")
    total_batch = batch.get("total")
    updated_at = train.get("updated_at")

    if (
        not isinstance(current_epoch, int)
        or not isinstance(current_batch, int)
        or not isinstance(total_batch, int)
        or not isinstance(updated_at, (int, float))
        or current_epoch <= 0
        or current_batch <= 0
        or total_batch <= 0
    ):
        return None

    global_batch = (
        (current_epoch - 1) * total_batch
        + current_batch
    )

    return global_batch, float(updated_at)


def _training_speed(
    samples: deque[tuple[int, float]],
) -> float | None:
    """Calculate average batch throughput across the sampled window."""
    if len(samples) < 2:
        return None

    first_batch, first_time = samples[0]
    last_batch, last_time = samples[-1]

    elapsed = last_time - first_time
    progressed = last_batch - first_batch

    if elapsed <= 0 or progressed < 0:
        return None

    return progressed / elapsed


def _build_view(
    title: str,
    lifecycle: dict[str, Any],
    train: dict[str, Any],
    speed: float | None = None,
) -> Panel:
    """Build the workspace status panel."""
    epoch = train.get("epoch", {})
    batch = train.get("batch", {})
    running = lifecycle.get("status") == "running"

    table = Table.grid(
        padding=(0, 2),
        expand=True,
    )

    table.add_column(
        width=10,
        style="bold",
    )
    table.add_column(ratio=1)

    table.add_row("", "")

    table.add_row(
        "Status",
        _status_text(str(lifecycle.get("status", ""))),
    )

    table.add_row(
        "Stage",
        str(lifecycle.get("stage", "-")).title(),
    )

    table.add_row("", "")

    table.add_row(
        "Epoch",
        _epoch_progress(
            epoch,
            batch,
        ),
    )

    table.add_row(
        "Batch",
        _batch_progress(batch),
    )

    speed_text = (
        f"{speed:.2f} batch/s"
        if speed is not None
        else "-"
    )

    table.add_row(
        "Speed",
        speed_text,
    )

    table.add_row("", "")

    loss = train.get("loss")
    loss_text = (
        f"{loss:.6f}"
        if isinstance(loss, (int, float))
        else "-"
    )

    learning_rates = train.get("learning_rates")
    lr_text = (
        ", ".join(f"{lr:.8f}" for lr in learning_rates)
        if learning_rates
        else "-"
    )

    table.add_row(
        "Loss",
        loss_text,
    )

    table.add_row(
        "LR",
        lr_text,
    )

    table.add_row(
        "Message",
        str(lifecycle.get("message") or "-"),
    )

    if running:
        table.add_row("", "")

        table.add_row(
            "",
            Text(
                "Press Ctrl+C to exit",
                style="dim",
                justify="right",
            ),
        )

    return Panel(
        table,
        title=Text(
            title,
            style="bold",
        ),
        title_align="left",
        border_style="blue",
    )


@app.command()
def status(
    path: Annotated[Path, typer.Argument()] = Path("."),
) -> None:
    """Inspect the current status of an Owl workspace.

    Rich live rendering may require a terminal with full ANSI control support.
    """
    snapshot_path = path / "snapshot"

    lifecycle_path = snapshot_path / "lifecycle.json"
    train_path = snapshot_path / "train.json"

    if not lifecycle_path.exists() and not train_path.exists():
        typer.echo(
            f"No Owl snapshot found in: {snapshot_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    console = Console()
    title_width = max(
        1,
        console.size.width - _TITLE_MARGIN,
    )
    title = _compact_path(
        path,
        max_width=title_width,
    )

    lifecycle = _read_snapshot(lifecycle_path)
    train = _read_snapshot(train_path)

    if lifecycle is _READ_FAILED or train is _READ_FAILED:
        typer.echo(
            f"Failed to read Owl snapshot in: {snapshot_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    if lifecycle.get("status") != "running":
        console.print(
            _build_view(
                title,
                lifecycle,
                train,
            )
        )
        return

    speed_samples: deque[tuple[int, float]] = deque(
        maxlen=_SPEED_SAMPLE_WINDOW
    )

    initial_position = _training_position(train)

    if initial_position is not None:
        speed_samples.append(initial_position)

    speed: float | None = None

    try:
        with Live(
            _build_view(
                title,
                lifecycle,
                train,
                speed,
            ),
            auto_refresh=False,
            transient=False,
            console=console,
        ) as live:
            while True:
                time.sleep(_POLL_INTERVAL_SECONDS)

                lifecycle = _read_snapshot(lifecycle_path)
                train = _read_snapshot(train_path)

                if lifecycle is _READ_FAILED or train is _READ_FAILED:
                    continue

                position = _training_position(train)

                if position is not None:
                    _, updated_at = position

                    if (
                        not speed_samples
                        or updated_at != speed_samples[-1][1]
                    ):
                        speed_samples.append(position)
                        speed = _training_speed(speed_samples)

                live.update(
                    _build_view(
                        title,
                        lifecycle,
                        train,
                        speed,
                    ),
                    refresh=True,
                )

                if lifecycle.get("status") in _TERMINAL_STATUSES:
                    break

    except KeyboardInterrupt:
        pass