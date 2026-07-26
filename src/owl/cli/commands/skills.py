"""Install the bundled Owl Skill for a supported coding agent."""

import shutil
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import typer

from ..app import app


_TARGETS = {
    "1": Path(".agents") / "skills" / "owl",
    "2": Path(".claude") / "skills" / "owl",
}


def _select_target() -> Path:
    """Prompt the user to select a Skill installation target."""
    typer.echo("请选择安装目标：")
    typer.echo()
    typer.echo("1. Codex")
    typer.echo("2. Claude Code")
    typer.echo()

    while True:
        choice = typer.prompt("请输入选项 [1/2]")

        if choice in _TARGETS:
            return Path.cwd() / _TARGETS[choice]

        typer.echo("无效选项，请输入 1 或 2。", err=True)


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    """Copy a package resource directory to the local filesystem."""
    destination.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        target = destination / child.name

        if child.is_dir():
            _copy_resource_tree(child, target)
            continue

        with child.open("rb") as source_file, target.open("wb") as target_file:
            shutil.copyfileobj(source_file, target_file)

@app.command()
def skills() -> None:
    """Install the bundled Owl Skill for Codex or Claude Code."""
    source = resources.files("owl.resources").joinpath("skills", "owl")

    if not source.is_dir():
        typer.echo("未找到 Owl Skill 资源。", err=True)
        raise typer.Exit(code=1)

    destination = _select_target()

    if destination.exists():
        overwrite = typer.confirm(
            f"{destination} 已存在，是否覆盖？",
            default=False,
        )
        if not overwrite:
            typer.echo("已取消安装。")
            return

        shutil.rmtree(destination)

    _copy_resource_tree(source, destination)

    typer.echo()
    typer.echo(f"Owl Skill 已安装到：{destination}")