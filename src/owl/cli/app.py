"""Typer application for the Owl command-line interface."""

import typer


app = typer.Typer(
    name="owl",
    help="Owl command-line interface.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Initialize the Owl command-line interface."""


__all__ = [
    "app",
]