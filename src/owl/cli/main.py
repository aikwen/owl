"""Main entry point for the Owl command-line interface."""

from .app import app
from . import commands as _commands


def main() -> None:
    """Run the Owl command-line interface."""
    app()


__all__ = [
    "main",
]