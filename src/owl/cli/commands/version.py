"""Version command implementation."""

from owl.__about__ import __version__

from ..app import app


@app.command()
def version() -> None:
    """Show the installed Owl version."""
    print(__version__)