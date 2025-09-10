"""CLI commands for managing secrets stored in a ``.env`` file."""
from __future__ import annotations

from pathlib import Path

import typer

from ...utils import secrets as secrets_utils


app = typer.Typer(help="Manage API secrets and other sensitive values")


@app.command("set")
def set_(
    key: str,
    value: str,
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file"),
) -> None:
    """Store ``key=value`` in the env file."""

    secrets_utils.set_secret(key, value, env_file)
    typer.echo(f"{key} set")


@app.command("show")
def show(
    key: str,
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file"),
) -> None:
    """Display the value for ``key`` from the env file."""

    val = secrets_utils.read_secret(key, env_file)
    if val is None:
        raise typer.Exit(1)
    typer.echo(val)


@app.command("delete")
def delete(
    key: str,
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file"),
) -> None:
    """Remove ``key`` from the env file."""

    secrets_utils.delete_secret(key, env_file)
    typer.echo(f"{key} deleted")

