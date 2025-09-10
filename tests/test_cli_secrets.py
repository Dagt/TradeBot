from typer.testing import CliRunner

from tradingbot.cli.commands import secrets as cli_secrets
from tradingbot.cli import main as cli_main


def test_cli_set_show_delete(tmp_path):
    runner = CliRunner()
    env_file = tmp_path / ".env"

    result = runner.invoke(cli_secrets.app, ["set", "FOO", "bar", "--env-file", str(env_file)])
    assert result.exit_code == 0
    assert "FOO=bar" in env_file.read_text()

    result = runner.invoke(cli_secrets.app, ["show", "FOO", "--env-file", str(env_file)])
    assert result.exit_code == 0
    assert "bar" in result.stdout

    result = runner.invoke(cli_secrets.app, ["delete", "FOO", "--env-file", str(env_file)])
    assert result.exit_code == 0
    assert env_file.read_text() == ""


def test_cli_main_exposes_secrets(tmp_path):
    runner = CliRunner()
    env_file = tmp_path / ".env"

    result = runner.invoke(
        cli_main.app, ["secrets", "set", "FOO", "bar", "--env-file", str(env_file)]
    )
    assert result.exit_code == 0
    assert "FOO=bar" in env_file.read_text()
