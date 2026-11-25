import aman_examples.d02kkpipi as d02kkpipi

from d02hhhh_toys import cli


def test_cli_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip()


def test_upstream_final_state():
    """The daughter ordering that the whole package relies on."""
    assert d02kkpipi.FINAL_STATE == ["K+", "K-", "pi+", "pi-"]
