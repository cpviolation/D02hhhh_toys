import pytest

from d02hhhh_toys.cli import main
from d02hhhh_toys.workspace import SetupError, link_output


def test_new_link(tmp_path):
    target = link_output(tmp_path / "data" / "output", tmp_path / "results")
    assert target.is_dir()
    assert (tmp_path / "results").resolve() == target
    # again: nothing to do
    assert link_output(target, tmp_path / "results") == target


def test_empty_directory_is_replaced(tmp_path):
    (tmp_path / "results").mkdir()
    link_output(tmp_path / "out", tmp_path / "results")
    assert (tmp_path / "results").is_symlink()


def test_existing_results_are_kept_or_moved(tmp_path):
    results = tmp_path / "results"
    (results / "study").mkdir(parents=True)
    (results / "study" / "summary.json").write_text("{}")
    with pytest.raises(SetupError, match="--move"):
        link_output(tmp_path / "out", results)
    assert (results / "study" / "summary.json").exists()
    # a clash moves nothing
    (tmp_path / "out" / "study").mkdir(parents=True)
    with pytest.raises(SetupError, match="nothing was moved"):
        link_output(tmp_path / "out", results, move=True)
    assert (results / "study" / "summary.json").exists()
    link_output(tmp_path / "new", results, move=True)
    assert results.is_symlink()
    assert (results / "study" / "summary.json").read_text() == "{}"


def test_other_link_needs_force(tmp_path):
    results = tmp_path / "results"
    old = link_output(tmp_path / "old", results)
    (old / "keep").write_text("x")
    with pytest.raises(SetupError, match="--force"):
        link_output(tmp_path / "new", results)
    new = link_output(tmp_path / "new", results, force=True)
    assert results.resolve() == new
    assert (old / "keep").exists()


def test_cli(tmp_path, capsys):
    assert main(["setup", str(tmp_path / "out"), "--link", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r").is_symlink()
    assert main(["setup", str(tmp_path / "x"), "--link", str(tmp_path / "r")]) == 1
    assert "--force" in capsys.readouterr().err
