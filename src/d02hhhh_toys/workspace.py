"""First-time setup: link the output directory ``results/`` to a folder of choice.

The study outputs (stored toys, normalisation moments, summaries) can take gigabytes,
so they live outside the repository, e.g. on a data disk. ``d02hhhh-toys setup DIR``
creates ``DIR`` and makes ``results`` a symbolic link to it. It never deletes data:

- ``results`` missing, an empty directory, or a link to ``DIR`` already: (re)linked;
- ``results`` a link to another folder: refused, unless ``force`` (only the link is
  replaced, the folder it pointed to is untouched);
- ``results`` a non-empty directory: refused, unless ``move``, which moves its
  contents into ``DIR`` first (refused if a name exists in both).
"""

from __future__ import annotations

import shutil
from pathlib import Path


class SetupError(RuntimeError):
    """The link cannot be made without risking data."""


def link_output(
    target: str | Path,
    link: str | Path = "results",
    *,
    move: bool = False,
    force: bool = False,
) -> Path:
    """Create ``target`` and make ``link`` a symbolic link to it; returns the target."""
    target = Path(target).expanduser().resolve()
    link = Path(link)
    if target == link.resolve() and not link.is_symlink():
        msg = f"{target} is {link} itself: choose a folder outside it"
        raise SetupError(msg)
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return target
        if not force:
            msg = (
                f"{link} already links to {link.resolve()}: use --force to point it to"
                f" {target} (the old folder is kept)"
            )
            raise SetupError(msg)
        link.unlink()
    elif link.is_dir():
        contents = list(link.iterdir())
        if contents and not move:
            msg = (
                f"{link} is a directory with {len(contents)} entries: use --move to move"
                f" them into {target}"
            )
            raise SetupError(msg)
        clashes = [p.name for p in contents if (target / p.name).exists()]
        if clashes:
            msg = (
                f"{target} already has {', '.join(sorted(clashes))}: nothing was moved"
            )
            raise SetupError(msg)
        for path in contents:
            shutil.move(str(path), target / path.name)
        link.rmdir()
    elif link.exists():
        msg = f"{link} is a file: remove or rename it first"
        raise SetupError(msg)
    link.symlink_to(target, target_is_directory=True)
    return target
