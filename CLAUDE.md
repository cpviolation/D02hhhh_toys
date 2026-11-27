# D02hhhh_toys

Toy studies of how a mis-modelled acceptance biases x_CP, y_CP, Δx, Δy in a
time-dependent D0 → hhhh amplitude fit. The plan is in `docs/TOY_STUDY_PLAN.md`; notes
on each phase go in `docs/notes/`.

## Ground rules

1. **Reuse, don't rewrite.** The amplitude model and plain phase-space generation come
   from `aman-examples` (pinned to tag `0.1.0`). New code wraps or extends it.
2. **Conventions are pinned by tests**, never assumed: daughter ordering
   `(K+, K−, π+, π−)`, CP conjugation (charge swap **and** p⃗ → −p⃗), mixing signs.
3. **Ask before choosing physics** (cuts, acceptance numbers, scenarios). Propose,
   explain the impact, and wait for confirmation.
4. Vectorised code only, float64 in the likelihood (`jax_enable_x64`), explicit seeds,
   and `pytest` green before a phase is closed.

## Conventions

- `z_CP ± Δz = −(q/p)^{±1}(y + ix)`, `x_CP = −Im z_CP`, `y_CP = −Re z_CP`,
  `Δx = −Im Δz`, `Δy = −Re Δz`. In code: `x_cp_mix, y_cp_mix, dx_mix, dy_mix`. This
  y_CP is **not** the lifetime-ratio observable.
- Decay time τ = Γt, in units of the D0 lifetime.
- Events are `(N, 4, 4)` arrays: particle `(K+, K−, π+, π−)` × component `(x, y, z, E)`,
  in GeV, in the D0 rest frame unless stated otherwise.

## Commands

- `uv sync` (add `--extra cuda` for the GPU), then `.venv/bin/python -m pytest`.
  Add `-m "slow or not slow"` for the tests that build the full model.
- `uvx ruff check && uvx ruff format --check`, or `pre-commit run --all-files`.
- The shell may have `VIRTUAL_ENV` set to another project: call `.venv/bin/python`
  directly, or use `uv run`.
- Toy studies: `.venv/bin/d02hhhh-toys study configs/study/<name>.yaml --out results/<name>
  --workers N` (resumable), then `.venv/bin/d02hhhh-toys summarise results/<name>`.
  Add `--store-toys` to keep the toys, and `--toys-from results/<other>` to refit stored
  toys. See `docs/notes/study.md`.
- `results/` is a symbolic link to `/data01/gr1/lhcb/mmartinelli/D02hhhh_toys/output`
  (large outputs, not in git). In a new clone, make it with
  `.venv/bin/d02hhhh-toys setup <folder>` (`--move`, `--force`: see `workspace.py`).
- Normalisation-offset factor v: `.venv/bin/python scripts/variance_budget.py
  configs/study/<name>.yaml` (see `docs/notes/moments.md`).
- GPU tests: `D02HHHH_TEST_BACKEND=jax .venv/bin/python -m pytest -m slow` (without `-n`).

## Workflow

- One branch and one pull request per phase (from Phase 7). Physics values (cuts,
  acceptance numbers, scenario magnitudes) are proposed and confirmed before use.
- Decisions live in `docs/decisions.md`.
  - A proposal goes under *Open decisions*, with an ID (P<n>), its options and their
    impact, and a recommendation.
  - When the user confirms it, move it to *Confirmed decisions* (C<n>), with the date.
  - Keep the status table in `docs/index.md` up to date.
- The site (MkDocs Material, `mkdocs.yml`) is published to GitHub Pages from `main` by
  `.github/workflows/docs.yml`. Build it with
  `uv run --group docs mkdocs build`, which is strict: broken links fail the build.
