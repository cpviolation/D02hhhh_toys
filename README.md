# D02hhhh_toys

Toy studies of how a mis-modelled LHCb-like acceptance biases x_CP, y_CP, Δx and Δy
in a time-dependent D0 → K⁺K⁻π⁺π⁻ amplitude fit.

**Documentation: <https://cpviolation.github.io/D02hhhh_toys/>**. It covers:
- the status of each phase;
- the open and confirmed decisions;
- the notes on each phase.

```bash
uv sync
.venv/bin/python -m pytest
.venv/bin/d02hhhh-toys study configs/study/smoke.yaml --out results/smoke --workers 8
```

To build the documentation locally, run `uv run --group docs mkdocs serve`.
