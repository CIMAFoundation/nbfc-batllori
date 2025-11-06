# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the full vegetation-fire simulation pipeline, from raster ingestion (`rasterio`) to grid encoding, stochastic updates, and Matplotlib visualisation. Keep new logic modular (helper functions near the existing loops) to avoid ballooning the 1 000×1 000 cell iteration blocks.
- `data/prp_mktx/` stores land-cover rasters (`prpYYYY_mktx.asc`) that seed the model; `data/inc_utm32/` holds yearly ignition rasters consumed per timestep. Preserve filenames because the loader relies on regex patterns.
- `results/` is where maps/arrays are written—treat it as disposable output. Version control only lightweight artifacts.
- `pyproject.toml` and `uv.lock` define Python 3.13 dependencies; edit via `uv` to keep the lock in sync.

## Build, Test, and Development Commands
- `uv sync` — create the virtual environment and install dependencies pinned in `uv.lock`.
- `uv run python main.py` — run the full simulation; expects both data folders present and writes PNG/ASC outputs into `results/`.
- `uv run python -m compileall main.py` — quick syntax check before committing when you do not need the full model run.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep imports grouped (stdlib, third-party, local) as in `main.py`.
- Module-level constants use `UPPER_SNAKE_CASE` (`DATA_PATH`, `RESULTS_PATH`); functions and variables use `snake_case`.
- Prefer NumPy vector operations when touching the inner simulation loops; document non-obvious math with short inline comments rather than block prose.
- Keep plotting palettes and boundary norms together so reviewers can trace legend changes easily.

## Testing Guidelines
- No automated suite exists yet; add `tests/test_*.py` powered by `pytest` when introducing new computation helpers. Use `uv run pytest` once such tests exist.
- For now, sanity-check runs by diffing new grids against prior outputs (e.g., `results/tosc2010.png`) and logging aggregate metrics (`A_system`, `U_system`, etc.) to catch regressions.
- When touching data-loading code, run a single-year dry run (temporarily slice `years = [1980]`) to keep feedback loops fast.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`type: summary`, e.g., `refactor: tighten transform handling`). Keep messages in the imperative mood and mention the subsystem (`grid`, `ingestion`, `viz`) when relevant.
- Each PR should describe the simulation impact, list new inputs/outputs, and attach before/after visuals if plots change. Link tracking issues and note any data dependencies so downstream automation knows when to refresh datasets.

## Data & Configuration Tips
- Raster paths are relative; avoid hard-coding absolute directories so CI agents can reproduce runs. Use `.env` overrides only if you document them.
- Large ASC files can exceed Git LFS limits—coordinate with the maintainers before adding new datasets, and document checksums in PRs to aid reproducibility.
