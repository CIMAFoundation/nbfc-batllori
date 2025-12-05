# Batllori Vegetation–Fire Simulator

This repo hosts a Python reimplementation of the Batllori et al. vegetation succession and fire dynamics model. It ingests yearly land-cover rasters (`prpYYYY_mktx.asc`) and ignition rasters (`i_YYYY_utm32.asc`), evolves six vegetation compartments (A, U, Sy, Sm, Ry, Rm) on a 1 000×1 000 grid via a Numba-accelerated core, and exports encoded rasters for downstream GIS analyses.

## Preparing the Environment

```bash
uv sync
```

`uv` creates a Python 3.13 virtual environment, installs runtime dependencies (NumPy, Numba, Rasterio, Matplotlib) and exposes the `batllori` console script.

## Running the Simulation

```bash
uv run batllori \
  --start-year 1978 \
  --end-year 2020 \
  --land-cover-path ./data/prp_mktx \
  --fire-path ./data/inc_utm32 \
  --results-path ./results
```

- `--start-year` / `--end-year` define the inclusive/exclusive year window.  
- `--land-cover-path` must contain files named `prp{year}_mktx.asc` (1978 is used to seed the model).  
- `--fire-path` must hold ignition rasters `i_{year}_utm32.asc`. Missing years default to an empty mask.  
- `--results-path` receives encoded grids `results_tosc_{year}.asc`.

Alternatively run `uv run python main.py`, which simply dispatches to the same CLI entry point.

## Using the Model in Other Projects

```python
import numpy as np
from batllori import VegetationFireModel, ModelParams

initial_map = np.zeros((1000, 1000, 6))
model = VegetationFireModel(initial_map, ModelParams())
fire_mask = np.zeros((1000, 1000), dtype=bool)
stats = model.step(fire_mask)
```

`stats["averages"]` returns per-compartment means, while `model.update_vegetation_map` and `model.encode_grid` let you integrate the core dynamics with your own I/O layer.
