from pathlib import Path
import time
from typing import Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

from model import VegetationFireModel

DATA_PATH = Path("./data/prp_mktx")
FIRE_PATH = Path("./data/inc_utm32")
RESULTS_PATH = Path("./results")
START_YEAR = 1978
END_YEAR = 2020  # exclusive upper bound, matching the previous np.arange(1978, 2020)
COMPONENT_NAMES = ("A", "U", "Sy", "Sm", "Ry", "Rm")


def load_land_cover(year: int) -> Tuple[np.ndarray, Affine, CRS | None]:
    """Load the starting land-cover raster and normalise codes."""
    filepath = DATA_PATH / f"prp{year}_mktx.asc"
    with rasterio.open(filepath) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    data = data.copy()
    data[data == 6] = 3  # coltivi -> non vegetated
    data[data == 7] = 1  # boschi poco soggetti -> latifoglie
    data[data == 9] = -9999  # undefined -> nodata
    return data, transform, crs


def build_initial_map(land_cover: np.ndarray) -> np.ndarray:
    """Translate land-cover codes into vegetation proportion vectors."""
    grid_size = land_cover.shape[0]
    initial_map = np.zeros((grid_size, grid_size, 6), dtype=float)

    vector_map = {
        1: np.array([0, 0, 0, 0, 0.2, 0.8]),  # latifoglie -> Ry, Rm
        2: np.array([0, 1, 0, 0, 0, 0]),  # vegetazione arbustiva -> U
        3: np.full(6, -3333.0),  # aree non vegetate
        4: np.array([1, 0, 0, 0, 0, 0]),  # praterie -> A
        5: np.array([0, 0, 0.2, 0.8, 0, 0]),  # conifere -> Sy, Sm
        -3333: np.full(6, -3333.0),
        -9999: np.full(6, -9999.0),
    }

    for i in range(grid_size):
        for j in range(grid_size):
            code = land_cover[i, j]
            if code not in vector_map:
                raise ValueError(f"Unexpected land-cover value {code} at position ({i}, {j})")
            initial_map[i, j] = vector_map[code]

    return initial_map


def load_fire_mask(year: int, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Load yearly ignition raster, falling back to zeros when missing."""
    filepath = FIRE_PATH / f"i_{year}_utm32.asc"
    try:
        with rasterio.open(filepath) as src:
            fires = src.read(1, out_shape=expected_shape, resampling=Resampling.nearest)
    except RasterioIOError:
        print(f"No fire raster for {year}. Using a zero mask.")
        return np.zeros(expected_shape, dtype=bool)

    fires = fires.copy()
    fires[fires == -9999] = 0
    if fires.shape != expected_shape:
        raise ValueError(f"Fire raster for {year} has shape {fires.shape}, expected {expected_shape}")
    return fires.astype(bool)


def save_encoded_grid(encoded: np.ndarray, transform, crs, year: int) -> None:
    """Persist the encoded grid for the current year."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    if transform is None:
        print(f"Transform missing; skipping save for {year}.")
        return

    output = RESULTS_PATH / f"results_tosc_{year}.asc"
    with rasterio.open(
        output,
        "w",
        driver="AAIGrid",
        height=encoded.shape[0],
        width=encoded.shape[1],
        count=1,
        dtype=encoded.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(encoded, 1)


def run_simulation() -> None:
    start_time = time.perf_counter()

    land_cover, transform, crs = load_land_cover(START_YEAR)
    grid_shape = land_cover.shape
    if grid_shape[0] != grid_shape[1]:
        raise ValueError(f"Land-cover grid must be square; got shape {grid_shape}")

    initial_map = build_initial_map(land_cover)
    model = VegetationFireModel(initial_map)

    years = np.arange(START_YEAR, END_YEAR)
    system_metrics = np.zeros((len(years), len(COMPONENT_NAMES)), dtype=float)

    for idx, year in enumerate(years):
        print(year)
        fire_mask = load_fire_mask(year, grid_shape)
        step_stats = model.step(fire_mask)
        system_metrics[idx] = step_stats["averages"]
        save_encoded_grid(model.encode_grid(), transform, crs, year)

    elapsed = time.perf_counter() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")

    for comp_idx, name in enumerate(COMPONENT_NAMES):
        print(f"{name}_system mean: {system_metrics[:, comp_idx].mean():.4f}")


if __name__ == "__main__":
    run_simulation()
