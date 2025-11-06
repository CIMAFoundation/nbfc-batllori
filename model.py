from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


INVALID_VALUES = (-9999, -3333, -6666)


@dataclass(frozen=True)
class ModelParams:
    k_sy_sm: float = 0.1
    k_ry_rm: float = 0.1
    k_au: float = 0.01
    rho_s: float = 0.05
    rho_sy: float = 0.0125
    rho_r: float = 0.015
    rho_rm: float = 0.0125
    fraction: float = 0.75
    w_ry: float = 0.3
    w_rm: float = 0.15
    w_sy: float = 0.4
    w_sm: float = 0.25
    w_u: float = 0.1
    omega_l: float = 0.7

    @property
    def omega_cell(self) -> float:
        return 1.0 - self.omega_l


class VegetationFireModel:
    """Encapsulates the vegetation–fire dynamics so callers can reuse it without I/O."""

    def __init__(self, initial_map: np.ndarray, params: ModelParams | None = None) -> None:
        self.params = params or ModelParams()
        self.proportions = np.asarray(initial_map, dtype=float).copy()
        if self.proportions.ndim != 3 or self.proportions.shape[2] != 6:
            raise ValueError("initial_map must have shape (rows, cols, 6)")

        self.grid_size = self.proportions.shape[0]
        if self.proportions.shape[0] != self.proportions.shape[1]:
            raise ValueError("initial_map must be square")

        self.tsf = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.total_cells = self.grid_size * self.grid_size
        self.num_fires = 0

        # Pre-compute coefficients that only depend on parameters.
        self.mu_s = self.params.rho_s * self.params.fraction
        self.mu_sy = self.params.rho_sy * self.params.fraction
        self.mu_r = self.params.rho_r * self.params.fraction
        self.mu_rm = self.params.rho_rm * self.params.fraction

    def step(self, fire_mask: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        """Advance the model by one timestep using a boolean fire mask."""
        mask = np.asarray(fire_mask, dtype=bool)
        if mask.shape != (self.grid_size, self.grid_size):
            raise ValueError("fire_mask must match the model grid size")

        sm_values = self.proportions[:, :, 3]
        rm_values = self.proportions[:, :, 5]

        sm_valid = ~np.isin(sm_values, INVALID_VALUES)
        rm_valid = ~np.isin(rm_values, INVALID_VALUES)

        Sm_mean = float(sm_values[sm_valid].mean()) if np.any(sm_valid) else 0.0
        Rm_mean = float(rm_values[rm_valid].mean()) if np.any(rm_valid) else 0.0

        totals = np.zeros(6, dtype=float)
        fires_this_step = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.proportions[i, j]

                if cell[0] in INVALID_VALUES:
                    continue

                A_old, U_old, Sy_old, Sm_old, Ry_old, Rm_old = cell
                self.tsf[i, j] += 1
                tsf = self.tsf[i, j]

                fire = bool(mask[i, j])
                if not fire:
                    F_s = self.params.omega_cell * Sm_old + self.params.omega_l * Sm_mean
                    F_r = self.params.omega_cell * Rm_old + self.params.omega_l * Rm_mean

                    K_u_sy = self.mu_s + (self.params.rho_s - self.mu_s) * F_s
                    K_ry_sy = self.mu_sy + (self.params.rho_sy - self.mu_sy) * F_s
                    K_u_ry = self.mu_r + (self.params.rho_r - self.mu_r) * F_r
                    K_sm_rm = self.mu_rm + (self.params.rho_rm - self.mu_rm) * F_s

                    A = (1 - self.params.k_au) * A_old
                    U = (1 - K_u_sy - K_u_ry) * U_old + self.params.k_au * A_old
                    Sy = (1 - self.params.k_sy_sm) * Sy_old + K_u_sy * U_old + K_ry_sy * Ry_old
                    Sm = (1 - K_sm_rm) * Sm_old + self.params.k_sy_sm * Sy_old
                    Ry = (1 - K_ry_sy - self.params.k_ry_rm) * Ry_old + K_u_ry * U_old
                    Rm = Rm_old + self.params.k_ry_rm * Ry_old + K_sm_rm * Sm_old
                else:
                    fires_this_step += 1
                    self.num_fires += 1

                    P_ry = min(0.35 ** (3.367 - 0.306 * (tsf - 1.0)), 1.0)
                    P_rm = min(tsf / 5.0, 1.0)

                    G = (1 - self.params.w_sm) * Sm_old + min((1 - self.params.w_sy) * Sy_old, Sm_old)
                    T = P_rm * (1 - self.params.w_rm) * Rm_old + P_ry * (1 - self.params.w_ry) * Ry_old

                    C_g = self.params.w_sm * Sm_old + (Sy_old - min((1 - self.params.w_sy) * Sy_old, Sm_old))
                    C_t = (
                        (1 - (1 - self.params.w_rm) * P_rm) * Rm_old
                        + (1 - (1 - self.params.w_ry) * P_ry) * Ry_old
                    )
                    C_u = self.params.w_u * U_old
                    C_u_sy = min(C_u / 2.0, Sm_old)
                    C_u_a = C_u - C_u_sy

                    A = A_old + C_u_a
                    U = U_old - C_u_sy - C_u_a + C_g + C_t
                    Sy = G + C_u_sy
                    Sm = 0.0
                    Ry = T
                    Rm = 0.0

                    self.tsf[i, j] = 0

                new_vals = np.array([A, U, Sy, Sm, Ry, Rm])
                self.proportions[i, j] = new_vals
                totals += new_vals

        averages = totals / self.total_cells
        return {
            "totals": totals,
            "averages": averages,
            "fires_this_step": fires_this_step,
            "cumulative_fires": self.num_fires,
        }

    def update_vegetation_map(self, new_map: np.ndarray, reset_tsf: bool = False) -> None:
        """Replace the full vegetation map, optionally resetting TSF counters."""
        new_map = np.asarray(new_map, dtype=float)
        if new_map.shape != self.proportions.shape:
            raise ValueError("new_map must match the current map shape")
        self.proportions = new_map.copy()
        if reset_tsf:
            self.tsf.fill(0)

    def update_cell(self, row: int, col: int, new_values: np.ndarray, reset_tsf: bool = False) -> None:
        """Update a single cell with custom proportions."""
        new_values = np.asarray(new_values, dtype=float)
        if new_values.shape != (6,):
            raise ValueError("new_values must have shape (6,)")
        self.proportions[row, col] = new_values
        if reset_tsf:
            self.tsf[row, col] = 0

    def get_vegetation_map(self, copy: bool = True) -> np.ndarray:
        """Return the current vegetation proportions."""
        return self.proportions.copy() if copy else self.proportions

    def encode_grid(self) -> np.ndarray:
        """Return the encoded integer grid used for exports."""
        encoded = np.full((self.grid_size, self.grid_size), -9999, dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.proportions[i, j]
                if np.all(cell == -9999):
                    encoded[i, j] = -9999
                elif np.all(cell == -3333):
                    encoded[i, j] = 3
                else:
                    s23 = cell[2] + cell[3]
                    s45 = cell[4] + cell[5]
                    candidates: Dict[float, int] = {
                        cell[0]: 4,
                        cell[1]: 2,
                        s23: 5,
                        s45: 1,
                    }
                    encoded[i, j] = candidates[max(candidates)]
        return encoded

    def grid_shape(self) -> Tuple[int, int]:
        return self.proportions.shape[:2]
