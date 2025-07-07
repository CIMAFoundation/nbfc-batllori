import numpy as np
import matplotlib.pyplot as plt
import time
from rasterio.errors import RasterioIOError
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
import re
import os
import glob
from matplotlib.patches import Patch

# Path to the directory containing the .asc files
DATA_PATH = './data/prp_mktx/'
RESULTS_PATH = './results/'


def encode_grid(grid):
    N = len(grid)
    out = np.empty((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            p = grid[i][j]['proportions'].ravel()
            # all-equal cases
            if np.all(p == -9999):
                out[i, j] = -9999
            elif np.all(p == -3333):
                out[i, j] = 3
            else:
                s23 = p[2] + p[3]
                s45 = p[4] + p[5]
                # dictionary of {value: code}
                candidates = { p[0]: 4, p[1]: 2, s23:    5, s45:    1 }
                # pick the entry with the largest key
                max_val = max(candidates)
                out[i, j] = candidates[max_val]

    return out


# Pattern to match the files
files = glob.glob(os.path.join(DATA_PATH, 'prp????_mktx.asc'))

crs_save = None
transform_save = None

input_data = {}

for file in files:
    # Extract year using regex
    match = re.search(r'prp(\d{4})_mktx\.asc', os.path.basename(file))
    if match:
        year = match.group(1)
        with rasterio.open(file) as src:
            data = src.read(1)  # Read the first (and usually only) band
            transform = src.transform  # Affine transformation (for georeferencing)
            crs = src.crs  # Coordinate reference system
            transform_save = transform
            crs_save = crs
            data[data==6] = 3   # "coltivi" in "aree non o poco vegetate"
            data[data==7] = 1   # "boschi poco soggetti al fuoco " in "latifoglie"
            data[data==9] = -9999   # Not defined areas in NaNs
            input_data[f'tosc{year}'] = data

# Starting from 1978
starting_data = input_data['tosc1978']
mask = (starting_data == -9999)
for key in input_data:
    input_data[key][mask] = -9999
starting_data = starting_data.copy()



start_time = time.perf_counter()
np.random.seed(0)

# Parameters
grid_size = 1000
years = np.arange(1978,2020)
timesteps = len(years) 
TSF_classes = 3  
level = 0 

A_system  = np.zeros(timesteps)
U_system  = np.zeros(timesteps)
Sy_system = np.zeros(timesteps)
Sm_system = np.zeros(timesteps)
Ry_system = np.zeros(timesteps)
Rm_system = np.zeros(timesteps)

k_sy_sm = 0.1
k_ry_rm = 0.1
k_au   = 0.01
rho_s = 0.05
rho_sy = 0.0125
rho_r = 0.015
rho_rm = 0.0125
fraction = 0.75

mu_s  = rho_s*fraction
mu_sy = rho_sy*fraction
mu_r  = rho_r*fraction
mu_rm = rho_rm*fraction
w_ry = 0.3
w_rm = 0.15
w_sy = 0.4
w_sm = 0.25
w_u = 0.1

omega_l = 0.7
omega_cell = 1 - omega_l

# previously np.zeros((TSF_classes, 6)), but a line is enough
grid = [[{'proportions': np.zeros((1, 6)), 'TSF': 0} for j in range(grid_size)] for i in range(grid_size)] 
total_cells = grid_size * grid_size


for i in range(grid_size):
    for j in range(grid_size):        
        if starting_data[i,j]==-9999: # NaN
            init_vector = np.array([-9999,-9999,-9999,-9999,-9999,-9999])
        elif starting_data[i,j]==1: # latifoglie -> Ry, Rm
            init_vector = np.array([0,0,0,0,0.2,0.8])
        elif starting_data[i,j]==2: # vegetazione arbustiva -> U
            init_vector = np.array([0,1,0,0,0,0])
        elif starting_data[i,j]==3: # aree non vegetate -> NaN
            init_vector = np.array([-3333,-3333,-3333,-3333,-3333,-3333])
        elif starting_data[i,j]==4: # praterie -> A
            init_vector = np.array([1,0,0,0,0,0])
        elif starting_data[i,j]==5: # conifere -> Sy, Sm
            init_vector = np.array([0,0,0.2,0.8,0,0])
        else:
            raise ValueError(f"Unexpected value {starting_data[i,j]} at position ({i}, {j})")
        
        grid[i][j]['proportions'][0, :] = init_vector

# Prealloca la griglia per la visualizzazione dei colori (1-based, come in MATLAB)
color_grid = np.zeros((grid_size, grid_size), dtype=int)

cell = []
num_fires = 0
tot_nf = []

# Simulation cycle
for t in range(timesteps):
    print(years[t])

    filepath = f'./data/inc_utm32/i_{years[t]}_utm32.asc'

    try:
        with rasterio.open(filepath) as src:
            fires = src.read(1)
            transform = src.transform
            crs = src.crs
            fires[fires == -9999] = 0
    except RasterioIOError:
        print('No fire. Creating zero array instead.')
        fires = np.zeros((grid_size, grid_size), dtype=np.uint8)
        transform = None
        crs = None

    # Calcolo delle medie locali per Sm e Rm (usando il livello "level")
    Sm_values = np.zeros((grid_size, grid_size))
    Rm_values = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # MATLAB: grid(i,j).proportions(level,4) diventa grid[i][j]['proportions'][level,3] in Python
            Sm_values[i, j] = grid[i][j]['proportions'][level, 3]
            Rm_values[i, j] = grid[i][j]['proportions'][level, 5]
    invalid_values = [-3333, -9999]
    
    mask = ~np.isin(Sm_values, invalid_values)  # Mask of valid values
    Sm_mean = np.mean(Sm_values[mask].mean())
    mask = ~np.isin(Rm_values, invalid_values)  # Mask of valid values
    Rm_mean = np.mean(Rm_values[mask].mean())
    
    # Inizializzazione degli accumulatori di sistema
    A_total = 0.0
    U_total = 0.0
    Sy_total = 0.0
    Sm_total = 0.0
    Ry_total = 0.0
    Rm_total = 0.0

    # Aggiornamento per ogni cella della griglia
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j]['proportions'][0,0] in (-9999, -3333, -6666): # If NaN -> no computation
                continue
            
            # Lettura dei valori correnti
            A_old = grid[i][j]['proportions'][level, 0]
            U_old = grid[i][j]['proportions'][level, 1]
            Sy_old= grid[i][j]['proportions'][level, 2]
            Sm_old= grid[i][j]['proportions'][level, 3]
            Ry_old= grid[i][j]['proportions'][level, 4]
            Rm_old= grid[i][j]['proportions'][level, 5]
            
            old_vals = np.array([A_old, U_old, Sy_old, Sm_old, Ry_old, Rm_old])

            # Updating fire stats and checking fire occurrence
            grid[i][j]['TSF'] += 1
            tsf = grid[i][j]['TSF']
            fire = False
            # if np.random.uniform() > 0.95:  # old code for fire generation
            #     fire = True
            if fires[i][j]==1:
                fire = True
                
            if not fire:
                # Calcolo delle influenze locali e globali
                F_s = omega_cell * Sm_old + omega_l * Sm_mean
                F_r = omega_cell * Rm_old + omega_l * Rm_mean
                
                K_u_sy = mu_s + (rho_s - mu_s) * F_s
                K_ry_sy = mu_sy + (rho_sy - mu_sy) * F_s
                K_u_ry = mu_r + (rho_r - mu_r) * F_r
                K_sm_rm = mu_rm + (rho_rm - mu_rm) * F_s
                
                # Aggiornamento delle proporzioni vegetative
                A  = (1 - k_au) * A_old
                U  = (1 - K_u_sy - K_u_ry) * U_old + k_au * A_old
                Sy = (1 - k_sy_sm) * Sy_old + K_u_sy * U_old + K_ry_sy * Ry_old
                Sm = (1 - K_sm_rm) * Sm_old + k_sy_sm * Sy_old
                Ry = (1 - K_ry_sy - k_ry_rm) * Ry_old + K_u_ry * U_old
                Rm = Rm_old + k_ry_rm * Ry_old + K_sm_rm * Sm_old

            else: # Fire occurs
                num_fires += 1

                # Resprouting skill, both younglings and boomers
                P_ry = min(0.35**(3.367-0.306*(tsf-1.)),1.)
                P_rm = min(tsf/5.,1)

                # Post-fire recovery, seeders and resprouters
                G = (1 - w_sm) * Sm_old + min((1 - w_sy) * Sy_old, Sm_old)
                T = P_rm * (1 - w_rm) * Rm_old + P_ry * (1 - w_ry) * Ry_old

                # Fire induced conversions
                C_g = w_sm * Sm_old + (Sy_old - min((1 - w_sy) * Sy_old, Sm_old))
                C_t = (1 - (1 - w_rm) * P_rm) * Rm_old + (1 - (1 - w_ry) * P_ry) * Ry_old
                C_u = w_u * U_old
                C_u_sy = min(C_u/2. , Sm_old)
                C_u_a = C_u - C_u_sy
                
                # Cell composition update
                A  = A_old + C_u_a
                U  = U_old - C_u_sy - C_u_a + C_g + C_t
                Sy = G + C_u_sy
                Sm = 0.
                Ry = T
                Rm = 0.

                # Reset no-fire time
                grid[i][j]['TSF'] = 0

            # Aggiornamento della griglia e definizione del colore in base alla componente massima
            new_vals = np.array([A, U, Sy, Sm, Ry, Rm])
            if np.any(new_vals > 1):
                print(new_vals,i,j)
                # stop # was this used for debugging?
            grid[i][j]['proportions'][level, :] = new_vals
            # np.argmax restituisce indice 0-based: aggiungiamo 1 per avere lo stesso range di MATLAB (1-6)
            color_grid[i, j] = np.argmax(new_vals) + 1
            
            # Accumulo per le medie di sistema (somma lungo tutte le classi TSF)
            A_total  += np.sum(grid[i][j]['proportions'][:, 0])
            U_total  += np.sum(grid[i][j]['proportions'][:, 1])
            Sy_total += np.sum(grid[i][j]['proportions'][:, 2])
            Sm_total += np.sum(grid[i][j]['proportions'][:, 3])
            Ry_total += np.sum(grid[i][j]['proportions'][:, 4])
            Rm_total += np.sum(grid[i][j]['proportions'][:, 5])

    # Saving the grid
    gr = encode_grid(grid)
    output_filename = os.path.join(RESULTS_PATH, f'results_tosc_{years[t]}.asc')
    
    if transform_save is None:
        print(f"Error: transform not defined for year {years[t]}. Skipping saving.")
    else:        
        with rasterio.open(
            output_filename,
            'w',
            driver='AAIGrid',
            height=gr.shape[0],
            width=gr.shape[1],
            count=1,
            dtype=gr.dtype,
            crs=crs_save,
            transform=transform_save,
            nodata=-9999
        ) as dst:
            dst.write(gr, 1)
    
    # Salvataggio dei risultati di sistema medi per timestep
    A_system[t]  = A_total / total_cells
    U_system[t]  = U_total / total_cells
    Sy_system[t] = Sy_total / total_cells
    Sm_system[t] = Sm_total / total_cells
    Ry_system[t] = Ry_total / total_cells
    Rm_system[t] = Rm_total / total_cells

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.2f} seconds")