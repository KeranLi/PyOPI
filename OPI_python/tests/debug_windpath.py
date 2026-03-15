import numpy as np
import h5py
from opi.io.data_loader import get_input
from opi.io.run_file import parse_run_file
from opi.wind_path import wind_path
from opi.coordinates import lonlat2xy

# Load actual data
print("Loading data...")
with h5py.File('../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat', 'r') as f:
    x = np.array(f['x']).flatten()
    y = np.array(f['y']).flatten()
    section_lon0 = float(np.array(f['sectionLon0']).flatten()[0])
    section_lat0 = float(np.array(f['sectionLat0']).flatten()[0])
    lon0 = float(np.array(f['lon0']).flatten()[0])
    lat0 = float(np.array(f['lat0']).flatten()[0])

params = parse_run_file('f:/code/OPI-Orographic-Precipitation-and-Isotopes/OPI_matlab/runs/run033.run')
input_data = get_input(params['data_path'], params['topo_file'], r_tukey=0.0, sample_file=None)

h_grid = input_data['h_grid']

# Swap x and y
x, y = y, x

# Downsample
factor = 4
h_grid_ds = h_grid[::factor, ::factor]
x_ds = x[::factor]
y_ds = y[::factor]

print(f"Grid: h_grid_ds={h_grid_ds.shape}, x_ds={len(x_ds)}, y_ds={len(y_ds)}")

# Convert section origin
x_section0, y_section0 = lonlat2xy(section_lon0, section_lat0, lon0, lat0)
print(f"Section origin: ({x_section0}, {y_section0})")

# Calculate wind path
azimuth = 34  # From beta
print(f"Calculating wind path with azimuth={azimuth}...")
x_path, y_path, s_path, s_limits = wind_path(
    x_section0, y_section0, azimuth, x_ds, y_ds, None
)

print(f"x_path shape: {x_path.shape}")
print(f"y_path shape: {y_path.shape}")
print(f"s_path shape: {s_path.shape}")
print(f"s_limits: {s_limits}")
print(f"n_path = {len(x_path)}")
