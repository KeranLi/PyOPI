import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from opi.io.data_loader import get_input
from opi.io.run_file import parse_run_file
from opi.wind_path import wind_path
from opi.coordinates import lonlat2xy
from opi.physics.fourier_solution import fourier_solution

# Load actual data
print("Loading data...")
with h5py.File('../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat', 'r') as f:
    x = np.array(f['x']).flatten()
    y = np.array(f['y']).flatten()
    section_lon0 = float(np.array(f['sectionLon0']).flatten()[0])
    section_lat0 = float(np.array(f['sectionLat0']).flatten()[0])
    lon0 = float(np.array(f['lon0']).flatten()[0])
    lat0 = float(np.array(f['lat0']).flatten()[0])
    z_bar = np.array(f['zBar_1']).flatten()
    T = np.array(f['T_1']).flatten()

params = parse_run_file('f:/code/OPI-Orographic-Precipitation-and-Isotopes/OPI_matlab/runs/run033.run')
input_data = get_input(params['data_path'], params['topo_file'], r_tukey=0.0, sample_file=None)

h_grid = input_data['h_grid']

# Swap x and y
x, y = y, x
print(f"After swap: x={len(x)}, y={len(y)}, h_grid={h_grid.shape}")

# Downsample
factor = 4
h_grid_ds = h_grid[::factor, ::factor]
x_ds = x[::factor]
y_ds = y[::factor]
print(f"Downsampled: h_grid_ds={h_grid_ds.shape}, x_ds={len(x_ds)}, y_ds={len(y_ds)}")

# Parameters
U, azimuth, NM, f_c, h_rho = 10, 34, 0.01, 0.0001, 8000
h_s = 1000
tau_c, tau_f, kappa = 1000, 500, 1000
gamma_ratio, rho_s0, h_v = 0.8, 0.01, 1000

# Calculate wind path
x_section0, y_section0 = lonlat2xy(section_lon0, section_lat0, lon0, lat0)
print(f"Section origin: ({x_section0:.2f}, {y_section0:.2f})")

x_path, y_path, s_path, s_limits = wind_path(
    x_section0, y_section0, azimuth, x_ds, y_ds, None
)
print(f"wind_path: x_path={len(x_path)}, s_path={len(s_path)}")

# Calculate s,t coordinates for path
s_path_coords = x_path * np.sin(np.deg2rad(azimuth)) + y_path * np.cos(np.deg2rad(azimuth))
t_path_coords = -x_path * np.cos(np.deg2rad(azimuth)) + y_path * np.sin(np.deg2rad(azimuth))
print(f"Path s,t: s_path_coords={len(s_path_coords)}, t_path_coords={len(t_path_coords)}")

# Fourier solution
print("\nRunning fourier_solution...")
fs = fourier_solution(x_ds, y_ds, h_grid_ds, U, azimuth, NM, f_c, h_rho)
s = fs['s']
t = fs['t']
n_s, n_t = len(s), len(t)
print(f"fourier_solution: s={n_s}, t={n_t}")

# Check if s,t from fourier_solution match s_path_coords, t_path_coords ranges
print(f"s range: [{s.min():.2f}, {s.max():.2f}]")
print(f"s_path_coords range: [{s_path_coords.min():.2f}, {s_path_coords.max():.2f}]")
print(f"t range: [{t.min():.2f}, {t.max():.2f}]")
print(f"t_path_coords range: [{t_path_coords.min():.2f}, {t_path_coords.max():.2f}]")

# Setup for cloud water calculation
k_s, k_t = fs['k_s'], fs['k_t']
k_s_grid = k_s.reshape(-1, 1) if k_s.ndim == 1 else k_s
k_t_grid = k_t.reshape(1, -1) if k_t.ndim == 1 else k_t
k_z = fs['k_z']
h_hat = fs['h_hat']

print(f"k_s_grid={k_s_grid.shape}, k_t_grid={k_t_grid.shape}")
print(f"k_z={k_z.shape}, h_hat={h_hat.shape}")

g_s_hat = gamma_ratio * rho_s0 * 1j * k_s_grid * U / (1 - 1j * k_z * h_v)
g_c_hat = 1.0 / (tau_c * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)
g_f_hat = 1.0 / (tau_f * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)

# Calculate f_v_wind
p_star_hat = g_s_hat * g_c_hat * g_f_hat * h_hat
p_star_pos_wind = np.fft.ifft2(p_star_hat).real[:n_s, :n_t]

q_c_star_pos_wind = np.fft.ifft2(tau_c * g_s_hat * g_c_hat * h_hat).real[:n_s, :n_t]
q_f_star_pos_wind = np.fft.ifft2(tau_f * g_s_hat * g_c_hat * g_f_hat * h_hat).real[:n_s, :n_t]
q_t_star_pos_wind = rho_s0 * h_v + q_c_star_pos_wind + q_f_star_pos_wind

d_s = s[1] - s[0]
f_v_integrand = p_star_pos_wind * d_s / (U * q_t_star_pos_wind)
f_v_integral = cumulative_trapezoid(f_v_integrand, axis=0, initial=0)
f_v_wind = (rho_s0 * h_v / q_t_star_pos_wind) * np.exp(-f_v_integral)
print(f"f_v_wind shape: {f_v_wind.shape}")

# Calculate rho_c for one z level
rho_c_star_hat_0 = (gamma_ratio * rho_s0 * 1j * k_s_grid * U / h_v) * h_hat / \
                   (kappa * (k_s_grid**2 + k_t_grid**2) + 1j * k_s_grid * U + 1.0/tau_c)

z_max = np.max(h_grid_ds) + 2 * h_s
n_z = 100
z_bar_rho_c = np.linspace(0, z_max, n_z)

print(f"\nz_max={z_max:.2f}, n_z={n_z}")
print(f"Will create rho_c with shape ({n_z}, {len(s_path_coords)})")

# Test one iteration
i = 0
exponent = (1j * k_z + 1/(2*h_rho) - 1/h_v) * z_bar_rho_c[i]
print(f"exponent shape: {exponent.shape}")

rho_c_for_z_bar = np.fft.ifft2(rho_c_star_hat_0 * np.exp(exponent)).real[:n_s, :n_t]
print(f"rho_c_for_z_bar shape: {rho_c_for_z_bar.shape}")

# Multiply
print(f"f_v_wind shape: {f_v_wind.shape}")
try:
    temp = f_v_wind * rho_c_for_z_bar
    print(f"f_v_wind * rho_c_for_z_bar success: {temp.shape}")
except Exception as e:
    print(f"f_v_wind * rho_c_for_z_bar failed: {e}")

# Interpolate to path points
print(f"\nInterpolating to path points...")
print(f"RegularGridInterpolator grid: s={len(s)}, t={len(t)}")
print(f"Query points: s_path_coords={len(s_path_coords)}, t_path_coords={len(t_path_coords)}")

try:
    interp = RegularGridInterpolator(
        (s, t), rho_c_for_z_bar,
        method='linear', bounds_error=False, fill_value=0
    )
    result = interp(np.column_stack([s_path_coords, t_path_coords]))
    print(f"Interpolation success: {result.shape}")
except Exception as e:
    print(f"Interpolation failed: {e}")
