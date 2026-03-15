import numpy as np
import h5py
from opi.io.data_loader import get_input
from opi.io.run_file import parse_run_file
from opi.physics.fourier_solution import fourier_solution
from scipy.integrate import cumulative_trapezoid

# Load actual data
print("Loading data...")
with h5py.File('../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat', 'r') as f:
    x = np.array(f['x']).flatten()
    y = np.array(f['y']).flatten()

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

# Test fourier_solution
U, azimuth, NM, f_c, h_rho = 10, 34, 0.01, 0.0001, 8000

print("Running fourier_solution...")
fs = fourier_solution(x_ds, y_ds, h_grid_ds, U, azimuth, NM, f_c, h_rho)

s = fs['s']
t = fs['t']
n_s, n_t = len(s), len(t)

print(f"fourier_solution output: s={len(s)}, t={len(t)}")
print(f"k_s shape: {fs['k_s'].shape}")
print(f"k_t shape: {fs['k_t'].shape}")
print(f"h_hat shape: {fs['h_hat'].shape}")
print(f"k_z shape: {fs['k_z'].shape}")

# Simulate cloud_water calculation
k_s = fs['k_s']
k_t = fs['k_t']
k_s_grid = k_s.reshape(-1, 1) if k_s.ndim == 1 else k_s
k_t_grid = k_t.reshape(1, -1) if k_t.ndim == 1 else k_t

print(f"k_s_grid shape: {k_s_grid.shape}")
print(f"k_t_grid shape: {k_t_grid.shape}")

gamma_ratio, rho_s0, h_v = 0.8, 0.01, 1000
tau_c, tau_f, kappa = 1000, 500, 1000

g_s_hat = gamma_ratio * rho_s0 * 1j * k_s_grid * U / (1 - 1j * fs['k_z'] * h_v)
g_c_hat = 1.0 / (tau_c * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)
g_f_hat = 1.0 / (tau_f * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)

print(f"g_s_hat shape: {g_s_hat.shape}")
print(f"g_c_hat shape: {g_c_hat.shape}")
print(f"g_f_hat shape: {g_f_hat.shape}")

p_star_hat = g_s_hat * g_c_hat * g_f_hat * fs['h_hat']
print(f"p_star_hat shape: {p_star_hat.shape}")

p_star_pos_wind = np.fft.ifft2(p_star_hat).real
print(f"p_star_pos_wind shape before slice: {p_star_pos_wind.shape}")
p_star_pos_wind = p_star_pos_wind[:n_s, :n_t]
print(f"p_star_pos_wind shape after slice: {p_star_pos_wind.shape}")

# Calculate f_v_wind
q_c_star_pos_wind = np.fft.ifft2(tau_c * g_s_hat * g_c_hat * fs['h_hat']).real
q_c_star_pos_wind = q_c_star_pos_wind[:n_s, :n_t]

q_f_star_pos_wind = np.fft.ifft2(tau_f * g_s_hat * g_c_hat * g_f_hat * fs['h_hat']).real
q_f_star_pos_wind = q_f_star_pos_wind[:n_s, :n_t]

q_t_star_pos_wind = rho_s0 * h_v + q_c_star_pos_wind + q_f_star_pos_wind
print(f"q_t_star_pos_wind shape: {q_t_star_pos_wind.shape}")

d_s = s[1] - s[0]
f_v_integrand = p_star_pos_wind * d_s / (U * q_t_star_pos_wind)
print(f"f_v_integrand shape: {f_v_integrand.shape}")

f_v_integral = cumulative_trapezoid(f_v_integrand, axis=0, initial=0)
print(f"f_v_integral shape: {f_v_integral.shape}")

f_v_wind = (rho_s0 * h_v / q_t_star_pos_wind) * np.exp(-f_v_integral)
print(f"f_v_wind shape: {f_v_wind.shape}")

# Now test the problematic calculation
rho_c_star_hat_0 = (gamma_ratio * rho_s0 * 1j * k_s_grid * U / h_v) * fs['h_hat'] / \
                   (kappa * (k_s_grid**2 + k_t_grid**2) + 1j * k_s_grid * U + 1.0/tau_c)

z_max = 8000
n_z = 100
z_bar_rho_c = np.linspace(0, z_max, n_z)

print(f"\nTesting rho_c calculation for z_bar={z_bar_rho_c[0]}...")
exponent = (1j * fs['k_z'] + 1/(2*h_rho) - 1/h_v) * z_bar_rho_c[0]
print(f"exponent shape: {exponent.shape}")

rho_c_for_z_bar = np.fft.ifft2(rho_c_star_hat_0 * np.exp(exponent)).real
print(f"rho_c_for_z_bar shape before slice: {rho_c_for_z_bar.shape}")

rho_c_for_z_bar = rho_c_for_z_bar[:n_s, :n_t]
print(f"rho_c_for_z_bar shape after slice: {rho_c_for_z_bar.shape}")
print(f"f_v_wind shape: {f_v_wind.shape}")

try:
    result = f_v_wind * rho_c_for_z_bar
    print(f"Multiplication successful: {result.shape}")
except ValueError as e:
    print(f"Multiplication failed: {e}")
