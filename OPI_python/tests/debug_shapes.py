import numpy as np
from scipy.interpolate import RegularGridInterpolator
from opi.physics.fourier_solution import fourier_solution

# Small test grid
x = np.linspace(0, 10000, 100)
y = np.linspace(0, 8000, 80)
X, Y = np.meshgrid(x, y)
h_grid = 1000 * np.exp(-((X-5000)**2 + (Y-4000)**2) / 2e6)

U, azimuth, NM, f_c, h_rho = 10, 45, 0.01, 0.0001, 8000

fs = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
print(f"s shape: {fs['s'].shape}")
print(f"t shape: {fs['t'].shape}")
print(f"k_s shape: {fs['k_s'].shape}")
print(f"k_t shape: {fs['k_t'].shape}")
print(f"h_hat shape: {fs['h_hat'].shape}")
print(f"k_z shape: {fs['k_z'].shape}")

# Check n_s, n_t
n_s, n_t = len(fs['s']), len(fs['t'])
print(f"n_s={n_s}, n_t={n_t}")

# Check k_s_grid and k_t_grid
k_s_grid = fs['k_s'].reshape(-1, 1) if fs['k_s'].ndim == 1 else fs['k_s']
k_t_grid = fs['k_t'].reshape(1, -1) if fs['k_t'].ndim == 1 else fs['k_t']
print(f"k_s_grid shape: {k_s_grid.shape}")
print(f"k_t_grid shape: {k_t_grid.shape}")

# Simulate what happens in cloud_water
from scipy.integrate import cumulative_trapezoid

gamma_ratio, rho_s0, h_v = 0.8, 0.01, 1000
tau_c, tau_f, kappa = 1000, 500, 1000

g_s_hat = gamma_ratio * rho_s0 * 1j * k_s_grid * U / (1 - 1j * fs['k_z'] * h_v)
g_c_hat = 1.0 / (tau_c * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)
g_f_hat = 1.0 / (tau_f * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)

p_star_hat = g_s_hat * g_c_hat * g_f_hat * fs['h_hat']
p_star_pos_wind = np.fft.ifft2(p_star_hat).real
p_star_pos_wind = p_star_pos_wind[:n_s, :n_t]
print(f"p_star_pos_wind shape: {p_star_pos_wind.shape}")

q_c_star_pos_wind = np.fft.ifft2(tau_c * g_s_hat * g_c_hat * fs['h_hat']).real
q_c_star_pos_wind = q_c_star_pos_wind[:n_s, :n_t]
print(f"q_c_star_pos_wind shape: {q_c_star_pos_wind.shape}")

q_f_star_pos_wind = np.fft.ifft2(tau_f * g_s_hat * g_c_hat * g_f_hat * fs['h_hat']).real
q_f_star_pos_wind = q_f_star_pos_wind[:n_s, :n_t]
print(f"q_f_star_pos_wind shape: {q_f_star_pos_wind.shape}")

q_t_star_pos_wind = rho_s0 * h_v + q_c_star_pos_wind + q_f_star_pos_wind
print(f"q_t_star_pos_wind shape: {q_t_star_pos_wind.shape}")

d_s = fs['s'][1] - fs['s'][0]
f_v_integrand = p_star_pos_wind * d_s / (U * q_t_star_pos_wind)
print(f"f_v_integrand shape: {f_v_integrand.shape}")

f_v_integral = cumulative_trapezoid(f_v_integrand, axis=0, initial=0)
print(f"f_v_integral shape: {f_v_integral.shape}")

f_v_wind = (rho_s0 * h_v / q_t_star_pos_wind) * np.exp(-f_v_integral)
print(f"f_v_wind shape: {f_v_wind.shape}")

# Now test rho_c calculation
z_max = 8000
n_z = 100
z_bar_rho_c = np.linspace(0, z_max, n_z)

rho_c_star_hat_0 = (gamma_ratio * rho_s0 * 1j * k_s_grid * U / h_v) * fs['h_hat'] / \
                   (kappa * (k_s_grid**2 + k_t_grid**2) + 1j * k_s_grid * U + 1.0/tau_c)

print(f"rho_c_star_hat_0 shape: {rho_c_star_hat_0.shape}")

# Test one iteration
i = 0
exponent = (1j * fs['k_z'] + 1/(2*h_rho) - 1/h_v) * z_bar_rho_c[i]
print(f"exponent shape: {exponent.shape}")

rho_c_for_z_bar = np.fft.ifft2(rho_c_star_hat_0 * np.exp(exponent)).real
print(f"rho_c_for_z_bar shape before slice: {rho_c_for_z_bar.shape}")

rho_c_for_z_bar = rho_c_for_z_bar[:n_s, :n_t]
print(f"rho_c_for_z_bar shape after slice: {rho_c_for_z_bar.shape}")
print(f"f_v_wind shape: {f_v_wind.shape}")

# This is the problematic line
try:
    result = f_v_wind * rho_c_for_z_bar
    print(f"Multiplication successful: {result.shape}")
except ValueError as e:
    print(f"Multiplication failed: {e}")
