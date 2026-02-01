#!/usr/bin/env python
"""
Debug precipitation calculation
"""
import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opi.physics.fourier import fourier_solution
from opi.physics.thermodynamics import base_state

# Load topography
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           '..', 'OPI Example Gaussian Mountain Range')
topo_file = os.path.join(example_dir, 'data', 
                         'EastDirectedGaussianTopography_3km height_lat45N.mat')

print("Loading topography...")
with h5py.File(topo_file, 'r') as f:
    h_grid = f['hGrid'][:].T
    lon = f['lon'][:].flatten()
    lat = f['lat'][:].flatten()

from opi.io.coordinates import lonlat2xy
lon0, lat0 = 0.0, 45.0
x, y = lonlat2xy(lon, lat, lon0, lat0)

# Parameters
U = 10.0
azimuth = 90.0
T0 = 290.0
M = 0.25
kappa = 0.0
tau_c = 1000.0
f_p0 = 1.0
h_max = h_grid.max()
NM = M * U / h_max
f_c = 1e-4
h_rho = 8000.0

print(f"Parameters: U={U}, azimuth={azimuth}, NM={NM:.6f}")

# Get Fourier solution
print("\nFourier solution...")
result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
s = result['s']
t = result['t']
Sxy = result['Sxy']
Txy = result['Txy']
h_wind = result['h_wind']
h_hat = result['h_hat']
k_s = result['k_s']
k_t = result['k_t']
k_z = result['k_z']

print(f"h_wind shape: {h_wind.shape}")
print(f"h_wind range: [{h_wind.min():.1f}, {h_wind.max():.1f}]")
print(f"h_hat shape: {h_hat.shape}")
print(f"h_hat max magnitude: {np.max(np.abs(h_hat)):.2e}")

# Get base state
print("\nBase state...")
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho_out = base_state(NM, T0)
print(f"rho_s0: {rho_s0:.6f}")
print(f"h_s: {h_s:.1f}")
print(f"gamma_ratio: {gamma_ratio:.4f}")

# Parameters for wind grid
d_s = s[1] - s[0]
n_s, n_t = h_wind.shape
n_s_pad, n_t_pad = h_hat.shape

print(f"\nGrid parameters:")
print(f"d_s: {d_s:.1f}")
print(f"n_s: {n_s}, n_t: {n_t}")
print(f"n_s_pad: {n_s_pad}, n_t_pad: {n_t_pad}")

# Calculate isotherm heights (simplified)
print("\nCalculating precipitation components...")

# Reshape k_s to column vector
k_s_col = k_s.reshape(-1, 1)

# Check k_z
print(f"k_z shape: {k_z.shape}")
print(f"k_z real range: [{np.real(k_z).min():.6f}, {np.real(k_z).max():.6f}]")
print(f"k_z imag range: [{np.imag(k_z).min():.6f}, {np.imag(k_z).max():.6f}]")

# Calculate GS_hat
GS_hat = (gamma_ratio * rho_s0 * 1j * k_s_col * U / 
          (1 - h_s * (1j * k_z + 1/(2*h_rho))))

# FIXME: Temporary fix for precipitation scaling issue
scale_factor = 1e-4
GS_hat = GS_hat * scale_factor

print(f"GS_hat max magnitude: {np.max(np.abs(GS_hat)):.2e}")

# For dry simulation, we need tau_f
# Estimate tau_f (simplified)
tau_f = 500.0  # Typical value
print(f"Using tau_f: {tau_f}")

# Calculate GC_hat and GF_hat
GC_hat = 1.0 / (tau_c * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)
GF_hat = 1.0 / (tau_f * (kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 1)
print(f"GC_hat max magnitude: {np.max(np.abs(GC_hat)):.2e}")
print(f"GF_hat max magnitude: {np.max(np.abs(GF_hat)):.2e}")

# Calculate p_star_hat
p_star_hat = GS_hat * GC_hat * GF_hat * h_hat
print(f"p_star_hat max magnitude: {np.max(np.abs(p_star_hat)):.2e}")

# Transform to space domain
p_star_pos_wind = np.fft.ifft2(p_star_hat)
p_star_pos_wind = np.real(p_star_pos_wind)
p_star_pos_wind = p_star_pos_wind[:n_s, :n_t]
p_star_pos_wind[p_star_pos_wind < 0] = 0

print(f"p_star_pos_wind shape: {p_star_pos_wind.shape}")
print(f"p_star_pos_wind range: [{p_star_pos_wind.min():.6f}, {p_star_pos_wind.max():.6f}]")
print(f"p_star_pos_wind non-zero: {np.sum(p_star_pos_wind > 0.0001)}")

# Calculate moisture components
QC_star_pos_wind = np.fft.ifft2(tau_c * GS_hat * GC_hat * h_hat)
QC_star_pos_wind = np.real(QC_star_pos_wind)
QC_star_pos_wind = QC_star_pos_wind[:n_s, :n_t]
QC_star_pos_wind[QC_star_pos_wind < 0] = 0

QF_star_pos_wind = np.fft.ifft2(tau_f * GS_hat * GC_hat * GF_hat * h_hat)
QF_star_pos_wind = np.real(QF_star_pos_wind)
QF_star_pos_wind = QF_star_pos_wind[:n_s, :n_t]
QF_star_pos_wind[QF_star_pos_wind < 0] = 0

print(f"QC_star range: [{QC_star_pos_wind.min():.6f}, {QC_star_pos_wind.max():.6f}]")
print(f"QF_star range: [{QF_star_pos_wind.min():.6f}, {QF_star_pos_wind.max():.6f}]")

QT_star_pos_wind = rho_s0 * h_s + QC_star_pos_wind + QF_star_pos_wind
print(f"QT_star range: [{QT_star_pos_wind.min():.6f}, {QT_star_pos_wind.max():.6f}]")

# No evaporation
f_p_wind = np.ones((n_s, n_t))

# Integrate
integrand = f_p_wind * p_star_pos_wind * d_s / QT_star_pos_wind
f_v_wind = (rho_s0 * h_s / QT_star_pos_wind) * np.exp(-(1.0/U) * np.cumsum(integrand, axis=0) * d_s)

print(f"f_v_wind range: [{f_v_wind.min():.6f}, {f_v_wind.max():.6f}]")

# Final precipitation
p_wind = f_v_wind * p_star_pos_wind
print(f"p_wind range: [{p_wind.min():.6f}, {p_wind.max():.6f}]")
print(f"p_wind non-zero: {np.sum(p_wind > 0.0001)}")

# Interpolate back to geographic grid
from scipy.interpolate import RegularGridInterpolator
interp_func = RegularGridInterpolator((s, t), p_wind, method='linear', bounds_error=False, fill_value=0)
p_grid = interp_func(np.column_stack([Sxy.ravel(), Txy.ravel()])).reshape(Sxy.shape)

print(f"\nFinal p_grid:")
print(f"p_grid shape: {p_grid.shape}")
print(f"p_grid range (kg/m2/s): [{p_grid.min():.6f}, {p_grid.max():.6f}]")
print(f"p_grid range (mm/day): [{p_grid.min()*1000*86400:.2f}, {p_grid.max()*1000*86400:.2f}]")
print(f"p_grid non-zero cells: {np.sum(p_grid > 0.001)} / {p_grid.size}")

# Find max location
max_idx = np.unravel_index(np.argmax(p_grid), p_grid.shape)
print(f"Max at index: {max_idx}")
print(f"Max at (lat, lon): ({lat[max_idx[0]]:.2f}, {lon[max_idx[1]]:.2f})")
