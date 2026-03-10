#!/usr/bin/env python
"""
Simple OPI Simulation Example

This script demonstrates a basic OPI simulation using synthetic data.
Run with: python simple_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path if running from examples directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opi
from opi.calc_one_wind import calc_one_wind
from opi.viz import plot_topography_map, plot_precipitation_map, plot_isotope_map

print("="*60)
print("OPI Simple Simulation Example")
print("="*60)

# 1. Create synthetic topography
print("\n1. Creating synthetic topography...")
dem = opi.create_synthetic_dem(
    topo_type='gaussian',
    grid_size=(400e3, 400e3),    # 400km x 400km domain
    grid_spacing=(4000, 4000),   # 4km resolution
    lon0=0,
    lat0=45,
    amplitude=1800,              # 1800m peak
    sigma=(50e3, 50e3),          # 50km width
    output_file='data/gaussian_topo.mat'
)

x, y = dem['x'], dem['y']
h_grid = dem['hGrid']
print(f"   Grid shape: {h_grid.shape}")
print(f"   Height range: {h_grid.min():.0f} - {h_grid.max():.0f} m")

# 2. Define simulation parameters
print("\n2. Setting up simulation parameters...")
beta = np.array([
    12.0,    # U: Wind speed (m/s)
    90.0,    # azimuth: Wind from east (degrees)
    288.0,   # T0: Temperature (K) = 15°C
    0.3,     # M: Mountain height number
    0.0,     # kappa: No eddy diffusion
    1200.0,  # tau_c: Condensation time (s)
    -6e-3,   # d2h0: Base d2H (-60 permil)
    -2e-3,   # d_d2h0_d_lat: Gradient (per degree)
    0.75     # f_p0: Precipitation fraction
])

print(f"   Wind: {beta[0]:.1f} m/s from {beta[1]:.0f}°")
print(f"   Temperature: {beta[2]:.1f} K ({beta[2]-273.15:.1f}°C)")

# 3. Prepare sample data (simplified for forward calculation)
print("\n3. Preparing sample data...")
n_samples = 3
sample_x = np.array([0, 50000, -50000])
sample_y = np.array([0, 0, 0])
sample_d2h = np.array([-100, -110, -105]) * 1e-3
sample_d18o = np.array([-12.5, -13.8, -13.1]) * 1e-3

# Simple catchment indices format:
# ij_catch is a flat list of (i, j) tuples
# ptr_catch points to where each sample's data starts
n_grid = len(x)
center_j = n_grid // 2

ij_catch = [
    (50, center_j),      # Sample 0: one catchment node
    (56, center_j),      # Sample 1: one catchment node  
    (44, center_j),      # Sample 2: one catchment node
]
ptr_catch = [0, 1, 2, 3]  # Each sample has 1 node

# 4. Run simulation
print("\n4. Running OPI simulation...")
result = calc_one_wind(
    beta=beta,
    f_c=1e-4,
    h_r=540,
    x=x,
    y=y,
    lat=np.array([45.0]),
    lat0=45.0,
    h_grid=h_grid,
    b_mwl_sample=np.array([9.47e-3, 8.03]),
    ij_catch=ij_catch,
    ptr_catch=ptr_catch,
    sample_d2h=sample_d2h,
    sample_d18o=sample_d18o,
    cov=np.array([[1e-6, 0], [0, 1e-6]]),
    n_parameters_free=9,
    is_fit=False
)

# Unpack results
(chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
 rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f,
 p_grid, f_m_grid, r_h_grid, evap_d2h_grid, u_evap_d2h_grid,
 evap_d18o_grid, u_evap_d18o_grid, d2h_grid, d18o_grid,
 i_wet, d2h_pred, d18o_pred) = result

print("   [OK] Simulation complete!")
print(f"   Chi-square: {chi_r2:.4f}")
print(f"   Precipitation: {p_grid.min()*1000*86400:.1f} - {p_grid.max()*1000*86400:.1f} mm/day")
print(f"   d2H range: {d2h_grid.min()*1000:.1f} to {d2h_grid.max()*1000:.1f} permil")

# 5. Visualize results
print("\n5. Creating visualizations...")

# Create output directory
os.makedirs('output', exist_ok=True)

# Plot 1: Topography
fig, ax = plot_topography_map(
    dem['lon'], dem['lat'], h_grid,
    title='Synthetic Gaussian Mountain',
    cmap='terrain',
    save_path='output/01_topography.png'
)
plt.close()

# Plot 2: Precipitation
fig, ax = plot_precipitation_map(
    dem['lon'], dem['lat'], p_grid,
    title=f'Precipitation (U={beta[0]:.0f} m/s, Az={beta[1]:.0f}°)',
    cmap='Blues',
    save_path='output/02_precipitation.png'
)
plt.close()

# Plot 3: Isotopes
fig, axes = plot_isotope_map(
    dem['lon'], dem['lat'], d2h_grid, d18o_grid,
    title_prefix='Predicted',
    cmap='RdYlBu_r',
    save_path='output/03_isotopes.png'
)
plt.close()

print("   [OK] Figures saved to output/ directory")

# 6. Save results
print("\n6. Saving results...")
from opi.io import save_opi_results

results_dict = {
    'lon': dem['lon'],
    'lat': dem['lat'],
    'h_grid': h_grid,
    'beta': beta,
    'chi_r2': chi_r2,
    'nu': nu,
    'p_grid': p_grid,
    'd2h_grid': d2h_grid,
    'd18o_grid': d18o_grid,
    'f_m_grid': f_m_grid,
    'r_h_grid': r_h_grid,
}

save_opi_results('output/opi_results.mat', results_dict)
print("   [OK] Results saved to output/opi_results.mat")

print("\n" + "="*60)
print("Simulation complete!")
print("="*60)
print("\nOutput files:")
print("  - output/01_topography.png")
print("  - output/02_precipitation.png")
print("  - output/03_isotopes.png")
print("  - output/opi_results.mat")
print("\nNext steps:")
print("  - Try different wind speeds (beta[0])")
print("  - Try different wind directions (beta[1])")
print("  - See notebooks/ for more examples")
