#!/usr/bin/env python
"""
Debug script to trace dimensions through OPI calculation
"""
import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opi.physics.fourier import fourier_solution, wind_grid
from opi.physics.precipitation import precipitation_grid
from opi.physics.thermodynamics import base_state

print("="*70)
print("Dimension Debugging for OPI")
print("="*70)

# Load topography
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           '..', 'OPI Example Gaussian Mountain Range')
topo_file = os.path.join(example_dir, 'data', 
                         'EastDirectedGaussianTopography_3km height_lat45N.mat')

print(f"\n[1] Loading topography from: {topo_file}")
with h5py.File(topo_file, 'r') as f:
    h_grid = f['hGrid'][:].T  # This is what reproduce_gaussian_example.py does
    lon = f['lon'][:].flatten()
    lat = f['lat'][:].flatten()

print(f"  h_grid shape: {h_grid.shape}")
print(f"  lon shape: {lon.shape}, range: [{lon.min():.2f}, {lon.max():.2f}]")
print(f"  lat shape: {lat.shape}, range: [{lat.min():.2f}, {lat.max():.2f}]")

# Convert to x,y
from opi.io.coordinates import lonlat2xy
lon0, lat0 = 0.0, 45.0
x, y = lonlat2xy(lon, lat, lon0, lat0)

print(f"\n[2] Converted to Cartesian coordinates:")
print(f"  x shape: {x.shape}, range: [{x.min():.0f}, {x.max():.0f}] m")
print(f"  y shape: {y.shape}, range: [{y.min():.0f}, {y.max():.0f}] m")

# Check expected vs actual h_grid shape
print(f"\n[3] Checking h_grid dimensions:")
print(f"  Expected shape for (y, x) order: ({len(y)}, {len(x)})")
print(f"  Actual shape: {h_grid.shape}")
if h_grid.shape == (len(y), len(x)):
    print("  [OK] h_grid is in correct (y, x) order")
elif h_grid.shape == (len(x), len(y)):
    print("  [WARNING] h_grid is in (x, y) order - needs transpose!")
else:
    print(f"  [ERROR] h_grid shape doesn't match either expected shape!")

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

print(f"\n[4] Parameters:")
print(f"  U = {U} m/s, azimuth = {azimuth}°")
print(f"  NM = {NM:.6f} rad/s")

# Test wind_grid
print(f"\n[5] Testing wind_grid with azimuth={azimuth}°:")
Sxy, Txy, s, t, Xst, Yst = wind_grid(x, y, azimuth)
print(f"  Sxy shape: {Sxy.shape} (should be {len(y)} x {len(x)})")
print(f"  Txy shape: {Txy.shape}")
print(f"  len(s) = {len(s)}")
print(f"  len(t) = {len(t)}")
print(f"  Xst shape: {Xst.shape} (should be {len(s)} x {len(t)})")
print(f"  Yst shape: {Yst.shape}")

# Check what happens with azimuth=90 (east wind)
print(f"\n[6] Checking coordinate transformation for east wind:")
print(f"  Sxy[0,0] = {Sxy[0,0]:.1f}, X[0,0] = {x[0]:.1f}")
print(f"  Txy[0,0] = {Txy[0,0]:.1f}, Y[0,0] = {y[0]:.1f}")
print(f"  For east wind, Sxy should ~ X and Txy should ~ Y")

# Test fourier_solution
print(f"\n[7] Testing fourier_solution:")
result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
print(f"  h_wind shape: {result['h_wind'].shape}")
print(f"  h_hat shape: {result['h_hat'].shape}")
print(f"  k_z shape: {result['k_z'].shape}")
print(f"  k_s length: {len(result['k_s'])}")
print(f"  k_t length: {len(result['k_t'])}")

# Check h_wind range
print(f"  h_wind range: [{result['h_wind'].min():.1f}, {result['h_wind'].max():.1f}]")

# Test base_state
print(f"\n[8] Testing base_state:")
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho_out = base_state(NM, T0)
print(f"  z_bar length: {len(z_bar)}")
print(f"  rho_s0: {rho_s0:.6f} kg/m3")
print(f"  h_s: {h_s:.1f} m")

# Test precipitation_grid
print(f"\n[9] Testing precipitation_grid:")
precip_result = precipitation_grid(
    x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c,
    h_rho, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
    rho_s0, h_s, f_p0
)
print(f"  p_grid shape: {precip_result['p_grid'].shape}")
print(f"  p_grid range (kg/m2/s): [{precip_result['p_grid'].min():.6f}, {precip_result['p_grid'].max():.6f}]")
print(f"  p_grid range: [{precip_result['p_grid'].min()*1000*86400:.2f}, {precip_result['p_grid'].max()*1000*86400:.2f}] mm/day")

# Check if precipitation pattern is reasonable
p_grid = precip_result['p_grid']
non_zero = np.sum(p_grid > 0.001)
total = p_grid.size
print(f"  Non-zero precipitation cells: {non_zero} / {total} ({100*non_zero/total:.1f}%)")

# Find location of max precipitation
max_idx = np.unravel_index(np.argmax(p_grid), p_grid.shape)
print(f"  Max precipitation at index: {max_idx}")
print(f"  Max precipitation at (lat, lon): ({lat[max_idx[0]]:.2f}, {lon[max_idx[1]]:.2f})")

print("\n" + "="*70)
print("Dimension debugging complete")
print("="*70)
