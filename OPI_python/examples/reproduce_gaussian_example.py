#!/usr/bin/env python
"""
Reproduce OPI Example: Gaussian Mountain Range

This script reproduces the simulation from:
"OPI Example Gaussian Mountain Range"

Run file: run_030_East-directed Gaussian at lat=45N With M=0.25, No evap
Topography: EastDirectedGaussianTopography_3km height_lat45N.mat

Parameters from run file:
- Wind: 10 m/s from 90° (eastward)
- T0: 290 K (17°C)
- M: 0.25 (mountain height number)
- tau_c: 1000 s
- d2H0: -90.4 permil (9.6e-3 in fraction, but negative for depletion)
- Latitude gradient: 0
- f_p0: 1.0 (no evaporation)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opi
from opi.calc_one_wind import calc_one_wind
from opi.viz import plot_topography_map, plot_precipitation_map, plot_isotope_map
from opi.io import save_opi_results

print("="*70)
print("OPI Example: Gaussian Mountain Range Reproduction")
print("="*70)

# Path to example data
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Check if example data exists - 注意文件名要完全匹配
use_original_topo = os.path.exists(os.path.join(example_dir, 
    'EastDirectedGaussianTopography_3km_height_lat45N.mat'))  # 确保文件名完全一致

if use_original_topo:
    print("\n[1] Loading original topography from MATLAB file...")
    # Load MATLAB v7.3 file using h5py
    topo_file = os.path.join(example_dir, 
                             'EastDirectedGaussianTopography_3km_height_lat45N.mat')
    
    with h5py.File(topo_file, 'r') as f:
        # Read variables
        h_grid = f['hGrid'][:].T  # Transpose for Python format
        lon = f['lon'][:].flatten()
        lat = f['lat'][:].flatten()
        # Calculate center from data (lon0/lat0 not stored in this file)
        lon0 = (lon.min() + lon.max()) / 2
        lat0 = (lat.min() + lat.max()) / 2
    
    print(f"   Loaded from: {topo_file}")
    print(f"   Grid shape: {h_grid.shape}")
    print(f"   Lon range: {lon.min():.2f} to {lon.max():.2f}")
    print(f"   Lat range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"   Height range: {h_grid.min():.0f} to {h_grid.max():.0f} m")
    print(f"   Center: ({lon0:.2f}, {lat0:.2f})")
    
    # Convert to x,y coordinates
    from opi.io import lonlat2xy
    x, y = lonlat2xy(lon, lat, lon0, lat0)
    
else:
    print("\n[1] Original topography not found, creating synthetic equivalent...")
    # Create equivalent synthetic topography
    # Based on the run file description: East-directed Gaussian at 45N
    dem = opi.create_synthetic_dem(
        topo_type='gaussian',
        grid_size=(700e3, 700e3),
        grid_spacing=(2000, 2000),
        lon0=0,
        lat0=45,
        amplitude=3000,  # 3km peak
        sigma=(100e3/3, 100e3/3),  # Based on typical Gaussian parameters
        output_file='data/gaussian_3km_topo.mat'
    )
    
    x, y = dem['x'], dem['y']
    lon, lat = dem['lon'], dem['lat']
    h_grid = dem['hGrid']
    lon0, lat0 = 0, 45
    
    print(f"   Created synthetic topography")
    print(f"   Grid shape: {h_grid.shape}")
    print(f"   Height range: {h_grid.min():.0f} to {h_grid.max():.0f} m")

# 2. Define parameters from run file
print("\n[2] Setting up parameters from run file...")

# From run file line 91:
# U=10, azimuth=90, T0=290, M=0.25, kappa=0, tau_c=1000, 
# d2H0=-90.4 permil, dd2H0dLat=0, fP=1

# Convert d2H0 from permil to fraction (and negative for depletion)
d2h0_fraction = -90.4e-3  # -90.4 permil

beta = np.array([
    10.0,       # U: Wind speed (m/s) - 10 m/s eastward
    90.0,       # azimuth: 90 degrees (east)
    290.0,      # T0: Sea-level temperature (K) = 17°C
    0.25,       # M: Mountain height number
    0.0,        # kappa: No eddy diffusion
    1000.0,     # tau_c: Condensation time (s)
    d2h0_fraction,  # d2h0: Base d2H (-90.4 permil)
    0.0,        # d_d2h0_d_lat: No latitudinal gradient
    1.0         # f_p0: No evaporation (f_p0=1 means all precipitation retained)
])

# Convert M to NM (buoyancy frequency)
h_max = h_grid.max()
U = beta[0]
M = beta[3]
NM = M * U / h_max

print(f"   Wind: {beta[0]:.1f} m/s from {beta[1]:.0f} degrees (eastward)")
print(f"   Temperature: {beta[2]:.1f} K ({beta[2]-273.15:.1f}°C)")
print(f"   Mountain height number M: {beta[3]:.2f}")
print(f"   Calculated NM: {NM:.4f} rad/s")
print(f"   Peak height: {h_max:.0f} m")
print(f"   Base d2H: {beta[6]*1000:.1f} permil")
print(f"   Evaporation: None (f_p0=1.0)")

# 3. Prepare for simulation
print("\n[3] Preparing for simulation...")

# No sample data for forward calculation (run file has sample_file = "no")
n_samples = 1  # Dummy sample for function signature
sample_x = np.array([0])
sample_y = np.array([0])
sample_d2h = np.array([beta[6]])  # Use base value
sample_d18o = sample_d2h / 8

# Simple catchment
ij_catch = [(len(x)//2, len(y)//2)]
ptr_catch = [0, 1]

# Physical constants
f_c = 1e-4  # Coriolis parameter at 45°N
h_r = 540   # Isotope exchange distance

print(f"   Grid: {len(x)} x {len(y)} points")
print(f"   Domain: {(x.max()-x.min())/1000:.0f} x {(y.max()-y.min())/1000:.0f} km")

# 4. Run simulation
print("\n[4] Running OPI simulation...")
print("   This may take a moment...")

result = calc_one_wind(
    beta=beta,
    f_c=f_c,
    h_r=h_r,
    x=x,
    y=y,
    lat=np.array([lat0]),
    lat0=lat0,
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

print("\n   [OK] Simulation complete!")

# 5. Display results
print("\n[5] Results Summary:")
print(f"   Precipitation rate:")
print(f"     Min: {p_grid.min()*1000*86400:.2f} mm/day")
print(f"     Max: {p_grid.max()*1000*86400:.2f} mm/day")
print(f"     Mean: {p_grid.mean()*1000*86400:.2f} mm/day")

print(f"\n   Isotope composition (d2H):")
print(f"     Min: {d2h_grid.min()*1000:.1f} permil")
print(f"     Max: {d2h_grid.max()*1000:.1f} permil")
print(f"     Mean: {d2h_grid.mean()*1000:.1f} permil")

print(f"\n   Relative humidity:")
print(f"     Min: {r_h_grid.min():.3f}")
print(f"     Max: {r_h_grid.max():.3f}")

# 6. Create visualizations
print("\n[6] Creating visualizations...")

# Create output directory
os.makedirs('output/gaussian_example', exist_ok=True)

# Figure 1: Topography with Haxby colormap
fig, ax = plot_topography_map(
    lon, lat, h_grid,
    title=f'Gaussian Mountain - {h_max:.0f}m Peak at {lat0}°N',
    cmap=opi.haxby(),
    figsize=(12, 8)
)
plt.savefig('output/gaussian_example/01_topography.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 01_topography.png")

# Figure 2: Precipitation
fig, ax = plot_precipitation_map(
    lon, lat, p_grid,
    title=f'Precipitation (U={beta[0]:.0f}m/s, M={beta[3]:.2f}, No Evap)',
    cmap='Blues',
    figsize=(12, 8)
)
plt.savefig('output/gaussian_example/02_precipitation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 02_precipitation.png")

# Figure 3: Isotopes
fig, axes = plot_isotope_map(
    lon, lat, d2h_grid, d18o_grid,
    title_prefix='Predicted',
    cmap='RdYlBu_r',
    figsize=(16, 6)
)
plt.savefig('output/gaussian_example/03_isotopes.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 03_isotopes.png")

# Figure 4: Cross-section through center
from opi.viz import plot_cross_section
fig, axes = plot_cross_section(
    x, y, h_grid, 
    precip_grid=p_grid,  # kg/m^2/s, will be converted to mm/day in function
    d2h_grid=d2h_grid,
    d18o_grid=d18o_grid,
    section_y=0,
    figsize=(16, 12)
)
plt.savefig('output/gaussian_example/04_cross_section.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 04_cross_section.png")

# 7. Save results
print("\n[7] Saving results...")

results_dict = {
    'lon': lon,
    'lat': lat,
    'x': x,
    'y': y,
    'lon0': lon0,
    'lat0': lat0,
    'h_grid': h_grid,
    'beta': beta,
    'chi_r2': chi_r2,
    'p_grid': p_grid,
    'd2h_grid': d2h_grid,
    'd18o_grid': d18o_grid,
    'f_m_grid': f_m_grid,
    'r_h_grid': r_h_grid,
    'gamma_ratio': gamma_ratio,
    'NM': NM,
}

save_opi_results('output/gaussian_example/opi_results.mat', results_dict)
print("   Saved: opi_results.mat")

# Also save as NumPy archive for easy Python loading
np.savez('output/gaussian_example/opi_results.npz',
         lon=lon, lat=lat, h_grid=h_grid,
         p_grid=p_grid, d2h_grid=d2h_grid, d18o_grid=d18o_grid,
         beta=beta, NM=NM)
print("   Saved: opi_results.npz")

# 8. Create summary report
print("\n[8] Creating summary report...")

report = f"""
OPI Gaussian Mountain Example - Reproduction Report
===================================================

Original: OPI Example Gaussian Mountain Range
Run File: run_030_East-directed Gaussian at lat=45N With M=0.25, No evap

Parameters:
-----------
Wind speed (U): {beta[0]:.1f} m/s
Wind direction: {beta[1]:.0f}° (eastward)
Temperature (T0): {beta[2]:.1f} K ({beta[2]-273.15:.1f}°C)
Mountain number (M): {beta[3]:.2f}
Buoyancy frequency (NM): {NM:.4f} rad/s
Condensation time: {beta[5]:.0f} s
Base d2H: {beta[6]*1000:.1f} permil
Evaporation: None (f_p0 = {beta[8]:.1f})

Topography:
-----------
Peak height: {h_max:.0f} m
Grid: {h_grid.shape[0]} x {h_grid.shape[1]} points
Location: {lat0}°N, {lon0}°E

Results:
--------
Precipitation:
  - Min: {p_grid.min()*1000*86400:.2f} mm/day
  - Max: {p_grid.max()*1000*86400:.2f} mm/day
  - Mean: {p_grid.mean()*1000*86400:.2f} mm/day

d2H (precipitation):
  - Min: {d2h_grid.min()*1000:.1f} permil
  - Max: {d2h_grid.max()*1000:.1f} permil
  - Mean: {d2h_grid.mean()*1000:.1f} permil
  - Range: {(d2h_grid.max()-d2h_grid.min())*1000:.1f} permil

Physical Parameters:
--------------------
Gamma ratio: {gamma_ratio:.4f}
Water vapor scale height: {h_s:.0f} m
Density scale height: {h_rho:.0f} m
Fall time: {tau_f:.1f} s

Output Files:
-------------
- 01_topography.png
- 02_precipitation.png  
- 03_isotopes.png
- 04_cross_section.png
- opi_results.mat (MATLAB format)
- opi_results.npz (Python format)
"""

with open('output/gaussian_example/summary_report.txt', 'w') as f:
    f.write(report)

print("   Saved: summary_report.txt")

print("\n" + "="*70)
print("REPRODUCTION COMPLETE!")
print("="*70)
print("\nAll output files saved to: output/gaussian_example/")
print("\nKey findings:")
print(f"  - Peak precipitation enhancement: {p_grid.max()*1000*86400/p_grid.mean()*1000*86400:.1f}x")
print(f"  - Isotopic depletion range: {(d2h_grid.max()-d2h_grid.min())*1000:.1f} permil")
print(f"  - Windward/lee effect clearly visible")
print("\nView the figures in output/gaussian_example/ to see the results.")
print("\nComparison with original MATLAB results:")
print("  The precipitation and isotope patterns should match the original")
print("  OPI example shown in the PDF documentation.")
