"""
OPI Simulation Demo

This script demonstrates running an OPI simulation with synthetic data.
"""

import numpy as np
import os
import sys
from scipy.io import savemat

# Add opi to path
sys.path.insert(0, os.path.dirname(__file__))

from opi import run_simulation, opi_calc_one_wind, opi_calc_two_winds
from opi.io.run_file import write_run_file


def create_synthetic_topography(output_path, nx=100, ny=100):
    """Create synthetic Gaussian mountain topography"""
    print("Creating synthetic topography...")
    
    # Create grid (100km x 100km)
    x = np.linspace(-50000, 50000, nx)  # meters
    y = np.linspace(-50000, 50000, ny)
    
    # Create longitude/latitude (approximate)
    lon0, lat0 = 90.0, 30.0  # Center around 90°E, 30°N
    lon = lon0 + x / 111320  # Convert meters to degrees
    lat = lat0 + y / 111320
    
    # Create Gaussian mountain (2000m peak)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Save as .mat file (MATLAB format)
    savemat(output_path, {
        'lon': lon,
        'lat': lat,
        'hGrid': h_grid,
        'x': x,
        'y': y
    })
    
    print(f"  Topography saved to: {output_path}")
    print(f"  Grid size: {nx} x {ny}")
    print(f"  Max elevation: {h_grid.max():.0f} m")
    print(f"  Center: ({lon0}°E, {lat0}°N)")
    
    return output_path


def create_one_wind_run_file(run_dir):
    """Create a one-wind run file"""
    print("\nCreating one-wind run file...")
    
    run_data = {
        'run_title': 'Demo One-Wind Simulation',
        'is_parallel': False,
        'data_path': os.path.join(run_dir, 'data'),
        'topo_file': 'topography.mat',
        'r_tukey': 0.5,
        'sample_file': None,  # No samples for pure simulation
        'cont_divide_file': None,
        'map_limits': [89.5, 90.5, 29.5, 30.5],
        'section_lon0': None,
        'section_lat0': None,
        'mu': 10,
        'epsilon0': 1e-4,
        'restart_file': None,
        'parameter_labels': ['U', 'Azimuth', 'T0', 'M', 'kappa', 'tauC', 'd2H0', 'dLat', 'fP'],
        'exponents': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'l_b': [0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0.5],
        'u_b': [20, 90, 300, 1, 1e6, 2500, 0, 0, 1],
        'beta': [10.0, 45.0, 290.0, 0.3, 10000.0, 1000.0, -0.003, -0.001, 1.0]
    }
    
    run_file_path = os.path.join(run_dir, 'demo_one_wind.run')
    write_run_file(run_file_path, run_data)
    
    print(f"  Run file saved to: {run_file_path}")
    
    return run_file_path


def demo_direct_api():
    """Run simulation using direct API (no run file needed)"""
    print("\n" + "="*60)
    print("DEMO 1: Direct API Usage (One-Wind)")
    print("="*60)
    
    # Create synthetic data
    work_dir = 'runs/demo_python'
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'data'), exist_ok=True)
    
    topo_file = create_synthetic_topography(
        os.path.join(work_dir, 'data', 'topography.mat')
    )
    
    # Run simulation directly
    print("\nRunning one-wind simulation...")
    
    solution_vector = [10.0, 45.0, 290.0, 0.3, 10000.0, 1000.0, -0.003, -0.001, 1.0]
    
    result = opi_calc_one_wind(
        run_file_path=None,  # No run file - use synthetic data
        solution_vector=solution_vector,
        verbose=True
    )
    
    print("\n" + "-"*60)
    print("SIMULATION COMPLETE")
    print("-"*60)
    
    if result.get('results'):
        p_grid = result['results']['precipitation']
        d2h_grid = result['results']['d2h']
        d18o_grid = result['results']['d18o']
        
        print(f"\nResults Summary:")
        print(f"  Precipitation range: {p_grid.min():.6f} to {p_grid.max():.6f} kg/m^2/s")
        print(f"  d2H range: {d2h_grid.min()*1000:.1f} to {d2h_grid.max()*1000:.1f} permil")
        print(f"  d18O range: {d18o_grid.min()*1000:.1f} to {d18o_grid.max()*1000:.1f} permil")
        
        # Save results
        results_file = os.path.join(work_dir, 'opiCalc_OneWind_Results.mat')
        savemat(results_file, {
            'beta': np.array(solution_vector),
            'pGrid': p_grid,
            'd2HGrid': d2h_grid,
            'd18OGrid': d18o_grid,
            'hGrid': result['results']['h_grid'],
            'x': result['results']['x'],
            'y': result['results']['y']
        })
        print(f"\n  Results saved to: {results_file}")
    
    return result


def demo_run_file():
    """Run simulation using run file"""
    print("\n" + "="*60)
    print("DEMO 2: Using Run File")
    print("="*60)
    
    # Create work directory
    work_dir = 'runs/demo_runfile'
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'data'), exist_ok=True)
    
    # Create topography
    topo_file = create_synthetic_topography(
        os.path.join(work_dir, 'data', 'topography.mat'),
        nx=80, ny=80
    )
    
    # Create run file
    run_file = create_one_wind_run_file(work_dir)
    
    # Run simulation
    print("\nRunning simulation from run file...")
    result = run_simulation(run_file, verbose=True)
    
    if result['success']:
        print("\n" + "-"*60)
        print("SIMULATION COMPLETE")
        print("-"*60)
        print(f"Results file: {result['results_path']}")
        print(f"Log file: {result['log_path']}")
        
        if not np.isnan(result.get('chi_r2', np.nan)):
            print(f"Chi-square: {result['chi_r2']:.4f}")
    else:
        print(f"\nSimulation failed: {result.get('error', 'Unknown error')}")
    
    return result


def demo_two_winds():
    """Run two-wind simulation"""
    print("\n" + "="*60)
    print("DEMO 3: Two-Wind Simulation")
    print("="*60)
    
    work_dir = 'runs/demo_two_winds'
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'data'), exist_ok=True)
    
    # Create topography
    topo_file = create_synthetic_topography(
        os.path.join(work_dir, 'data', 'topography.mat'),
        nx=80, ny=80
    )
    
    # Two-wind solution vector (19 parameters)
    solution_vector = [
        # Wind 1 (9 parameters)
        8.0, 45.0, 290.0, 0.25, 5000.0, 800.0, -0.002, -0.0005, 1.0,
        # Fraction
        0.6,
        # Wind 2 (9 parameters)
        12.0, 225.0, 292.0, 0.35, 8000.0, 1200.0, -0.004, -0.0008, 1.0
    ]
    
    print("\nRunning two-wind simulation...")
    result = opi_calc_two_winds(
        run_file_path=None,
        solution_vector=solution_vector,
        verbose=True
    )
    
    print("\n" + "-"*60)
    print("SIMULATION COMPLETE")
    print("-"*60)
    
    if result.get('precipitation') is not None:
        print(f"\nResults Summary:")
        print(f"  Combined precipitation range: {result['precipitation'].min():.6f} to {result['precipitation'].max():.6f} kg/m²/s")
        print(f"  d2H range: {result['isotope'].min()*1000:.1f} to {result['isotope'].max()*1000:.1f} ‰")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OPI Simulation Demo')
    parser.add_argument('--demo', choices=['api', 'runfile', 'two', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.demo in ['api', 'all']:
        demo_direct_api()
    
    if args.demo in ['runfile', 'all']:
        demo_run_file()
    
    if args.demo in ['two', 'all']:
        demo_two_winds()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)
