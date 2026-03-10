"""
Example: Running OPI Simulation from .run File

This example demonstrates how to run an OPI simulation from a .run configuration file,
replicating the MATLAB workflow.

The workflow:
1. Parse the .run file
2. Load topography and sample data
3. Run the calculation (one-wind or two-wind)
4. Save results to .mat file
5. Generate log file
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi import run_simulation, opi_maps_one_wind, opi_maps_two_winds


def example_one_wind():
    """Example of running a one-wind simulation."""
    print("=" * 60)
    print("EXAMPLE: One-Wind Simulation")
    print("=" * 60)
    
    # Path to .run file
    # Replace with your actual run file path
    run_file = "../../runs/test_run/test_run.run"
    
    if not os.path.exists(run_file):
        print(f"Run file not found: {run_file}")
        print("Please create a run file or update the path.")
        return
    
    # Run simulation
    print(f"\nRunning simulation from: {run_file}")
    result = run_simulation(run_file, verbose=True)
    
    if result['success']:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results file: {result['results_path']}")
        print(f"Log file: {result['log_path']}")
        print(f"Chi-square: {result.get('chi_r2', 'N/A')}")
        
        # Generate maps
        print("\nGenerating maps...")
        try:
            map_files = opi_maps_one_wind(
                results_file=result['results_path'],
                save_plots=True,
                show_plots=False,
                verbose=True
            )
            print(f"\nGenerated {len(map_files)} map files")
        except Exception as e:
            print(f"Map generation failed: {e}")
    else:
        print("\n" + "=" * 60)
        print("SIMULATION FAILED")
        print("=" * 60)
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"See log: {result.get('log_path', 'N/A')}")


def example_two_winds():
    """Example of running a two-wind simulation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Two-Wind Simulation")
    print("=" * 60)
    
    # Path to .run file
    run_file = "../../runs/test_run/test_run_two_winds.run"
    
    if not os.path.exists(run_file):
        print(f"Run file not found: {run_file}")
        print("Two-wind run file not available for this example.")
        return
    
    # Run simulation
    print(f"\nRunning simulation from: {run_file}")
    result = run_simulation(run_file, verbose=True)
    
    if result['success']:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results file: {result['results_path']}")
        
        # Generate maps
        print("\nGenerating maps...")
        try:
            map_files = opi_maps_two_winds(
                results_file=result['results_path'],
                save_plots=True,
                show_plots=False,
                verbose=True
            )
            print(f"\nGenerated {len(map_files)} map files")
        except Exception as e:
            print(f"Map generation failed: {e}")
    else:
        print("\nSimulation failed:", result.get('error', 'Unknown error'))


def example_direct_api():
    """Example of using the API directly without .run files."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Direct API Usage")
    print("=" * 60)
    
    import numpy as np
    from opi import opi_calc_one_wind
    
    # Create synthetic parameters
    solution_vector = [10.0, 90.0, 290.0, 0.25, 10000.0, 1000.0, -5e-3, -2e-3, 0.7]
    
    print("\nRunning calculation with direct parameters...")
    print(f"Solution vector: {solution_vector}")
    
    # Run calculation (without run file, uses synthetic data)
    result = opi_calc_one_wind(solution_vector=solution_vector, verbose=True)
    
    print("\n" + "=" * 60)
    print("CALCULATION COMPLETED")
    print("=" * 60)
    print(f"Solution parameters: {result['solution_params']}")
    print(f"Derived parameters: {result['derived_params']}")
    
    if result['results']:
        print(f"\nPrecipitation shape: {result['results']['precipitation'].shape}")
        print(f"d2H range: {np.min(result['results']['d2h'])*1000:.1f} to {np.max(result['results']['d2h'])*1000:.1f} permil")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OPI Simulation Examples')
    parser.add_argument('--example', choices=['one', 'two', 'direct', 'all'],
                       default='all', help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example in ['one', 'all']:
        example_one_wind()
    
    if args.example in ['two', 'all']:
        example_two_winds()
    
    if args.example in ['direct', 'all']:
        example_direct_api()
