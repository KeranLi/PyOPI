"""
Example: Parameter Fitting with OPI

This example demonstrates how to fit OPI model parameters to observational data
using the CRS3 global optimization algorithm.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi import opi_fit_one_wind, opi_fit_two_winds


def example_fit_one_wind():
    """Example of fitting one-wind model parameters."""
    print("=" * 60)
    print("EXAMPLE: One-Wind Parameter Fitting")
    print("=" * 60)
    
    # Path to .run file with constraints
    run_file = "../../runs/test_run/test_fit_one.run"
    
    if not os.path.exists(run_file):
        print(f"Run file not found: {run_file}")
        print("Please create a run file with constraints for fitting.")
        print("\nRun file format:")
        print("  Line 1: Run title")
        print("  Line 2: Parallel flag (0 or 1)")
        print("  Line 3-4: Data paths")
        print("  Line 5: Topography file")
        print("  Line 6: Tukey window size (0-1)")
        print("  Line 7: Sample file (required for fitting)")
        print("  Line 8: Continental divide file ('no' for one-wind)")
        print("  Line 9: Map limits [minLon, maxLon, minLat, maxLat]")
        print("  Line 10: Section origin (lon lat, or 'map')")
        print("  Line 11: CRS3 parameters (mu epsilon)")
        print("  Line 12: Restart file ('no' if none)")
        print("  Line 13: Parameter labels (9 labels separated by |)")
        print("  Line 14: Exponents (9 values)")
        print("  Line 15: Lower bounds (9 values)")
        print("  Line 16: Upper bounds (9 values)")
        print("  (No solution vector - will be fitted)")
        return
    
    print(f"\nFitting parameters using: {run_file}")
    print("This may take several minutes depending on the grid size...")
    
    # Run fitting
    result = opi_fit_one_wind(
        run_file_path=run_file,
        verbose=True,
        max_iterations=1000  # Limit for demonstration
    )
    
    if result['success']:
        print("\n" + "=" * 60)
        print("FITTING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nBest-fit parameters:")
        for param, value in result['solution_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nDerived parameters:")
        for param, value in result['derived_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nFinal misfit (chi-square): {result['misfit']:.6f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Results saved to: {result['results_path']}")
    else:
        print("\n" + "=" * 60)
        print("FITTING FAILED")
        print("=" * 60)
        print(f"Error: {result.get('message', 'Unknown error')}")


def example_fit_two_winds():
    """Example of fitting two-wind model parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Two-Wind Parameter Fitting")
    print("=" * 60)
    
    # Path to .run file
    run_file = "../../runs/test_run/test_fit_two.run"
    
    if not os.path.exists(run_file):
        print(f"Run file not found: {run_file}")
        print("Two-wind fitting requires:")
        print("  - Sample data")
        print("  - Continental divide file (to separate samples)")
        print("  - 19 parameters (9 for each wind + 1 fraction)")
        return
    
    print(f"\nFitting two-wind parameters using: {run_file}")
    print("This may take significantly longer than one-wind fitting...")
    
    # Run fitting
    result = opi_fit_two_winds(
        run_file_path=run_file,
        verbose=True,
        max_iterations=1000
    )
    
    if result['success']:
        print("\n" + "=" * 60)
        print("FITTING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        print(f"\nBest-fit parameters:")
        for param, value in result['solution_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nFinal misfit: {result['misfit']:.6f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Results saved to: {result['results_path']}")
    else:
        print("\nFitting failed:", result.get('message', 'Unknown error'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OPI Fitting Examples')
    parser.add_argument('--example', choices=['one', 'two', 'all'],
                       default='all', help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example in ['one', 'all']:
        example_fit_one_wind()
    
    if args.example in ['two', 'all']:
        example_fit_two_winds()
