"""
Cleanup script to remove old deprecated files from OPI package.

This script removes old files that have been replaced by the new modular structure.
Run this after confirming the new structure works correctly.
"""

import os
import shutil
from pathlib import Path

# List of old files to remove (deprecated, replaced by new structure)
OLD_FILES = [
    # Core functions (now in opi/core/)
    'opi/base_state.py',
    'opi/fourier_solution.py',
    'opi/precipitation_grid.py',
    'opi/isotope_grid.py',
    'opi/fractionation_hydrogen.py',
    'opi/fractionation_oxygen.py',
    'opi/saturated_vapor_pressure.py',
    'opi/coordinates.py',
    'opi/catchment_nodes.py',
    'opi/catchment_indices.py',
    'opi/wind_path.py',
    'opi/lifting.py',
    
    # Models (now in opi/models/)
    'opi/calc_one_wind.py',
    'opi/calc_two_winds.py',
    'opi/opi_calc_one_wind.py',
    'opi/opi_calc_two_winds.py',
    'opi/opi_fit_one_wind.py',
    'opi/opi_fit_two_winds.py',
    
    # Solvers (now in opi/solvers/)
    'opi/fmin_crs3.py',
    
    # IO (now in opi/io/)
    'opi/get_input.py',
    
    # Utilities (now in opi/utils/)
    'opi/utils.py',
    
    # Maps/Plots (now in opi/visualization/)
    'opi/opi_maps_one_wind.py',
    'opi/opi_maps_two_winds.py',
    'opi/opi_plots_one_wind.py',
    'opi/run_opi_simulation.py',
]

# Old root-level example files (keep examples/ directory versions)
OLD_EXAMPLES = [
    'functionality_comparison.py',
    'plot_results.py',
    'reproduce_with_real_data.py',
    'run_demo_simulation.py',
    'run_simple_simulation.py',
]


def remove_file(filepath):
    """Remove a file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"  Removed: {filepath}")
        return True
    return False


def main():
    print("=" * 60)
    print("OPI Package Cleanup")
    print("=" * 60)
    print("\nThis script removes old deprecated files.")
    print("New structure is in opi/core/, opi/models/, etc.\n")
    
    # Check if we're in the right directory
    if not os.path.exists('opi'):
        print("Error: Run this script from OPI_python/ directory")
        return
    
    # Count files before
    removed_count = 0
    not_found_count = 0
    
    print("Removing old opi/ files...")
    for filepath in OLD_FILES:
        if remove_file(filepath):
            removed_count += 1
        else:
            not_found_count += 1
    
    print(f"\n  Total removed: {removed_count}")
    print(f"  Already gone: {not_found_count}")
    
    # Clean up old example files at root
    print("\nCleaning old example files...")
    for filepath in OLD_EXAMPLES:
        if remove_file(filepath):
            removed_count += 1
    
    # Clean up __pycache__ directories
    print("\nCleaning __pycache__ directories...")
    pycache_count = 0
    for root, dirs, files in os.walk('.', topdown=False):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"  Removed: {pycache_path}")
                    pycache_count += 1
                except Exception as e:
                    print(f"  Error removing {pycache_path}: {e}")
    
    print(f"\n  Removed {pycache_count} __pycache__ directories")
    
    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary")
    print("=" * 60)
    print(f"Files removed: {removed_count}")
    print("\nRemaining structure:")
    print("  opi/core/          - Core computations (9 modules)")
    print("  opi/models/        - Model interfaces (2 modules)")
    print("  opi/solvers/       - Optimization (1 module)")
    print("  opi/io/            - I/O utilities (2 modules)")
    print("  opi/utils/         - Utilities (3 modules)")
    print("  opi/visualization/ - Visualization (4 modules)")
    print("  opi/parallel/      - Parallel processing (2 modules)")
    print("\nRun 'python -c \"import opi\"' to verify the package works.")


if __name__ == "__main__":
    main()
