"""
Verification script for OPI package structure.
Checks that all modules are properly organized and importable.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {description}: {filepath}")
    return exists

def main():
    print("=" * 70)
    print("OPI Package Structure Verification")
    print("=" * 70)
    
    all_ok = True
    
    # Check core structure
    print("\n[Core Modules] (opi/core/)")
    core_files = [
        ('opi/core/__init__.py', 'Init'),
        ('opi/core/base_state.py', 'Base state'),
        ('opi/core/fourier_solution.py', 'Fourier solution'),
        ('opi/core/precipitation.py', 'Precipitation'),
        ('opi/core/isotopes.py', 'Isotopes'),
        ('opi/core/fractionation.py', 'Fractionation'),
        ('opi/core/coordinates.py', 'Coordinates'),
        ('opi/core/catchment.py', 'Catchment'),
        ('opi/core/lifting.py', 'Lifting'),
        ('opi/core/thermodynamics.py', 'Thermodynamics'),
    ]
    for filepath, desc in core_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check models
    print("\n[Models] (opi/models/)")
    model_files = [
        ('opi/models/__init__.py', 'Init'),
        ('opi/models/one_wind.py', 'One-wind model'),
        ('opi/models/two_winds.py', 'Two-winds model'),
    ]
    for filepath, desc in model_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check solvers
    print("\n[Solvers] (opi/solvers/)")
    solver_files = [
        ('opi/solvers/__init__.py', 'Init'),
        ('opi/solvers/crs3.py', 'CRS3 optimizer'),
    ]
    for filepath, desc in solver_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check IO
    print("\n[I/O] (opi/io/)")
    io_files = [
        ('opi/io/__init__.py', 'Init'),
        ('opi/io/data_loader.py', 'Data loader'),
        ('opi/io/run_file.py', 'Run file parser'),
    ]
    for filepath, desc in io_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check utils
    print("\n[Utils] (opi/utils/)")
    util_files = [
        ('opi/utils/__init__.py', 'Init'),
        ('opi/utils/statistics.py', 'Statistics'),
        ('opi/utils/validation.py', 'Validation'),
        ('opi/utils/plotting.py', 'Plotting'),
    ]
    for filepath, desc in util_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check visualization
    print("\n[Visualization] (opi/visualization/)")
    viz_files = [
        ('opi/visualization/__init__.py', 'Init'),
        ('opi/visualization/maps.py', 'Maps'),
        ('opi/visualization/cross_sections.py', 'Cross-sections'),
        ('opi/visualization/analysis.py', 'Analysis'),
        ('opi/visualization/summary.py', 'Summary'),
    ]
    for filepath, desc in viz_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check parallel
    print("\n[Parallel] (opi/parallel/)")
    parallel_files = [
        ('opi/parallel/__init__.py', 'Init'),
        ('opi/parallel/sweep.py', 'Parameter sweep'),
        ('opi/parallel/monte_carlo.py', 'Monte Carlo'),
    ]
    for filepath, desc in parallel_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Check root files
    print("\n[Root Package Files]")
    root_files = [
        ('opi/__init__.py', 'Package init'),
        ('opi/constants.py', 'Constants'),
    ]
    for filepath, desc in root_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # Test imports
    print("\n[Testing Imports]")
    sys.path.insert(0, '.')
    
    try:
        import opi
        print("  [OK] Package 'opi' imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import 'opi': {e}")
        all_ok = False
    
    try:
        from opi.core import base_state
        print("  [OK] opi.core imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.core: {e}")
        all_ok = False
    
    try:
        from opi.models import OneWindModel
        print("  [OK] opi.models imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.models: {e}")
        all_ok = False
    
    try:
        from opi.solvers import CRS3Optimizer
        print("  [OK] opi.solvers imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.solvers: {e}")
        all_ok = False
    
    try:
        from opi.io import load_topography
        print("  [OK] opi.io imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.io: {e}")
        all_ok = False
    
    try:
        from opi.visualization import create_summary_figure
        print("  [OK] opi.visualization imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.visualization: {e}")
        all_ok = False
    
    try:
        from opi.parallel import ParameterSweep
        print("  [OK] opi.parallel imports successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import opi.parallel: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("SUCCESS: All checks passed! Package structure is correct.")
    else:
        print("FAILED: Some checks failed. Please review the errors above.")
    print("=" * 70)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
