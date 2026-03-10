#!/usr/bin/env python3
"""
Functionality comparison between MATLAB and Python versions of OPI
"""

import os
import inspect
from opi import __all__ as python_functions
from opi.constants import *
import pkgutil


def analyze_python_structure():
    """Analyze the Python package structure"""
    print("Python OPI Package Structure Analysis")
    print("=" * 50)
    
    # Get all modules in the opi package
    import opi
    package_path = opi.__path__
    package_name = opi.__name__
    
    print(f"Package: {package_name}")
    print(f"Path: {package_path}")
    
    modules = []
    for importer, modname, ispkg in pkgutil.walk_packages(path=package_path, prefix=package_name+'.', onerror=lambda x: None):
        modules.append(modname)
    
    print(f"Modules found: {len(modules)}")
    for module in sorted(modules):
        print(f"  - {module}")
    
    print()
    
    # List functions available in the package
    print("Available functions in opi package:")
    for func_name in sorted(python_functions):
        print(f"  - {func_name}")
    
    print()
    
    return modules


def compare_with_matlab():
    """Compare with MATLAB functions based on the directory structure"""
    print("MATLAB OPI Functions Analysis")
    print("=" * 35)
    
    # Analyze the MATLAB directory structure
    matlab_dir = "../OPI programs"
    
    if os.path.exists(matlab_dir):
        files = os.listdir(matlab_dir)
        
        main_functions = [f for f in files if f.startswith("opi")]
        private_functions = []
        
        # Check private directory
        private_dir = os.path.join(matlab_dir, "private")
        if os.path.exists(private_dir):
            private_files = os.listdir(private_dir)
            private_functions = [f for f in private_files if f.endswith(".m")]
        
        print(f"Main functions found: {len(main_functions)}")
        for func in sorted(main_functions):
            print(f"  - {func}")
        
        print(f"\nPrivate functions found: {len(private_functions)}")
        for func in sorted(private_functions):
            print(f"  - {func}")
        
        print()
        
        return main_functions, private_functions
    
    else:
        print(f"MATLAB directory not found: {matlab_dir}")
        return [], []


def map_equivalents():
    """Map equivalent functions between MATLAB and Python"""
    print("Function Mapping Between MATLAB and Python")
    print("=" * 42)
    
    # Known mappings based on the project structure
    mapping = {
        # Main functions
        "opiCalc_OneWind.m": "opi_calc_one_wind.py",
        "opiCalc_TwoWinds.m": "opi_calc_two_winds.py (not yet implemented)",
        "opiFit_OneWind.m": "opi_fit_one_wind.py (not yet implemented)",
        "opiFit_TwoWinds.m": "opi_fit_two_winds.py (not yet implemented)",
        "opiMaps_OneWind.m": "opi_maps_one_wind.py (not yet implemented)",
        "opiMaps_TwoWinds.m": "opi_maps_two_winds.py (not yet implemented)",
        "opiPlots_OneWind.m": "opi_plots_one_wind.py (not yet implemented)",
        "opiPlots_TwoWinds.m": "opi_plots_two_winds.py (not yet implemented)",
        "opiPredictCalc.m": "opi_predict_calc.py (not yet implemented)",
        "opiPredictPlot.m": "opi_predict_plot.py (not yet implemented)",
        
        # Core functions
        "calc_OneWind.m": "calc_one_wind.py",
        "baseState.m": "base_state.py",
        "saturatedVaporPressure.m": "saturated_vapor_pressure.py",
        "isotopeGrid.m": "isotope_grid.py",
        "precipitationGrid.m": "precipitation_grid.py",
        "windPath.m": "wind_path.py",
        "catchmentNodes.m": "catchment_nodes.py",
        "catchmentIndices.m": "catchment_indices.py",
        "gridRead.m": "grid_read.py (not yet implemented)",
        "getSolutions.m": "get_solutions.py (not yet implemented)",
        "writeSolutions.m": "write_solutions.py (not yet implemented)",
    }
    
    print("Known mappings:")
    for matlab_func, python_func in mapping.items():
        status = "✓" if "not yet implemented" not in python_func else "⚠"
        print(f"  {status} {matlab_func:<25} -> {python_func}")
    
    print()
    
    # Count implemented vs missing
    implemented = sum(1 for v in mapping.values() if "not yet implemented" not in v)
    total = len(mapping)
    
    print(f"Implementation status: {implemented}/{total} functions implemented")
    print(f"Completion: {implemented/total*100:.1f}%")
    
    return mapping


def identify_missing_features():
    """Identify missing features that need to be implemented"""
    print("\nMissing Features Analysis")
    print("=" * 25)
    
    missing_main_functions = [
        "opiCalc_TwoWinds",
        "opiFit_OneWind", 
        "opiFit_TwoWinds",
        "opiMaps_OneWind",
        "opiMaps_TwoWinds", 
        "opiPlots_OneWind",
        "opiPlots_TwoWinds",
        "opiPredictCalc",
        "opiPredictPlot"
    ]
    
    missing_core_functions = [
        "gridRead",
        "getSolutions", 
        "writeSolutions",
        "estimateMWL",
        "fminCRS3",
        "fractionationHydrogen",
        "fractionationOxygen",
        "isotherm",
        "uPrime",
        "streamline",
        "getInput",
        "getRunFile"
    ]
    
    print("Missing main functions:")
    for func in missing_main_functions:
        print(f"  - {func}")
    
    print("\nMissing core functions:")
    for func in missing_core_functions:
        print(f"  - {func}")
        
    print(f"\nTotal missing main functions: {len(missing_main_functions)}")
    print(f"Total missing core functions: {len(missing_core_functions)}")


def suggest_implementation_priority():
    """Suggest priority for implementing missing functions"""
    print("\nImplementation Priority Suggestions")
    print("=" * 35)
    
    print("High Priority (needed for full workflow):")
    high_priority = [
        "opiFit_OneWind",  # Parameter fitting is essential
        "opiCalc_TwoWinds",  # Two-wind model capability
        "opiPlots_OneWind",  # Visualization
        "opiMaps_OneWind"    # Mapping capabilities
    ]
    
    for func in high_priority:
        print(f"  1. {func}")
    
    print("\nMedium Priority:")
    medium_priority = [
        "opiFit_TwoWinds",
        "opiPredictCalc",
        "getSolutions",
        "writeSolutions"
    ]
    
    idx = 2
    for func in medium_priority:
        print(f"  {idx}. {func}")
        idx += 1
    
    print("\nLow Priority:")
    low_priority = [
        "opiPredictPlot",
        "streamline",
        "gridRead"
    ]
    
    for func in low_priority:
        print(f"  {idx}. {func}")
        idx += 1


def main():
    print("OPI MATLAB-Python Functionality Comparison")
    print("=" * 45)
    
    # Analyze Python structure
    python_modules = analyze_python_structure()
    
    # Compare with MATLAB
    matlab_main, matlab_private = compare_with_matlab()
    
    # Map equivalents
    mapping = map_equivalents()
    
    # Identify missing features
    identify_missing_features()
    
    # Suggest priorities
    suggest_implementation_priority()
    
    print("\nNext Steps:")
    print("- Focus on implementing high-priority missing functions")
    print("- Ensure all mathematical formulas match MATLAB implementation")
    print("- Add comprehensive tests comparing outputs with MATLAB version")
    print("- Implement visualization functions to match MATLAB plots")


if __name__ == "__main__":
    main()