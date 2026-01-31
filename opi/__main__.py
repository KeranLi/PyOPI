"""
OPI Command Line Interface

Usage:
    python -m opi calc-one-wind [runfile]
    python -m opi calc-two-winds [runfile]
    python -m opi fit-one-wind [runfile]
    python -m opi fit-two-winds [runfile]
    python -m opi test
    python -m opi info
"""

import sys
import argparse
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='OPI (Orographic Precipitation and Isotopes) Command Line Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m opi calc-one-wind                    # Run with defaults
  python -m opi calc-one-wind path/to/run.run    # Run with run file
  python -m opi fit-one-wind path/to/run.run     # Parameter fitting
  python -m opi test                             # Run tests
  python -m opi info                             # Show package info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # calc-one-wind command
    calc_one = subparsers.add_parser('calc-one-wind', help='Single wind field calculation')
    calc_one.add_argument('runfile', nargs='?', help='Path to run file')
    calc_one.add_argument('-o', '--output', help='Output file path')
    calc_one.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
    
    # calc-two-winds command
    calc_two = subparsers.add_parser('calc-two-winds', help='Two wind fields calculation')
    calc_two.add_argument('runfile', nargs='?', help='Path to run file')
    calc_two.add_argument('-o', '--output', help='Output file path')
    calc_two.add_argument('-v', '--verbose', action='store_true', default=True)
    
    # fit-one-wind command
    fit_one = subparsers.add_parser('fit-one-wind', help='Single wind field parameter fitting')
    fit_one.add_argument('runfile', nargs='?', help='Path to run file')
    fit_one.add_argument('-i', '--iter', type=int, default=1000, help='Max iterations')
    fit_one.add_argument('-o', '--output', help='Output file path')
    
    # fit-two-winds command
    fit_two = subparsers.add_parser('fit-two-winds', help='Two wind fields parameter fitting')
    fit_two.add_argument('runfile', nargs='?', help='Path to run file')
    fit_two.add_argument('-i', '--iter', type=int, default=1000, help='Max iterations')
    fit_two.add_argument('-o', '--output', help='Output file path')
    
    # test command
    test_cmd = subparsers.add_parser('test', help='Run tests')
    test_cmd.add_argument('-v', '--verbose', action='store_true', help='Verbose test output')
    
    # info command
    info_cmd = subparsers.add_parser('info', help='Show package information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'calc-one-wind':
        cmd_calc_one_wind(args)
    elif args.command == 'calc-two-winds':
        cmd_calc_two_winds(args)
    elif args.command == 'fit-one-wind':
        cmd_fit_one_wind(args)
    elif args.command == 'fit-two-winds':
        cmd_fit_two_winds(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'info':
        cmd_info()


def cmd_calc_one_wind(args):
    """Run single wind field calculation."""
    from .app.calc_one_wind import opi_calc_one_wind
    
    print("=" * 60)
    print("OPI Single Wind Field Calculation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = opi_calc_one_wind(
        run_file_path=args.runfile,
        verbose=args.verbose
    )
    
    if result['results']:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print("Solution Parameters:")
        for name, val in result['solution_params'].items():
            print(f"  {name}: {val}")
        
        print("\nDerived Parameters:")
        for name, val in result['derived_params'].items():
            if isinstance(val, float):
                print(f"  {name}: {val:.6f}")
            else:
                print(f"  {name}: {val}")
        
        # Save if output specified
        if args.output:
            save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nCalculation failed or no results produced.")


def cmd_calc_two_winds(args):
    """Run two wind fields calculation."""
    from .app.calc_two_winds import opi_calc_two_winds
    
    print("=" * 60)
    print("OPI Two Wind Fields Calculation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = opi_calc_two_winds(
        run_file_path=args.runfile,
        verbose=args.verbose
    )
    
    if result.get('precipitation') is not None:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print("\nWind Field 1:")
        print(f"  U: {result['solution_params']['wind1_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind1_az']:.1f} deg")
        
        print("\nWind Field 2:")
        print(f"  U: {result['solution_params']['wind2_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind2_az']:.1f} deg")
        print(f"  Fraction: {result['solution_params']['frac2']:.2f}")
        
        print("\nDerived Parameters:")
        for name, val in result['derived_params'].items():
            if isinstance(val, float):
                print(f"  {name}: {val:.6f}")
        
        if args.output:
            save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nCalculation failed.")


def cmd_fit_one_wind(args):
    """Run single wind field parameter fitting."""
    from .app.fitting import opi_fit_one_wind
    
    print("=" * 60)
    print("OPI Single Wind Field Parameter Fitting")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = opi_fit_one_wind(
        run_file_path=args.runfile,
        verbose=True,
        max_iterations=args.iter
    )
    
    print("\n" + "=" * 60)
    print("FIT RESULTS")
    print("=" * 60)
    print(f"Convergence: {result['convergence']}")
    print(f"Final misfit: {result['misfit']:.6f}")
    print(f"Iterations: {result['iterations']}")
    
    if result['solution_params']:
        print("\nFitted Parameters:")
        for name, val in result['solution_params'].items():
            print(f"  {name}: {val:.6f}")
        
        if args.output:
            save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")


def cmd_fit_two_winds(args):
    """Run two wind fields parameter fitting."""
    from .app.fitting import opi_fit_two_winds
    
    print("=" * 60)
    print("OPI Two Wind Fields Parameter Fitting")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = opi_fit_two_winds(
        run_file_path=args.runfile,
        verbose=True,
        max_iterations=args.iter
    )
    
    print("\n" + "=" * 60)
    print("FIT RESULTS")
    print("=" * 60)
    print(f"Convergence: {result['convergence']}")
    print(f"Final misfit: {result['misfit']:.6f}")
    print(f"Iterations: {result['iterations']}")
    
    if result['solution_params']:
        print("\nWind Field 1:")
        print(f"  U: {result['solution_params']['wind1_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind1_az']:.1f} deg")
        
        print("\nWind Field 2:")
        print(f"  U: {result['solution_params']['wind2_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind2_az']:.1f} deg")
        print(f"  Fraction: {result['solution_params']['frac2']:.2f}")
        
        if args.output:
            save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")


def cmd_test(args):
    """Run tests."""
    print("=" * 60)
    print("OPI Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import all modules
    print("\n[Test 1] Import all modules...")
    try:
        from opi import (
            fourier_solution, precipitation_grid, isotope_grid,
            fractionation_hydrogen, fractionation_oxygen,
            opi_calc_one_wind, opi_calc_two_winds,
            opi_fit_one_wind, opi_fit_two_winds
        )
        print("  PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Single wind calculation
    print("\n[Test 2] Single wind calculation...")
    try:
        from opi import opi_calc_one_wind
        result = opi_calc_one_wind(verbose=False)
        assert result['results'] is not None
        print("  PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Two wind calculation
    print("\n[Test 3] Two wind calculation...")
    try:
        from opi import opi_calc_two_winds
        result = opi_calc_two_winds(verbose=False)
        assert result.get('precipitation') is not None
        print("  PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Test 4: CRS3 optimization
    print("\n[Test 4] CRS3 optimization...")
    try:
        from opi import fmin_crs3
        
        def objective(x):
            return sum((xi - 1.0)**2 for xi in x)
        
        result = fmin_crs3(objective, [(-2, 2), (-2, 2)], max_iter=100)
        assert result.success
        print("  PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)


def cmd_info():
    """Show package information."""
    print("=" * 60)
    print("OPI (Orographic Precipitation and Isotopes)")
    print("=" * 60)
    print("\nPackage Information:")
    print("  Version: 1.0.0")
    print("  Python Port: AI Assistant")
    print("  Original Author: Mark Brandon (Yale University)")
    
    print("\nImplemented Features:")
    print("  [Phase 1] Core Physics (100%):")
    print("    - FFT terrain solution")
    print("    - LTOP precipitation")
    print("    - Isotope fractionation")
    print("    - Isotope grid calculation")
    
    print("\n  [Phase 2] Application Layer (100%):")
    print("    - Single wind field calculation")
    print("    - Two wind fields calculation")
    print("    - Parameter fitting (CRS3)")
    print("    - Data loading (MAT/Excel)")
    
    print("\nAvailable Commands:")
    print("  python -m opi calc-one-wind [runfile]")
    print("  python -m opi calc-two-winds [runfile]")
    print("  python -m opi fit-one-wind [runfile]")
    print("  python -m opi fit-two-winds [runfile]")
    print("  python -m opi test")
    print("  python -m opi info")
    
    print("\nFor more information:")
    print("  See README.md and COMPLETION_REPORT.md")
    print("=" * 60)


def save_results(result, filepath):
    """Save results to file."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    result_serializable = convert(result)
    
    with open(filepath, 'w') as f:
        json.dump(result_serializable, f, indent=2)


if __name__ == '__main__':
    main()
