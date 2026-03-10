"""
Command-line interface for OPI Python package

Usage:
    python -m opi <command> [options]

Commands:
    run         Run simulation from .run file
    fit-one     Fit one-wind model parameters
    fit-two     Fit two-wind model parameters
    maps-one    Generate maps for one-wind results
    maps-two    Generate maps for two-wind results
    version     Print version information

Examples:
    python -m opi run runs/run001/run001.run
    python -m opi fit-one runs/run001/run001.run
    python -m opi maps-one runs/run001/opiCalc_OneWind_Results.mat
"""

import sys
import argparse

from . import __version__, print_version


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='OPI (Orographic Precipitation and Isotopes)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simulation
  python -m opi run runs/run001/run001.run
  
  # Fit one-wind model
  python -m opi fit-one runs/run001/run001.run --max-iter 5000
  
  # Generate maps
  python -m opi maps-one runs/run001/opiCalc_OneWind_Results.mat
        """
    )
    
    parser.add_argument('--version', action='version', version=f'OPI {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run simulation from .run file')
    run_parser.add_argument('run_file', help='Path to .run configuration file')
    run_parser.add_argument('-v', '--verbose', action='store_true', default=True,
                          help='Print detailed progress (default: True)')
    run_parser.add_argument('-q', '--quiet', action='store_true',
                          help='Suppress console output')
    
    # Fit-one command
    fit_one_parser = subparsers.add_parser('fit-one', help='Fit one-wind model parameters')
    fit_one_parser.add_argument('run_file', help='Path to .run configuration file')
    fit_one_parser.add_argument('--max-iter', type=int, default=10000,
                               help='Maximum optimization iterations (default: 10000)')
    fit_one_parser.add_argument('--parallel', action='store_true',
                               help='Use parallel processing')
    
    # Fit-two command
    fit_two_parser = subparsers.add_parser('fit-two', help='Fit two-wind model parameters')
    fit_two_parser.add_argument('run_file', help='Path to .run configuration file')
    fit_two_parser.add_argument('--max-iter', type=int, default=10000,
                               help='Maximum optimization iterations (default: 10000)')
    fit_two_parser.add_argument('--parallel', action='store_true',
                               help='Use parallel processing')
    
    # Maps-one command
    maps_one_parser = subparsers.add_parser('maps-one', help='Generate maps for one-wind results')
    maps_one_parser.add_argument('results_file', help='Path to opiCalc_OneWind_Results.mat')
    maps_one_parser.add_argument('-o', '--output-dir', help='Output directory for figures')
    maps_one_parser.add_argument('--show', action='store_true',
                                help='Display plots interactively')
    
    # Maps-two command
    maps_two_parser = subparsers.add_parser('maps-two', help='Generate maps for two-wind results')
    maps_two_parser.add_argument('results_file', help='Path to opiCalc_TwoWinds_Results.mat')
    maps_two_parser.add_argument('-o', '--output-dir', help='Output directory for figures')
    maps_two_parser.add_argument('--show', action='store_true',
                                help='Display plots interactively')
    
    # Version command
    subparsers.add_parser('version', help='Print version information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'version':
        print_version()
        sys.exit(0)
    
    elif args.command == 'run':
        from .run_opi_simulation import run_simulation
        verbose = not args.quiet and args.verbose
        result = run_simulation(args.run_file, verbose=verbose)
        
        if result['success']:
            print(f"\nSimulation completed successfully!")
            print(f"Results: {result['results_path']}")
            sys.exit(0)
        else:
            print(f"\nSimulation failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif args.command == 'fit-one':
        from .opi_fit_one_wind import opi_fit_one_wind
        result = opi_fit_one_wind(
            args.run_file,
            verbose=True,
            max_iterations=args.max_iter,
            parallel=args.parallel
        )
        
        if result['success']:
            print(f"\nFitting completed successfully!")
            print(f"Final misfit: {result['misfit']:.6f}")
            print(f"Results: {result['results_path']}")
            sys.exit(0)
        else:
            print(f"\nFitting failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
    
    elif args.command == 'fit-two':
        from .opi_fit_two_winds import opi_fit_two_winds
        result = opi_fit_two_winds(
            args.run_file,
            verbose=True,
            max_iterations=args.max_iter,
            parallel=args.parallel
        )
        
        if result['success']:
            print(f"\nFitting completed successfully!")
            print(f"Final misfit: {result['misfit']:.6f}")
            print(f"Results: {result['results_path']}")
            sys.exit(0)
        else:
            print(f"\nFitting failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
    
    elif args.command == 'maps-one':
        from .opi_maps_one_wind import opi_maps_one_wind
        files = opi_maps_one_wind(
            results_file=args.results_file,
            output_dir=args.output_dir,
            save_plots=True,
            show_plots=args.show,
            verbose=True
        )
        print(f"\nGenerated {len(files)} figure(s)")
        sys.exit(0)
    
    elif args.command == 'maps-two':
        from .opi_maps_two_winds import opi_maps_two_winds
        files = opi_maps_two_winds(
            results_file=args.results_file,
            output_dir=args.output_dir,
            save_plots=True,
            show_plots=args.show,
            verbose=True
        )
        print(f"\nGenerated {len(files)} figure(s)")
        sys.exit(0)


if __name__ == '__main__':
    main()
