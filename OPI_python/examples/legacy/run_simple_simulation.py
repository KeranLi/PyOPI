"""
Simple OPI Simulation

Run a quick OPI simulation with customizable parameters.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from opi import opi_calc_one_wind


def run_simulation(
    U=10.0,           # Wind speed (m/s)
    azimuth=45.0,     # Wind direction (degrees from North)
    T0=290.0,         # Sea-level temperature (K)
    M=0.3,            # Mountain-height number
    kappa=10000.0,    # Eddy diffusion (m^2/s)
    tau_c=1000.0,     # Condensation time (s)
    d2H0=-0.003,      # Base d2H value (fraction)
    d_d2H0_dlat=-0.001,  # Latitudinal gradient
    fP0=1.0,          # Residual precipitation fraction
    grid_size=100,    # Grid size (nx = ny)
    mountain_height=2000.0,  # Peak mountain height (m)
    verbose=True
):
    """
    Run a simple OPI one-wind simulation.
    
    Parameters
    ----------
    U : float
        Wind speed (m/s)
    azimuth : float
        Wind direction (degrees from North, 0=North, 90=East)
    T0 : float
        Sea-level temperature (K)
    M : float
        Mountain-height number (dimensionless, <1 for linear regime)
    kappa : float
        Horizontal eddy diffusion (m^2/s)
    tau_c : float
        Cloud water condensation time (s)
    d2H0 : float
        Base hydrogen isotope value (fraction, e.g., -0.003 = -30 permil)
    d_d2H0_dlat : float
        Latitudinal gradient of d2H (fraction per degree)
    fP0 : float
        Residual precipitation after evaporation (1.0 = no evaporation)
    grid_size : int
        Number of grid cells in each direction
    mountain_height : float
        Peak mountain elevation (m)
    verbose : bool
        Print progress information
    
    Returns
    -------
    dict
        Simulation results
    """
    
    if verbose:
        print("="*60)
        print("OPI Simple Simulation")
        print("="*60)
        print(f"\nParameters:")
        print(f"  Wind speed: {U} m/s")
        print(f"  Wind direction: {azimuth} degrees")
        print(f"  Temperature: {T0} K")
        print(f"  Mountain-height number: {M}")
        print(f"  Eddy diffusion: {kappa} m^2/s")
        print(f"  Condensation time: {tau_c} s")
        print(f"  Base d2H: {d2H0*1000:.1f} permil")
        print(f"  Grid size: {grid_size}x{grid_size}")
    
    # Build solution vector
    solution_vector = [U, azimuth, T0, M, kappa, tau_c, d2H0, d_d2H0_dlat, fP0]
    
    # Run simulation
    result = opi_calc_one_wind(
        run_file_path=None,
        solution_vector=solution_vector,
        verbose=verbose
    )
    
    if verbose and result.get('results'):
        print("\n" + "-"*60)
        print("Results:")
        print("-"*60)
        p = result['results']['precipitation']
        d2h = result['results']['d2h']
        d18o = result['results']['d18o']
        
        print(f"  Precipitation: {p.min():.6f} to {p.max():.6f} kg/m^2/s")
        print(f"  d2H: {d2h.min()*1000:.1f} to {d2h.max()*1000:.1f} permil")
        print(f"  d18O: {d18o.min()*1000:.1f} to {d18o.max()*1000:.1f} permil")
        
        # Calculate d-excess
        d_excess = (d2h * 1000) - 8 * (d18o * 1000)
        print(f"  d-excess: {d_excess.min():.1f} to {d_excess.max():.1f} permil")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple OPI Simulation')
    parser.add_argument('--U', type=float, default=10.0, help='Wind speed (m/s)')
    parser.add_argument('--azimuth', type=float, default=45.0, help='Wind direction (deg)')
    parser.add_argument('--T0', type=float, default=290.0, help='Temperature (K)')
    parser.add_argument('--M', type=float, default=0.3, help='Mountain-height number')
    parser.add_argument('--kappa', type=float, default=10000.0, help='Eddy diffusion (m^2/s)')
    parser.add_argument('--tau-c', type=float, default=1000.0, help='Condensation time (s)')
    parser.add_argument('--grid-size', type=int, default=100, help='Grid size')
    
    args = parser.parse_args()
    
    result = run_simulation(
        U=args.U,
        azimuth=args.azimuth,
        T0=args.T0,
        M=args.M,
        kappa=args.kappa,
        tau_c=args.tau_c,
        grid_size=args.grid_size
    )
    
    print("\nSimulation completed!")
