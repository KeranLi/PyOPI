#!/usr/bin/env python3
"""
Complete Workflow Example for OPI

This script demonstrates the complete workflow of using OPI:
1. Load or create input data
2. Run single wind field calculation
3. Run two wind fields calculation
4. Compare results
5. Save outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi import (
    opi_calc_one_wind,
    opi_calc_two_winds,
    fourier_solution,
    precipitation_grid,
    isotope_grid,
    base_state,
    deuterium_excess,
    wind_components,
    save_grids_to_numpy
)


def example_1_single_wind():
    """Example 1: Single wind field calculation with custom parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Wind Field Calculation")
    print("=" * 70)
    
    # Define custom solution vector
    # [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_d_lat, f_p0]
    solution = [12.0, 90.0, 293.0, 0.3, 5.0, 1200.0, -6.0e-3, -2.0e-3, 0.8]
    
    print("\nInput parameters:")
    print(f"  Wind speed (U): {solution[0]:.1f} m/s")
    print(f"  Azimuth: {solution[1]:.1f} degrees")
    print(f"  Temperature (T0): {solution[2]:.1f} K")
    print(f"  Mountain-height number (M): {solution[3]:.3f}")
    
    # Run calculation
    result = opi_calc_one_wind(solution_vector=solution, verbose=False)
    
    # Display results
    print("\nResults:")
    print(f"  Mean precipitation: {result['derived_params']['precip_fraction']:.6f}")
    print(f"  Mean d2H: {result['derived_params']['mean_isotope']:.2f} permil")
    
    if result['results']:
        p_grid = result['results']['precipitation']
        d2h_grid = result['results']['d2h']
        
        print(f"  Precipitation range: {p_grid.min():.6f} to {p_grid.max():.6f}")
        print(f"  d2H range: {d2h_grid.min()*1000:.1f} to {d2h_grid.max()*1000:.1f} permil")
        
        # Create simple plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        im1 = axes[0].imshow(p_grid, origin='lower', cmap='Blues')
        axes[0].set_title('Precipitation Rate')
        axes[0].set_xlabel('X (grid)')
        axes[0].set_ylabel('Y (grid)')
        plt.colorbar(im1, ax=axes[0], label='kg/m²/s')
        
        im2 = axes[1].imshow(d2h_grid * 1000, origin='lower', cmap='RdYlBu_r')
        axes[1].set_title('d²H Isotope')
        axes[1].set_xlabel('X (grid)')
        axes[1].set_ylabel('Y (grid)')
        plt.colorbar(im2, ax=axes[1], label='permil')
        
        plt.tight_layout()
        plt.savefig('example1_single_wind.png', dpi=150)
        print("\nPlot saved to: example1_single_wind.png")
        plt.close()
    
    return result


def example_2_two_winds():
    """Example 2: Two wind fields calculation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Two Wind Fields Calculation")
    print("=" * 70)
    
    # Define 19-parameter solution vector
    solution = [
        # Wind Field 1 (westerly)
        10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5.0e-3, -2.0e-3, 0.7,
        # Wind Field 2 (easterly)
        8.0, 270.0, 288.0, 0.3, 0.0, 1200.0, -7.0e-3, -1.5e-3, 0.75,
        # Mixing fraction (50% from wind 2)
        0.5
    ]
    
    print("\nWind Field 1:")
    print(f"  Speed: {solution[0]:.1f} m/s, Azimuth: {solution[1]:.1f}°")
    print("\nWind Field 2:")
    print(f"  Speed: {solution[9]:.1f} m/s, Azimuth: {solution[10]:.1f}°")
    print(f"\nMixing fraction (Wind 2): {solution[18]:.2f}")
    
    # Run calculation
    result = opi_calc_two_winds(solution_vector=solution, verbose=False)
    
    # Display results
    print("\nResults:")
    print(f"  Combined precipitation range: {result['precipitation'].min():.6f} to {result['precipitation'].max():.6f}")
    print(f"  Combined d2H range: {result['isotope'].min()*1000:.1f} to {result['isotope'].max()*1000:.1f} permil")
    print(f"  Wind 1 precip range: {result['precipitation1'].min():.6f} to {result['precipitation1'].max():.6f}")
    print(f"  Wind 2 precip range: {result['precipitation2'].min():.6f} to {result['precipitation2'].max():.6f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Wind 1
    im1 = axes[0, 0].imshow(result['precipitation1'], origin='lower', cmap='Blues', vmin=0)
    axes[0, 0].set_title('Wind Field 1: Precipitation')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Wind 2
    im2 = axes[0, 1].imshow(result['precipitation2'], origin='lower', cmap='Blues', vmin=0)
    axes[0, 1].set_title('Wind Field 2: Precipitation')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Combined
    im3 = axes[1, 0].imshow(result['precipitation'], origin='lower', cmap='Blues', vmin=0)
    axes[1, 0].set_title('Combined Precipitation')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Isotopes
    im4 = axes[1, 1].imshow(result['isotope'] * 1000, origin='lower', cmap='RdYlBu_r')
    axes[1, 1].set_title('Combined d²H')
    plt.colorbar(im4, ax=axes[1, 1], label='permil')
    
    plt.tight_layout()
    plt.savefig('example2_two_winds.png', dpi=150)
    print("\nPlot saved to: example2_two_winds.png")
    plt.close()
    
    return result


def example_3_sensitivity_analysis():
    """Example 3: Sensitivity analysis of wind speed."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Sensitivity to Wind Speed")
    print("=" * 70)
    
    wind_speeds = [5.0, 10.0, 15.0, 20.0]
    results = []
    
    print("\nTesting wind speeds:", wind_speeds)
    
    for u in wind_speeds:
        solution = [u, 90.0, 290.0, 0.25, 0.0, 1000.0, -5.0e-3, -2.0e-3, 0.7]
        result = opi_calc_one_wind(solution_vector=solution, verbose=False)
        
        if result['results']:
            mean_precip = result['results']['precipitation'].mean()
            max_precip = result['results']['precipitation'].max()
            results.append({
                'U': u,
                'mean_precip': mean_precip,
                'max_precip': max_precip
            })
            print(f"  U={u:.1f} m/s: mean={mean_precip:.6f}, max={max_precip:.6f}")
    
    # Plot sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    U_vals = [r['U'] for r in results]
    mean_vals = [r['mean_precip'] for r in results]
    max_vals = [r['max_precip'] for r in results]
    
    ax.plot(U_vals, mean_vals, 'o-', label='Mean Precipitation', linewidth=2)
    ax.plot(U_vals, max_vals, 's-', label='Max Precipitation', linewidth=2)
    ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax.set_ylabel('Precipitation Rate (kg/m²/s)', fontsize=12)
    ax.set_title('Sensitivity to Wind Speed', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example3_sensitivity.png', dpi=150)
    print("\nPlot saved to: example3_sensitivity.png")
    plt.close()


def example_4_utils():
    """Example 4: Using utility functions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Utility Functions")
    print("=" * 70)
    
    from opi.utils import (
        deuterium_excess,
        wind_components,
        rossby_number,
        froude_number,
        meteoric_water_line
    )
    
    # Deuterium excess
    d2h = -120  # permil
    d18o = -16  # permil
    dxs = deuterium_excess(d2h, d18o)
    print(f"\nDeuterium excess:")
    print(f"  d2H = {d2h}‰, d18O = {d18o}‰")
    print(f"  d-excess = {dxs:.1f}‰")
    
    # Wind components
    u, v = wind_components(10.0, 45.0)
    print(f"\nWind components (speed=10, azimuth=45°):")
    print(f"  u (east) = {u:.2f} m/s")
    print(f"  v (north) = {v:.2f} m/s")
    
    # Dimensionless numbers
    ro = rossby_number(10.0, 1e-4, 100000)
    fr = froude_number(10.0, 0.01, 2000)
    print(f"\nDimensionless numbers:")
    print(f"  Rossby number = {ro:.4f}")
    print(f"  Froude number = {fr:.4f}")
    
    # Meteoric water line
    d18o_vals = np.linspace(-20, 0, 5)
    d2h_expected = meteoric_water_line(d18o_vals, slope=8.0, intercept=10.0)
    print(f"\nMeteoric water line (slope=8, intercept=10):")
    for d18o, d2h in zip(d18o_vals, d2h_expected):
        print(f"  d18O = {d18o:.1f}‰ → d2H = {d2h:.1f}‰")


def main():
    """Run all examples."""
    print("=" * 70)
    print("OPI Complete Workflow Examples")
    print("=" * 70)
    print("\nThis script demonstrates various OPI functionalities.")
    print("Output files will be saved to the current directory.")
    
    try:
        # Run examples
        example_1_single_wind()
        example_2_two_winds()
        example_3_sensitivity_analysis()
        example_4_utils()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print("\nOutput files:")
        print("  - example1_single_wind.png")
        print("  - example2_two_winds.png")
        print("  - example3_sensitivity.png")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
