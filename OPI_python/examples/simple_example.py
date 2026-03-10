"""
Simple OPI Example
==================

This example demonstrates the basic usage of the OPI package
for a single-wind simulation over a Gaussian mountain.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import OPI package
import sys
sys.path.insert(0, '..')
from opi import (
    OneWindModel, OneWindParameters,
    plot_topography, plot_precipitation, plot_isotopes
)


def create_gaussian_mountain(
    nx: int = 100,
    ny: int = 100,
    dx: float = 1000.0,  # 1 km resolution
    mountain_height: float = 2000.0,  # 2 km peak
    half_width: float = 20000.0  # 20 km half-width
) -> dict:
    """Create a Gaussian mountain topography."""
    # Create coordinate vectors
    x = np.arange(nx) * dx - (nx // 2) * dx
    y = np.arange(ny) * dx - (ny // 2) * dx
    
    # Create grid
    X, Y = np.meshgrid(x, y)
    
    # Gaussian elevation
    h_grid = mountain_height * np.exp(-(X**2 + Y**2) / (2 * half_width**2))
    
    return {
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'lat': np.zeros_like(h_grid),
        'lon': np.zeros_like(h_grid)
    }


def main():
    """Run a simple OPI simulation."""
    print("=" * 60)
    print("OPI Simple Example: Gaussian Mountain")
    print("=" * 60)
    
    # 1. Create topography
    print("\n1. Creating topography...")
    topography = create_gaussian_mountain()
    print(f"   Grid size: {len(topography['x'])} x {len(topography['y'])}")
    print(f"   Max elevation: {topography['h_grid'].max():.0f} m")
    
    # 2. Create model
    print("\n2. Creating OPI model...")
    constants = {
        'f_c': 1e-4,  # Coriolis parameter (~45° latitude)
        'h_r': 540.0   # Isotope exchange distance
    }
    model = OneWindModel(topography, constants)
    
    # 3. Define parameters
    print("\n3. Setting model parameters...")
    params = OneWindParameters(
        U=10.0,           # Wind speed: 10 m/s
        azimuth=90.0,     # From west (90°)
        T0=288.0,         # Sea-level temperature: 15°C
        M=0.8,            # Mountain-height number
        kappa=0.0,        # No eddy diffusion
        tau_c=1000.0,     # Cloud water residence time: 1000 s
        d2H0=-0.100,      # Base d2H: -100‰ (as fraction)
        d_d2H0_d_lat=0.0, # No latitudinal gradient
        f_p0=1.0          # No evaporative recycling
    )
    print(f"   Wind speed: {params.U} m/s")
    print(f"   Wind direction: {params.azimuth}°")
    print(f"   Temperature: {params.T0} K ({params.T0 - 273.15:.1f}°C)")
    
    # 4. Run simulation
    print("\n4. Running simulation...")
    result = model.run(params)
    print("   Simulation complete!")
    
    # 5. Display results
    print("\n5. Results:")
    print(f"   Mean precipitation: {result.p_grid.mean():.6f} kg/m²/s")
    print(f"   Max precipitation: {result.p_grid.max():.6f} kg/m²/s")
    print(f"   Precipitation range: {result.p_grid.min():.6f} to {result.p_grid.max():.6f}")
    print(f"   d2H range: {result.d2H_grid.min()*1000:.1f}‰ to {result.d2H_grid.max()*1000:.1f}‰")
    print(f"   d18O range: {result.d18O_grid.min()*1000:.1f}‰ to {result.d18O_grid.max()*1000:.1f}‰")
    
    # 6. Plot results
    print("\n6. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Topography
    ax = axes[0, 0]
    plot_topography(topography['x'], topography['y'], 
                    topography['h_grid'], ax=ax)
    ax.set_title('Topography (m)')
    
    # Precipitation
    ax = axes[0, 1]
    plot_precipitation(topography['x'], topography['y'],
                       result.p_grid, ax=ax)
    ax.set_title('Precipitation Rate (kg/m²/s)')
    
    # d2H
    ax = axes[1, 0]
    plot_isotopes(topography['x'], topography['y'],
                  result.d2H_grid * 1000, ax=ax)  # Convert to per mil
    ax.set_title('δ²H (‰)')
    
    # d18O
    ax = axes[1, 1]
    plot_isotopes(topography['x'], topography['y'],
                  result.d18O_grid * 1000, ax=ax)
    ax.set_title('δ¹⁸O (‰)')
    
    plt.tight_layout()
    plt.savefig('opi_simple_example.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'opi_simple_example.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
