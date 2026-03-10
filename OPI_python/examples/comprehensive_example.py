"""
Comprehensive OPI Example
=========================

This example demonstrates the full capabilities of the OPI package including:
- Model simulation
- Parameter sensitivity analysis
- Visualization
- Parallel processing
- Result saving/loading
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, '..')

from opi import (
    OneWindModel, OneWindParameters,
    load_topography, save_results,
    CRS3Optimizer
)

# Optional imports with fallbacks
try:
    from opi.visualization import (
        plot_topography_contour,
        plot_precipitation_contour,
        plot_isotope_maps,
        create_summary_figure
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Visualization module not available")

try:
    from opi.parallel import ParameterSweep
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    print("Parallel module not available")


def create_test_topography():
    """Create a test Gaussian mountain topography."""
    # 200 km x 200 km domain, 2 km resolution
    dx = 2000.0
    nx, ny = 101, 101
    x = np.arange(nx) * dx - (nx // 2) * dx
    y = np.arange(ny) * dx - (ny // 2) * dx
    
    X, Y = np.meshgrid(x, y)
    
    # Gaussian mountain: 2000 m peak, 20 km width
    h_grid = 2000.0 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    return {
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'X': X,
        'Y': Y
    }


def example_1_basic_simulation():
    """Example 1: Basic OPI simulation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic OPI Simulation")
    print("=" * 60)
    
    # Create topography
    topo = create_test_topography()
    print(f"Topography: {len(topo['x'])} x {len(topo['y'])} grid")
    print(f"Peak elevation: {topo['h_grid'].max():.0f} m")
    
    # Create model
    model = OneWindModel(topo)
    
    # Define parameters
    params = OneWindParameters(
        U=10.0,           # Wind speed (m/s)
        azimuth=90.0,     # From west
        T0=288.0,         # 15°C at sea level
        M=0.8,            # Mountain-height number
        kappa=0.0,        # No eddy diffusion
        tau_c=1000.0,     # Cloud residence time
        d2H0=-0.100,      # -100‰ base value
        d_d2H0_d_lat=0.0, # No latitudinal gradient
        f_p0=1.0          # No evaporative recycling
    )
    
    print("\nRunning simulation...")
    result = model.run(params)
    
    print("\nResults:")
    print(f"  Mean precipitation: {result.p_grid.mean():.6f} kg/m²/s")
    print(f"  Max precipitation: {result.p_grid.max():.6f} kg/m²/s")
    print(f"  Precipitation efficiency: {(result.p_grid.max() / result.p_grid.mean()):.1f}x")
    print(f"  d2H range: {result.d2H_grid.min()*1000:.1f}‰ to {result.d2H_grid.max()*1000:.1f}‰")
    print(f"  d18O range: {result.d18O_grid.min()*1000:.1f}‰ to {result.d18O_grid.max()*1000:.1f}‰")
    
    # Deuterium excess
    d_excess = (result.d2H_grid - 8 * result.d18O_grid) * 1000
    print(f"  d-excess range: {d_excess.min():.1f}‰ to {d_excess.max():.1f}‰")
    
    return model, result


def example_2_parameter_sweep(model):
    """Example 2: Parameter sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    if not HAS_PARALLEL:
        print("Parallel module not available. Skipping...")
        return
    
    base_params = OneWindParameters(
        U=10.0, azimuth=90.0, T0=288.0, M=0.8,
        kappa=0.0, tau_c=1000.0, d2H0=-0.100,
        d_d2H0_d_lat=0.0, f_p0=1.0
    )
    
    # Wind speed sweep
    wind_speeds = np.linspace(5, 20, 8)
    print(f"\nTesting {len(wind_speeds)} wind speeds: {wind_speeds}")
    
    sweep = ParameterSweep(model, n_jobs=2, verbose=1)
    results = sweep.run_one_factor('U', wind_speeds, base_params)
    
    # Analyze results
    print("\nResults:")
    for r in results:
        if r.success:
            p_mean = r.result.p_grid.mean()
            p_max = r.result.p_grid.max()
            print(f"  U={r.params.U:.1f} m/s: mean={p_mean:.6f}, max={p_max:.6f}")


def example_3_optimization():
    """Example 3: Parameter optimization with synthetic data."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Optimization")
    print("=" * 60)
    
    # Create synthetic observations
    topo = create_test_topography()
    model = OneWindModel(topo)
    
    # Generate "true" data
    true_params = OneWindParameters(
        U=12.0, azimuth=95.0, T0=290.0, M=0.75,
        kappa=0.0, tau_c=1200.0, d2H0=-0.095,
        d_d2H0_d_lat=0.0, f_p0=1.0
    )
    true_result = model.run(true_params)
    
    # Create synthetic samples
    n_samples = 20
    np.random.seed(42)
    sample_indices = np.random.choice(
        topo['h_grid'].size, n_samples, replace=False
    )
    sample_i = sample_indices // topo['h_grid'].shape[1]
    sample_j = sample_indices % topo['h_grid'].shape[1]
    
    samples = {
        'x': topo['x'][sample_j],
        'y': topo['y'][sample_i],
        'd2H': true_result.d2H_grid[sample_i, sample_j] + np.random.normal(0, 0.005, n_samples),
        'd18O': true_result.d18O_grid[sample_i, sample_j] + np.random.normal(0, 0.0005, n_samples),
        'type': ['L'] * n_samples
    }
    
    print(f"\nCreated {n_samples} synthetic samples")
    print(f"True parameters: U={true_params.U}, azimuth={true_params.azimuth}, T0={true_params.T0}")
    
    # Define objective function
    def objective(beta):
        try:
            params = OneWindParameters(
                U=beta[0], azimuth=beta[1], T0=beta[2],
                M=beta[3], kappa=beta[4], tau_c=beta[5],
                d2H0=beta[6], d_d2H0_d_lat=beta[7], f_p0=beta[8]
            )
            result = model.run(params, samples, is_fit=True)
            
            # Calculate RMSE
            mask = ~np.isnan(result.d2H_pred)
            if np.sum(mask) > 0:
                rmse = np.sqrt(np.mean(
                    (samples['d2H'][mask] - result.d2H_pred[mask])**2 +
                    (samples['d18O'][mask] - result.d18O_pred[mask])**2
                ))
                return rmse
            return 1e6
        except Exception:
            return 1e6
    
    # Set bounds
    lower_bounds = np.array([5, 0, 270, 0.1, 0, 500, -0.15, -0.01, 0.5])
    upper_bounds = np.array([20, 180, 310, 2.0, 1000, 3000, -0.05, 0.01, 1.0])
    
    # Initial guess
    x0 = np.array([10, 90, 288, 0.8, 0, 1000, -0.1, 0, 1.0])
    
    print("\nRunning CRS3 optimization...")
    optimizer = CRS3Optimizer(bounds=(lower_bounds, upper_bounds), 
                              max_iter=500, random_state=42)
    result_opt = optimizer.minimize(objective, x0)
    
    print(f"\nOptimization complete:")
    print(f"  Success: {result_opt.success}")
    print(f"  Best RMSE: {result_opt.fun:.6f}")
    print(f"  Iterations: {result_opt.n_iter}")
    print(f"  Function evaluations: {result_opt.n_eval}")
    print(f"\nOptimized parameters:")
    print(f"  U: {result_opt.x[0]:.2f} (true: {true_params.U})")
    print(f"  Azimuth: {result_opt.x[1]:.1f} (true: {true_params.azimuth})")
    print(f"  T0: {result_opt.x[2]:.1f} (true: {true_params.T0})")


def example_4_visualization(model, result):
    """Example 4: Advanced visualization."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)
    
    if not HAS_VIZ:
        print("Visualization module not available. Skipping...")
        return
    
    topo = create_test_topography()
    
    # Create summary figure
    print("\nCreating summary figure...")
    fig = create_summary_figure(topo, result)
    fig.savefig('opi_summary.png', dpi=150, bbox_inches='tight')
    print("  Saved: opi_summary.png")
    
    # Create cross-section
    from opi.visualization import plot_cross_section
    
    print("\nCreating cross-section...")
    start_point = (-40000, 0)  # 40 km west
    end_point = (40000, 0)     # 40 km east
    
    fig = plot_cross_section(
        topo['x'], topo['y'], topo['h_grid'],
        result.p_grid, result.d2H_grid,
        start_point, end_point,
        title='Cross-section (East-West)'
    )
    fig.savefig('opi_cross_section.png', dpi=150, bbox_inches='tight')
    print("  Saved: opi_cross_section.png")
    
    plt.show()


def example_5_io_operations(result):
    """Example 5: Save and load results."""
    print("\n" + "=" * 60)
    print("Example 5: I/O Operations")
    print("=" * 60)
    
    # Save results
    print("\nSaving results...")
    save_results(result, 'opi_results.npz', format='npz')
    print("  Saved: opi_results.npz")
    
    # Try NetCDF if available
    try:
        save_results(result, 'opi_results.nc', format='nc')
        print("  Saved: opi_results.nc")
    except ImportError:
        print("  NetCDF support not available (pip install xarray)")
    
    # Load results
    print("\nLoading results...")
    loaded = np.load('opi_results.npz', allow_pickle=True)
    print(f"  Loaded {len(loaded.files)} variables")
    print(f"  Available: {', '.join(loaded.files[:5])}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OPI Comprehensive Example")
    print("=" * 60)
    print("\nThis example demonstrates the full OPI package capabilities.")
    
    # Example 1: Basic simulation
    model, result = example_1_basic_simulation()
    
    # Example 2: Parameter sweep
    example_2_parameter_sweep(model)
    
    # Example 3: Optimization
    example_3_optimization()
    
    # Example 4: Visualization
    example_4_visualization(model, result)
    
    # Example 5: I/O
    example_5_io_operations(result)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
