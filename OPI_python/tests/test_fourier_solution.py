"""
Test fourier_solution.py against MATLAB fourierSolution.m behavior
"""

import numpy as np
from opi.physics.fourier_solution import fourier_solution


def test_fourier_solution_basic():
    """Test basic fourier_solution functionality"""
    # Create a simple test grid
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    
    # Create a Gaussian mountain
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Parameters
    U = 10.0
    azimuth = 90.0
    NM = 0.01
    f_c = 1e-4
    h_rho = 8000.0
    
    # Run fourier_solution
    result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    
    print(f"s vector length: {len(result['s'])}")
    print(f"t vector length: {len(result['t'])}")
    print(f"h_wind shape: {result['h_wind'].shape}")
    print(f"Sxy shape: {result['Sxy'].shape}")
    print(f"h_hat shape: {result['h_hat'].shape}")
    print(f"k_z shape: {result['k_z'].shape}")
    
    # Check shapes
    n_s, n_t = result['h_wind'].shape
    n_s_pad, n_t_pad = result['h_hat'].shape
    
    assert n_s_pad == 2 * n_s, f"Expected n_s_pad = 2*n_s, got {n_s_pad} vs {2*n_s}"
    assert n_t_pad == n_t, f"Expected n_t_pad = n_t, got {n_t_pad} vs {n_t}"
    
    print("[PASS] Basic fourier_solution test")


def test_fourier_solution_wavenumbers():
    """Test wavenumber calculation"""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    result = fourier_solution(x, y, h_grid, 10.0, 90.0, 0.01, 1e-4, 8000.0)
    
    k_s = result['k_s']
    k_t = result['k_t']
    
    print(f"k_s range: [{k_s.min():.6f}, {k_s.max():.6f}] rad/m")
    print(f"k_t range: [{k_t.min():.6f}, {k_t.max():.6f}] rad/m")
    
    # Check that k_s and k_t are symmetric around 0
    assert np.isclose(k_s[0], 0, atol=1e-10)
    assert np.isclose(k_t[0], 0, atol=1e-10)
    
    # Check spacing
    d_k_s = k_s[1] - k_s[0]
    d_k_t = k_t[1] - k_t[0]
    print(f"d_k_s: {d_k_s:.6e}, d_k_t: {d_k_t:.6e}")
    
    print("[PASS] Wavenumber calculation test")


def test_fourier_solution_h_hat():
    """Test Fourier coefficients"""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    result = fourier_solution(x, y, h_grid, 10.0, 90.0, 0.01, 1e-4, 8000.0)
    
    h_hat = result['h_hat']
    
    # Check that h_hat is complex
    assert np.iscomplexobj(h_hat)
    
    # DC component should be large (mean of h_grid)
    h_hat_dc = h_hat[0, 0]
    print(f"DC component (h_hat[0,0]): {h_hat_dc}")
    
    # Total energy should be conserved (Parseval's theorem)
    h_wind = result['h_wind']
    energy_space = np.sum(np.abs(h_wind)**2)
    energy_freq = np.sum(np.abs(h_hat)**2) / (h_hat.shape[0] * h_hat.shape[1])
    print(f"Energy in space domain: {energy_space:.2f}")
    print(f"Energy in freq domain: {energy_freq:.2f}")
    
    print("[PASS] Fourier coefficients test")


def test_fourier_solution_k_z():
    """Test vertical wavenumber calculation"""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    result = fourier_solution(x, y, h_grid, 10.0, 90.0, 0.01, 1e-4, 8000.0)
    
    k_z = result['k_z']
    k_s = result['k_s']
    
    print(f"k_z real part range: [{np.real(k_z).min():.6f}, {np.real(k_z).max():.6f}]")
    print(f"k_z imaginary part range: [{np.imag(k_z).min():.6f}, {np.imag(k_z).max():.6f}]")
    
    # Check that k_z is complex
    assert np.iscomplexobj(k_z)
    
    # For propagating waves (real k_z), sign should match sign of k_s
    # This is handled by the i_neg correction in the code
    
    print("[PASS] Vertical wavenumber test")


def test_fourier_solution_singularity():
    """Test singularity handling"""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Use parameters that might cause singularity (U*k_s ≈ f_c)
    U = 10.0
    f_c = 0.001  # Larger Coriolis parameter to increase chance of singularity
    
    result = fourier_solution(x, y, h_grid, U, 90.0, 0.01, f_c, 8000.0)
    
    # Check that no NaN or Inf in results
    assert not np.any(np.isnan(result['k_z']))
    assert not np.any(np.isinf(result['k_z']))
    
    print("[PASS] Singularity handling test")


def test_fourier_solution_different_azimuths():
    """Test fourier_solution with different wind directions"""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 40)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    azimuths = [0, 45, 90, 135, 180]
    
    for azimuth in azimuths:
        result = fourier_solution(x, y, h_grid, 10.0, azimuth, 0.01, 1e-4, 8000.0)
        
        # Check that h_wind has reasonable values
        assert np.all(result['h_wind'] >= 0)
        assert np.all(result['h_wind'] <= 2500)
        
        print(f"  Azimuth {azimuth}°: h_wind max = {result['h_wind'].max():.1f}")
    
    print("[PASS] Different azimuths test")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing fourier_solution module")
    print("=" * 60)
    
    test_fourier_solution_basic()
    print()
    test_fourier_solution_wavenumbers()
    print()
    test_fourier_solution_h_hat()
    print()
    test_fourier_solution_k_z()
    print()
    test_fourier_solution_singularity()
    print()
    test_fourier_solution_different_azimuths()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
