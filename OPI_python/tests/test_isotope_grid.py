"""
Test isotope_grid.py against MATLAB isotopeGrid.m behavior
"""

import numpy as np
from opi.physics.isotope_grid import isotope_grid


def create_test_data(n_s=50, n_t=50):
    """Create test data for isotope grid calculation"""
    # Wind grid coordinates
    s = np.linspace(-50000, 50000, n_s)
    t = np.linspace(-50000, 50000, n_t)
    Sxy, Txy = np.meshgrid(s, t, indexing='ij')
    
    # Simple Gaussian topography
    h_wind = 2000 * np.exp(-(Sxy**2 + Txy**2) / (2 * 20000**2))
    
    # Uniform moisture and humidity
    f_m_wind = np.ones_like(h_wind) * 0.8
    r_h_wind = np.ones_like(h_wind) * 0.9
    f_p_wind = np.ones_like(h_wind) * 0.9
    
    # Isotherm heights above topography
    z223_wind = h_wind + 2500  # 223K isotherm
    z258_wind = h_wind + 1500  # 258K isotherm (freezing level)
    
    # Parameters
    tau_f = 500.0  # s
    U = 10.0  # m/s
    T = np.linspace(290, 220, 100)  # Temperature profile
    gamma_sat = np.ones(100) * 0.0065  # K/m
    h_s = 4000.0  # m
    h_r = 540.0  # m
    
    # Isotope parameters (fraction, not permil)
    d2h0 = -100e-3  # -100 permil
    d18o0 = -14e-3  # -14 permil
    d_d2h0_d_lat = -3e-3  # per degree
    d_d18o0_d_lat = -0.4e-3  # per degree
    
    # Latitude - column vector for broadcasting
    lat = np.linspace(44, 46, n_t).reshape(1, -1)  # Row vector that broadcasts
    lat0 = 45.0
    
    is_fit = False
    
    return {
        's': s, 't': t, 'Sxy': Sxy, 'Txy': Txy,
        'h_wind': h_wind, 'f_m_wind': f_m_wind, 'r_h_wind': r_h_wind,
        'f_p_wind': f_p_wind, 'z223_wind': z223_wind, 'z258_wind': z258_wind,
        'tau_f': tau_f, 'U': U, 'T': T, 'gamma_sat': gamma_sat,
        'h_s': h_s, 'h_r': h_r,
        'd2h0': d2h0, 'd18o0': d18o0,
        'd_d2h0_d_lat': d_d2h0_d_lat, 'd_d18o0_d_lat': d_d18o0_d_lat,
        'lat': lat, 'lat0': lat0, 'is_fit': is_fit
    }


def test_isotope_grid_basic():
    """Test basic isotope grid calculation"""
    data = create_test_data()
    
    result = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid, d18o_grid, evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid = result
    
    print(f"d2H grid shape: {d2h_grid.shape}")
    print(f"d18O grid shape: {d18o_grid.shape}")
    print(f"d2H range: {d2h_grid.min()*1000:.2f} to {d2h_grid.max()*1000:.2f} permil")
    print(f"d18O range: {d18o_grid.min()*1000:.2f} to {d18o_grid.max()*1000:.2f} permil")
    
    # Check shapes
    assert d2h_grid.shape == data['Sxy'].shape
    assert d18o_grid.shape == data['Sxy'].shape
    
    # Check that values are in reasonable range
    assert -200e-3 < d2h_grid.min() < d2h_grid.max() < 50e-3
    assert -30e-3 < d18o_grid.min() < d18o_grid.max() < 10e-3
    
    # Check deuterium excess
    d_excess = (d2h_grid * 1000 - 8 * d18o_grid * 1000)
    print(f"d-excess range: {d_excess.min():.2f} to {d_excess.max():.2f} permil")
    
    # d-excess should be in reasonable range (typically -20 to 30 permil)
    assert -50 < d_excess.min() < d_excess.max() < 50
    
    # Check evaporation grids
    assert evap_d2h_grid.shape == data['Sxy'].shape
    assert u_evap_d2h_grid.shape == data['Sxy'].shape
    
    print("[PASS] Basic isotope grid test")


def test_isotope_grid_lat_gradient():
    """Test that latitudinal gradient works correctly"""
    data = create_test_data(n_s=30, n_t=30)
    
    # Test with strong latitudinal gradient
    data['d_d2h0_d_lat'] = -10e-3  # Strong gradient
    
    result = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid, _, _, _, _, _ = result
    
    # With lat from 44 to 46 and lat0=45, we should see variation
    # Check that there's variation across the grid
    print(f"d2H variation with lat gradient: {(d2h_grid.max() - d2h_grid.min())*1000:.2f} permil")
    
    # Should have significant variation due to lat gradient
    assert (d2h_grid.max() - d2h_grid.min()) > 5e-3  # At least 5 permil variation
    
    print("[PASS] Latitudinal gradient test")


def test_isotope_grid_fit_mode():
    """Test isotope grid in fitting mode (is_fit=True)"""
    data = create_test_data()
    data['is_fit'] = True
    
    result = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid, d18o_grid, evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid = result
    
    # In fit mode, evaporation grids should be empty
    assert evap_d2h_grid.size == 0
    assert u_evap_d2h_grid.size == 0
    assert evap_d18o_grid.size == 0
    assert u_evap_d18o_grid.size == 0
    
    # But isotope grids should still be calculated
    assert d2h_grid.shape == data['Sxy'].shape
    assert d18o_grid.shape == data['Sxy'].shape
    
    print("[PASS] Fit mode test")


def test_isotope_grid_latitude_broadcast():
    """Test different latitude array shapes"""
    data = create_test_data(n_s=30, n_t=40)
    
    # Test 1: 1D lat vector (length n_t)
    data['lat'] = np.linspace(44, 46, 40)
    
    result1 = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid_1, _, _, _, _, _ = result1
    assert d2h_grid_1.shape == (30, 40)
    
    # Test 2: 2D lat array (n_s, n_t)
    data['lat'] = np.linspace(44, 46, 40).reshape(1, 40) * np.ones((30, 40))
    
    result2 = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid_2, _, _, _, _, _ = result2
    assert d2h_grid_2.shape == (30, 40)
    
    print("[PASS] Latitude broadcast test")


def test_isotope_consistency():
    """Test that d2H and d18O are consistent (GMWL-like relationship)"""
    data = create_test_data()
    
    result = isotope_grid(
        data['s'], data['t'], data['Sxy'], data['Txy'],
        data['lat'], data['lat0'], data['h_wind'], data['f_m_wind'],
        data['r_h_wind'], data['f_p_wind'], data['z223_wind'], data['z258_wind'],
        data['tau_f'], data['U'], data['T'], data['gamma_sat'],
        data['h_s'], data['h_r'], data['d2h0'], data['d18o0'],
        data['d_d2h0_d_lat'], data['d_d18o0_d_lat'], data['is_fit']
    )
    
    d2h_grid, d18o_grid, _, _, _, _ = result
    
    # Calculate slope of d2H vs d18O relationship
    d2h_flat = d2h_grid.flatten() * 1000  # Convert to permil
    d18o_flat = d18o_grid.flatten() * 1000
    
    # Fit linear relationship
    slope = np.polyfit(d18o_flat, d2h_flat, 1)[0]
    
    print(f"d2H vs d18O slope: {slope:.2f}")
    print(f"  (Global Meteoric Water Line slope is ~8)")
    
    # Slope should be reasonably close to 8 (GMWL), allow some variation
    # due to different environmental conditions
    assert 5 < slope < 12, f"Slope {slope:.2f} is outside expected range (5-12)"
    
    print("[PASS] Isotope consistency test")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing isotope_grid.py")
    print("=" * 60)
    
    test_isotope_grid_basic()
    print()
    test_isotope_grid_lat_gradient()
    print()
    test_isotope_grid_fit_mode()
    print()
    test_isotope_grid_latitude_broadcast()
    print()
    test_isotope_consistency()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
