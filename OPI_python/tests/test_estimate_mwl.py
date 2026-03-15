"""
Test estimate_meteoric_water_line against MATLAB estimateMWL.m
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.physics.isotope_analysis import estimate_meteoric_water_line


class TestEstimateMWL:
    """Tests for estimate_meteoric_water_line function"""
    
    def test_basic_functionality(self):
        """Test basic MWL estimation with clean data"""
        # Create synthetic data following MWL: d2H = 10 + 8*d18O
        np.random.seed(42)
        n = 100
        d18o = np.random.uniform(-15, -5, n)  # permil
        noise = np.random.normal(0, 2, n)  # Small noise
        d2h = 10 + 8 * d18o + noise
        
        b_hat, sd_min, sd_max, cov, i_fit = estimate_meteoric_water_line(d18o, d2h)
        
        # Check that we get reasonable values
        # b_hat returns intercept in fraction (0.010 = 10 permil)
        assert b_hat[0] == pytest.approx(0.010, abs=0.001)  # Intercept ~0.010
        assert b_hat[1] == pytest.approx(8, abs=0.5)        # Slope ~8
        assert sd_min > 0
        assert sd_max > sd_min
        assert cov.shape == (2, 2)
        assert len(i_fit) == n
    
    def test_fraction_input(self):
        """Test with fraction input (not permil)"""
        np.random.seed(42)
        n = 100
        d18o = np.random.uniform(-0.015, -0.005, n)  # fraction
        noise = np.random.normal(0, 0.002, n)
        d2h = 0.010 + 8 * d18o + noise
        
        b_hat, sd_min, sd_max, cov, i_fit = estimate_meteoric_water_line(d18o, d2h)
        
        # Results should be similar (converted internally)
        assert b_hat[0] == pytest.approx(0.010, abs=0.001)
        assert b_hat[1] == pytest.approx(8, abs=0.5)
    
    def test_dexcess_filtering(self):
        """Test that samples with low dExcess are filtered out"""
        np.random.seed(42)
        n = 50
        # Normal precipitation samples (high dExcess ~10)
        # dExcess = d2H - 8*d18O, for GMWL: d2H = 10 + 8*d18O, so dExcess = 10
        d18o_normal = np.random.uniform(-15, -5, n)
        d2h_normal = 10 + 8 * d18o_normal + np.random.normal(0, 1, n)
        
        # Evaporated samples (low dExcess < 5)
        # Evaporation increases d18O and d2H but with lower slope
        # Create samples with dExcess < 5 explicitly
        d18o_evap = np.array([-8.0, -7.0, -6.0])  # permil
        # For dExcess = 3: d2H = 3 + 8*d18O
        d2h_evap = 3 + 8 * d18o_evap  # dExcess = 3 < 5
        
        d18o = np.concatenate([d18o_normal, d18o_evap])
        d2h = np.concatenate([d2h_normal, d2h_evap])
        
        b_hat, sd_min, sd_max, cov, i_fit = estimate_meteoric_water_line(d18o, d2h)
        
        # Check that evaporated samples were filtered out
        n_total = len(d18o)
        assert np.sum(i_fit) < n_total, f"Expected some samples to be filtered, but all {n_total} were included"
        # The last 3 samples (evaporated) should be filtered out
        assert not np.any(i_fit[-3:]), "Evaporated samples should be filtered out"
        # Most normal samples should be included
        assert np.sum(i_fit[:n]) > n * 0.9, "Most normal samples should be included"
    
    def test_sd_res_ratio(self):
        """Test different sd_res_ratio values"""
        np.random.seed(42)
        n = 100
        d18o = np.random.uniform(-15, -5, n)
        d2h = 10 + 8 * d18o + np.random.normal(0, 2, n)
        
        # Different ratios
        _, sd_min1, sd_max1, cov1, _ = estimate_meteoric_water_line(d18o, d2h, sd_res_ratio=2.0)
        _, sd_min2, sd_max2, cov2, _ = estimate_meteoric_water_line(d18o, d2h, sd_res_ratio=4.0)
        
        # sd_min should be the same (based on data)
        assert sd_min1 == pytest.approx(sd_min2, rel=1e-10)
        # But covariance matrices should differ
        assert not np.allclose(cov1, cov2)
    
    def test_insufficient_data(self):
        """Test error handling with insufficient data"""
        d18o = np.array([-10.0, -11.0])  # 2 points
        d2h = np.array([-70.0, -78.0])   # Both with dExcess ~10
        
        # Should raise error because n_fit < 2 after filtering
        # Actually with 2 points both having dExcess > 5, n_fit = 2
        # But we need n_fit > 2 for meaningful statistics
        # Let's use points with dExcess < 5
        d18o = np.array([-5.0, -6.0])   # Will have dExcess < 5
        d2h = np.array([-30.0, -36.0])  # dExcess = 10 for first, but let's check
        
        # dExcess = d2h - 8*d18o = -30 - 8*(-5) = -30 + 40 = 10 > 5, so included
        # Need to create points with dExcess < 5
        d18o = np.array([-5.0])  # 1 point
        d2h = np.array([-45.0])  # dExcess = -45 - 8*(-5) = -45 + 40 = -5 < 5
        
        with pytest.raises(ValueError, match="Insufficient data"):
            estimate_meteoric_water_line(d18o, d2h)
    
    def test_known_mwl(self):
        """Test with data exactly on GMWL"""
        # Global MWL: d2H = 10 + 8*d18O
        d18o = np.linspace(-20, 0, 50)  # permil
        d2h = 10 + 8 * d18o  # Exactly on line
        
        b_hat, sd_min, sd_max, cov, i_fit = estimate_meteoric_water_line(d18o, d2h)
        
        # Should recover exact values
        assert b_hat[0] == pytest.approx(0.010, abs=1e-6)  # intercept in fraction
        assert b_hat[1] == pytest.approx(8.0, abs=1e-6)    # slope
        assert sd_min == pytest.approx(0.0, abs=1e-10)     # No scatter perpendicular
    
    def test_covariance_structure(self):
        """Test that covariance matrix has correct structure"""
        np.random.seed(42)
        n = 100
        d18o = np.random.uniform(-15, -5, n)
        d2h = 10 + 8 * d18o + np.random.normal(0, 2, n)
        
        _, _, _, cov, _ = estimate_meteoric_water_line(d18o, d2h)
        
        # Check covariance matrix properties
        assert cov.shape == (2, 2)
        assert np.isfinite(cov).all()
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        # Diagonal elements should be positive (variances)
        assert cov[0, 0] > 0
        assert cov[1, 1] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
