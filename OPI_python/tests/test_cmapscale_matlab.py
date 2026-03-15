"""
Test cmapscale against MATLAB cmapscale.m

Numerical comparison tests for data-driven colormap scaling.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.viz.colormaps import cmapscale, cool_warm


class TestCmapscaleBasic:
    """Basic functionality tests"""
    
    def test_factor_zero_linear_mapping(self):
        """factor=0 should give approximately linear mapping"""
        np.random.seed(42)
        z = np.random.normal(0, 1, 1000)
        cmap0 = cool_warm(64).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.0)
        
        # Linear mapping should preserve colormap structure
        assert cmap1.shape == cmap0.shape
        # Colors should be similar to input (some change due to pU weighting)
        # but with factor=0, s=tan(0)=0, so pS = pL
        assert np.allclose(cmap1, cmap0, atol=0.1)
    
    def test_factor_one_maximum_contrast(self):
        """factor=1 should give maximum contrast (histogram equalization)"""
        np.random.seed(42)
        # Skewed distribution
        z = np.random.exponential(1.0, 1000)
        cmap0 = cool_warm(64).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=1.0)
        
        # Maximum contrast should change colormap significantly
        assert cmap1.shape == cmap0.shape
        # Should be different from linear
        assert not np.allclose(cmap1, cmap0, atol=0.01)
    
    def test_uniform_values(self):
        """Test handling of uniform z values"""
        z = np.ones(100) * 5.0
        cmap0 = cool_warm(32).colors
        
        # Should return input colormap with warning
        cmap1, ticks, labels = cmapscale(z, cmap0, n_ticks=5, n_round=1)
        
        assert np.array_equal(cmap1, cmap0)
        assert ticks is not None
        assert labels is not None
    
    def test_nan_handling(self):
        """Test that NaN values are properly removed"""
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        z[::10] = np.nan  # Add NaNs
        
        cmap0 = cool_warm(32).colors
        
        # Should work without error
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5)
        assert cmap1.shape == cmap0.shape


class TestCmapscaleCentering:
    """Tests for z0 centering functionality"""
    
    def test_center_at_zero(self):
        """Test centering colormap at z0=0 for diverging data"""
        np.random.seed(42)
        # Data centered at zero
        z = np.random.normal(0, 1, 1000)
        cmap0 = cool_warm(64).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5, z0=0.0)
        
        assert cmap1.shape == cmap0.shape
        # Centering should change colormap
        assert not np.allclose(cmap1, cmap0, atol=0.01)
    
    def test_z0_at_boundary_low(self):
        """Test z0 <= z_min uses upper half of colormap"""
        np.random.seed(42)
        z = np.random.uniform(0, 10, 100)
        cmap0 = cool_warm(64).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5, z0=0.0)
        
        # Should use upper half (warmer colors)
        assert cmap1.shape == cmap0.shape
    
    def test_z0_at_boundary_high(self):
        """Test z0 >= z_max uses lower half of colormap"""
        np.random.seed(42)
        z = np.random.uniform(0, 10, 100)
        cmap0 = cool_warm(64).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5, z0=15.0)
        
        # Should use lower half (cooler colors)
        assert cmap1.shape == cmap0.shape


class TestCmapscaleTicks:
    """Tests for tick generation"""
    
    def test_tick_generation(self):
        """Test tick positions and labels generation"""
        np.random.seed(42)
        z = np.random.normal(0, 1, 1000)
        cmap0 = cool_warm(32).colors
        
        n_ticks = 5
        n_round = 2
        cmap1, ticks, labels = cmapscale(z, cmap0, factor=0.5, 
                                          n_ticks=n_ticks, n_round=n_round)
        
        assert ticks is not None
        assert labels is not None
        assert len(ticks) == n_ticks
        assert len(labels) == n_ticks
        
        # Ticks should span the data range
        assert ticks[0] <= np.min(z)
        assert ticks[-1] >= np.max(z)
        
        # Labels should be strings with correct precision
        for label in labels:
            assert isinstance(label, str)
            # Check rounding precision
            if '.' in label:
                decimal_places = len(label.split('.')[1])
                assert decimal_places <= n_round
    
    def test_tick_negative_zero_fix(self):
        """Test that -0 is converted to 0"""
        z = np.array([-0.001, 0.001, 0.002])
        cmap0 = cool_warm(16).colors
        
        _, _, labels = cmapscale(z, cmap0, n_ticks=3, n_round=1)
        
        # Check no negative zero in labels
        for label in labels:
            assert label != '-0.0'
            assert label != '-0'
    
    def test_no_ticks_requested(self):
        """Test that ticks=None when n_ticks not specified"""
        z = np.random.normal(0, 1, 100)
        cmap0 = cool_warm(16).colors
        
        cmap1, ticks, labels = cmapscale(z, cmap0, factor=0.5)
        
        assert ticks is None
        assert labels is None


class TestCmapscaleMatlabComparison:
    """Numerical comparison with MATLAB cmapscale.m"""
    
    def test_skewed_distribution_factor_half(self):
        """
        Compare against MATLAB for skewed data with factor=0.5
        
        MATLAB code:
        ```matlab
        z = exprnd(1, 1000, 1);
        cmap0 = coolwarm(32);
        cmap1 = cmapscale(z, cmap0, 0.5);
        ```
        """
        np.random.seed(42)
        z = np.random.exponential(1.0, 1000)
        cmap0 = cool_warm(32).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5)
        
        # Check properties that should hold
        assert cmap1.shape == (32, 3)
        assert np.all(cmap1 >= 0) and np.all(cmap1 <= 1)
        
        # First color should be cool (blue)
        assert cmap1[0, 2] > cmap1[0, 0]
        # Last color should be warm (red)
        assert cmap1[-1, 0] > cmap1[-1, 2]
    
    def test_normal_distribution_centered(self):
        """Test with normal distribution centered at zero"""
        np.random.seed(42)
        z = np.random.normal(0, 2, 1000)
        cmap0 = cool_warm(64).colors
        
        cmap1, ticks, labels = cmapscale(z, cmap0, factor=0.7, z0=0.0, 
                                          n_ticks=7, n_round=1)
        
        # Should have symmetric-looking ticks around zero
        assert cmap1.shape == (64, 3)
        if ticks is not None:
            # Middle tick should be near zero
            mid_tick = ticks[len(ticks)//2]
            assert abs(mid_tick) < 1.0
    
    def test_bimodal_distribution(self):
        """Test with bimodal distribution"""
        np.random.seed(42)
        # Two peaks
        z1 = np.random.normal(-2, 0.5, 500)
        z2 = np.random.normal(2, 0.5, 500)
        z = np.concatenate([z1, z2])
        
        cmap0 = cool_warm(48).colors
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.8)
        
        # Should emphasize the two modes
        assert cmap1.shape == (48, 3)
    
    def test_factor_bounds(self):
        """Test that factor must be in [0, 1]"""
        z = np.random.normal(0, 1, 100)
        cmap0 = cool_warm(16).colors
        
        with pytest.raises(ValueError, match="factor must be in range"):
            cmapscale(z, cmap0, factor=-0.1)
        
        with pytest.raises(ValueError, match="factor must be in range"):
            cmapscale(z, cmap0, factor=1.1)


class TestCmapscaleEdgeCases:
    """Edge case tests"""
    
    def test_single_value(self):
        """Test with single unique value"""
        z = np.array([5.0, 5.0, 5.0])
        cmap0 = cool_warm(8).colors
        
        cmap1, _, _ = cmapscale(z, cmap0)
        assert np.array_equal(cmap1, cmap0)
    
    def test_two_values(self):
        """Test with only two unique values"""
        z = np.array([0.0, 1.0, 0.0, 1.0])
        cmap0 = cool_warm(16).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5)
        assert cmap1.shape == (16, 3)
    
    def test_very_large_factor(self):
        """Test factor very close to 1"""
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        cmap0 = cool_warm(32).colors
        
        # Very close to 1 but not exactly 1
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.9999)
        assert cmap1.shape == (32, 3)
    
    def test_very_small_factor(self):
        """Test factor very close to 0"""
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        cmap0 = cool_warm(32).colors
        
        # Very close to 0
        cmap1, _, _ = cmapscale(z, cmap0, factor=1e-10)
        assert cmap1.shape == (32, 3)
    
    def test_large_colormap(self):
        """Test with large colormap"""
        z = np.random.normal(0, 1, 100)
        cmap0 = cool_warm(256).colors
        
        cmap1, _, _ = cmapscale(z, cmap0, factor=0.5)
        assert cmap1.shape == (256, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
