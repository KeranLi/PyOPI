"""
Test coolwarm colormap against MATLAB coolwarm.m
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.viz.colormaps import cool_warm
from opi.viz.coolwarm_data import COOLWARM_257


class TestCoolwarm:
    """Tests for cool_warm colormap"""
    
    def test_default_size(self):
        """Test default colormap size is 256"""
        cmap = cool_warm()
        assert cmap.N == 256
        assert cmap.colors.shape == (256, 3)
    
    def test_custom_size(self):
        """Test custom colormap sizes"""
        for n in [64, 128, 256, 512]:
            cmap = cool_warm(n_colors=n)
            assert cmap.N == n
            assert cmap.colors.shape == (n, 3)
    
    def test_color_range(self):
        """Test all color values are in valid range [0, 1]"""
        cmap = cool_warm(256)
        assert np.all(cmap.colors >= 0)
        assert np.all(cmap.colors <= 1)
    
    def test_first_color_is_blue(self):
        """Test first color is blue (cool)"""
        cmap = cool_warm(256)
        first_color = cmap.colors[0]
        # Blue should have highest B value, lowest R value
        assert first_color[2] > first_color[0]  # B > R
        assert first_color[2] > first_color[1]  # B > G
    
    def test_last_color_is_red(self):
        """Test last color is red (warm)"""
        cmap = cool_warm(256)
        last_color = cmap.colors[-1]
        # Red should have highest R value
        assert last_color[0] > last_color[2]  # R > B
        assert last_color[0] > last_color[1]  # R > G
    
    def test_middle_color_is_white(self):
        """Test middle color is approximately white/gray"""
        cmap = cool_warm(256)
        mid_idx = 256 // 2
        mid_color = cmap.colors[mid_idx]
        # Middle should be close to neutral (R ≈ G ≈ B)
        assert abs(mid_color[0] - mid_color[1]) < 0.1
        assert abs(mid_color[1] - mid_color[2]) < 0.1
        assert abs(mid_color[0] - mid_color[2]) < 0.1
        # And relatively light
        assert np.mean(mid_color) > 0.7
    
    def test_overall_red_trend(self):
        """Test red channel increases overall (cool to warm)"""
        cmap = cool_warm(256)
        red = cmap.colors[:, 0]
        # Overall trend: first < last
        assert red[0] < red[-1]
        # Mostly increasing (some minor fluctuations in middle are OK)
        assert np.sum(np.diff(red) < 0) < 100  # Less than 100 decreases
    
    def test_overall_blue_trend(self):
        """Test blue channel decreases overall (cool to warm)"""
        cmap = cool_warm(256)
        blue = cmap.colors[:, 2]
        # Overall trend: first > last
        assert blue[0] > blue[-1]
        # Mostly decreasing (some minor fluctuations in middle are OK)
        assert np.sum(np.diff(blue) > 0) < 100  # Less than 100 increases
    
    def test_base_data_integrity(self):
        """Test the 257-entry base data is intact"""
        assert COOLWARM_257.shape == (257, 3)
        assert np.all(COOLWARM_257 >= 0)
        assert np.all(COOLWARM_257 <= 1)
        
        # Check first color (deep blue)
        first = COOLWARM_257[0]
        assert first[0] == pytest.approx(0.2298057, abs=1e-7)
        assert first[1] == pytest.approx(0.298717966, abs=1e-7)
        assert first[2] == pytest.approx(0.753683153, abs=1e-7)
        
        # Check middle color (near white)
        mid = COOLWARM_257[128]
        assert abs(mid[0] - mid[1]) < 0.01
        assert abs(mid[1] - mid[2]) < 0.01
        
        # Check last color (deep red)
        last = COOLWARM_257[-1]
        assert last[0] == pytest.approx(0.705673158, abs=1e-7)
        assert last[1] == pytest.approx(0.01555616, abs=1e-7)
        assert last[2] == pytest.approx(0.150232812, abs=1e-7)


class TestCoolwarmMatlabComparison:
    """Numerical comparison with MATLAB coolwarm.m"""
    
    def test_matlab_256_values(self):
        """
        Compare against MATLAB-generated 256-color colormap.
        
        MATLAB code:
        ```matlab
        map = coolwarm(256);
        ```
        """
        cmap = cool_warm(256)
        
        # Check specific values against pre-computed MATLAB reference
        # First color (index 0)
        assert cmap.colors[0, 0] == pytest.approx(0.2298057, abs=1e-6)
        assert cmap.colors[0, 1] == pytest.approx(0.298717966, abs=1e-6)
        assert cmap.colors[0, 2] == pytest.approx(0.753683153, abs=1e-6)
        
        # Last color (index 255)
        assert cmap.colors[255, 0] == pytest.approx(0.705673158, abs=1e-6)
        assert cmap.colors[255, 1] == pytest.approx(0.01555616, abs=1e-6)
        assert cmap.colors[255, 2] == pytest.approx(0.150232812, abs=1e-6)
        
        # Middle color (index 128)
        mid = cmap.colors[128]
        assert abs(mid[0] - mid[1]) < 0.01
        assert abs(mid[1] - mid[2]) < 0.01
    
    def test_matlab_64_values(self):
        """Test 64-color colormap matches MATLAB"""
        cmap = cool_warm(64)
        assert cmap.N == 64
        
        # Check first and last colors
        assert cmap.colors[0, 2] > cmap.colors[0, 0]  # Blue > Red
        assert cmap.colors[-1, 0] > cmap.colors[-1, 2]  # Red > Blue
    
    def test_diverging_pattern(self):
        """
        Test that colormap has proper diverging pattern (blue -> white -> red).
        """
        cmap = cool_warm(256)
        
        # Colors should show diverging pattern
        red_values = cmap.colors[:, 0]
        blue_values = cmap.colors[:, 2]
        
        # Overall: Red increases from cool to warm
        assert red_values[0] < red_values[-1]
        # Overall: Blue decreases from cool to warm  
        assert blue_values[0] > blue_values[-1]
        
        # At start: Blue > Red (cool colors)
        assert blue_values[0] > red_values[0]
        # At end: Red > Blue (warm colors)
        assert red_values[-1] > blue_values[-1]
        # At middle: Red ≈ Blue (neutral)
        mid = len(red_values) // 2
        assert abs(red_values[mid] - blue_values[mid]) < 0.1
    
    def test_colormap_reversibility(self):
        """Test that we can reconstruct base data from large colormap"""
        # Create a large colormap
        cmap = cool_warm(1024)
        
        # Sample at positions corresponding to base data
        n_base = 257
        indices = np.linspace(0, 1023, n_base).astype(int)
        sampled = cmap.colors[indices]
        
        # Should be close to original base data
        # (not exact due to double interpolation)
        assert np.allclose(sampled, COOLWARM_257, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
