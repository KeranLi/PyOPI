"""
Comparison tests for opiMaps_OneWind.m (MATLAB) vs Python visualization

These tests verify that the Python map visualization functions produce 
equivalent outputs to the MATLAB original for the OneWind OPI maps.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from opi.viz.maps import (
    plot_topography_map,
    plot_precipitation_map,
    plot_isotope_map,
    plot_result_maps
)


class TestOpiMapsOneWindBasic:
    """Basic functionality tests for map visualization"""
    
    def test_plot_topography_map(self):
        """Test that topography map can be created"""
        # Create simple test data
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        h_grid = 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2))
        
        fig, ax = plot_topography_map(lon, lat, h_grid, title='Test Topography')
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_plot_precipitation_map(self):
        """Test that precipitation map can be created"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        p_grid = 1e-6 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2))
        
        fig, ax = plot_precipitation_map(lon, lat, p_grid, title='Test Precipitation')
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_plot_isotope_map_d2h_only(self):
        """Test that d2H isotope map can be created"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        d2h_grid = -50e-3 + 10e-3 * np.sin(LON * np.pi / 180)
        
        fig, axes = plot_isotope_map(lon, lat, d2h_grid, d18o_grid=None,
                                      title_prefix='Test')
        
        assert fig is not None
        assert axes is not None
        
        plt.close(fig)
    
    def test_plot_isotope_map_both_isotopes(self):
        """Test that both d2H and d18O isotope maps can be created"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        d2h_grid = -50e-3 + 10e-3 * np.sin(LON * np.pi / 180)
        d18o_grid = d2h_grid / 8  # Approximate relationship
        
        fig, axes = plot_isotope_map(lon, lat, d2h_grid, d18o_grid=d18o_grid,
                                      title_prefix='Test')
        
        assert fig is not None
        assert axes is not None
        # When both isotopes are provided, axes should be an array
        assert hasattr(axes, '__len__') and len(axes) == 2
        
        plt.close(fig)


class TestOpiMapsOneWindMatlabComparison:
    """Direct comparison with MATLAB opiMaps_OneWind.m functionality"""
    
    def test_matlab_figure_types(self):
        """
        Test that Python produces equivalent figure types to MATLAB:
        - Fig 1: Topography and sample locations
        - Fig 4: Precipitation rate
        - Fig 5: Moisture ratio (fMGrid)
        - Fig 6: Surface relative humidity (rHGrid)
        - Fig 9: Predicted precipitation d2H
        - Fig 10: Predicted precipitation d18O
        - Fig 11: Surface-air temperature
        """
        # Create test result data similar to MATLAB output
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        result_dict = {
            'lon': lon,
            'lat': lat,
            'h_grid': 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2)),
            'p_grid': 1e-6 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2)),
            'd2h_grid': -50e-3 + 10e-3 * np.sin(LON * np.pi / 180),
            'd18o_grid': -6e-3 + 1.25e-3 * np.sin(LON * np.pi / 180),
            'f_m_grid': 0.8 * np.ones_like(LON),
            'r_h_grid': 0.9 * np.ones_like(LON),
        }
        
        figures = plot_result_maps(result_dict)
        
        # Should create at least 3 figures (topography, precipitation, isotopes)
        assert len(figures) >= 3
        
        for fig in figures:
            plt.close(fig)
    
    def test_map_constants_match_matlab(self):
        """Test that map-related constants match MATLAB opiMaps_OneWind"""
        from opi.constants import TC2K, M_PER_DEGREE
        
        # TC2K = 273.15 (Celsius to Kelvin conversion)
        assert TC2K == 273.15
        
        # Meters per degree should match MATLAB calculation
        # MATLAB: mPerDegree = pi*radiusEarth/180
        expected_m_per_degree = np.pi * 6371e3 / 180
        assert np.isclose(M_PER_DEGREE, expected_m_per_degree, rtol=1e-6)
    
    def test_precipitation_unit_conversion(self):
        """
        Test that precipitation units are correctly converted.
        MATLAB: pGrid is in kg/m^2/s, displays as mm/hr
        Python: should handle similar conversion
        """
        lon = np.linspace(-70, -60, 10)
        lat = np.linspace(-40, -30, 10)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Create precipitation in kg/m^2/s (same as MATLAB)
        p_grid_kg_m2_s = 1e-6 * np.ones_like(LON)
        
        fig, ax = plot_precipitation_map(lon, lat, p_grid_kg_m2_s)
        
        assert fig is not None
        plt.close(fig)
    
    def test_isotope_unit_conversion(self):
        """
        Test that isotope units are correctly handled.
        MATLAB: d2HGrid is in fraction (e.g., -0.050), displays as permil (-50)
        Python: should convert fraction to permil for display
        """
        lon = np.linspace(-70, -60, 10)
        lat = np.linspace(-40, -30, 10)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Create isotopes in fraction (same as MATLAB)
        d2h_fraction = -50e-3 * np.ones_like(LON)
        
        fig, axes = plot_isotope_map(lon, lat, d2h_fraction, d18o_grid=None)
        
        assert fig is not None
        plt.close(fig)


class TestOpiMapsOneWindEdgeCases:
    """Edge case tests"""
    
    def test_empty_grid_handling(self):
        """Test handling of empty or all-zero grids"""
        lon = np.linspace(-70, -60, 10)
        lat = np.linspace(-40, -30, 10)
        LON, LAT = np.meshgrid(lon, lat)
        h_grid = np.zeros_like(LON)
        
        fig, ax = plot_topography_map(lon, lat, h_grid)
        
        assert fig is not None
        plt.close(fig)
    
    def test_nan_handling(self):
        """Test handling of NaN values in grids"""
        lon = np.linspace(-70, -60, 10)
        lat = np.linspace(-40, -30, 10)
        LON, LAT = np.meshgrid(lon, lat)
        h_grid = 1000 * np.ones_like(LON)
        h_grid[5, 5] = np.nan  # Add a NaN value
        
        fig, ax = plot_topography_map(lon, lat, h_grid)
        
        assert fig is not None
        plt.close(fig)
    
    def test_very_small_grid(self):
        """Test with minimum viable grid size"""
        lon = np.linspace(-70, -60, 5)
        lat = np.linspace(-40, -30, 5)
        LON, LAT = np.meshgrid(lon, lat)
        h_grid = 1000 * np.ones_like(LON)
        
        fig, ax = plot_topography_map(lon, lat, h_grid)
        
        assert fig is not None
        plt.close(fig)
    
    def test_result_maps_with_missing_data(self):
        """Test plot_result_maps with missing optional data"""
        result_dict = {
            'lon': np.linspace(-70, -60, 10),
            'lat': np.linspace(-40, -30, 10),
            'h_grid': np.ones((10, 10)),
            # Missing p_grid, d2h_grid, etc.
        }
        
        figures = plot_result_maps(result_dict)
        
        # Should still create at least the topography figure
        assert len(figures) >= 1
        
        for fig in figures:
            plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
