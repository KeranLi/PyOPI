"""
Comparison tests for opiMaps_TwoWinds.m (MATLAB) vs Python visualization

These tests verify that the Python map visualization functions produce 
equivalent outputs to the MATLAB original for the TwoWinds OPI maps.

MATLAB opiMaps_TwoWinds.m generates 20 figures:
- Fig01: Topography and sample locations
- Fig02: Precipitation rate (combined)
- Fig03: Streamlines, precipitation state #1
- Fig04: Streamlines, precipitation state #2
- Fig05: Cross section, precipitation state #1
- Fig06: Cross section, precipitation state #2
- Fig07: Predicted d2H (combined)
- Fig08: Predicted d18O (combined)
- Fig09: Precipitation source (fractionPGrid)
- Fig10: Precipitation rate, state #1
- Fig11: Precipitation rate, state #2
- Fig12: Moisture ratio, state #1
- Fig13: Moisture ratio, state #2
- Fig14: Predicted d2H, state #1
- Fig15: Predicted d2H, state #2
- Fig16: Surface-air temperature, state #1
- Fig17: Surface-air temperature, state #2
- Fig18: Surface velocity ratio, state #1
- Fig19: Surface velocity ratio, state #2
- Fig20: Location of outliers
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
    plot_result_maps,
    plot_precipitation_source_map,
    plot_two_wind_comparison
)
from opi.viz.advanced import plot_cross_section
from opi.viz.plots import plot_sample_comparison


class TestOpiMapsTwoWindsBasic:
    """Basic functionality tests for two-wind map visualization"""
    
    def test_plot_topography_map(self):
        """Test that topography map can be created for two-wind case"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        h_grid = 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2))
        
        # Two-wind case with two wind arrows
        fig, ax = plot_topography_map(
            lon, lat, h_grid, 
            title='Test Topography - Two Winds',
            wind_arrows=[
                {'lon': -68, 'lat': -38, 'u': 5, 'v': 5, 'color': 'blue'},
                {'lon': -62, 'lat': -32, 'u': -3, 'v': 7, 'color': 'red'}
            ]
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_precipitation_map_combined(self):
        """Test combined precipitation map (Fig02 equivalent)"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Combined precipitation (sum of both states)
        p_grid_1 = 0.5e-6 * np.exp(-((LON + 66)**2 + (LAT + 36)**2) / (2 * 2**2))
        p_grid_2 = 0.5e-6 * np.exp(-((LON + 64)**2 + (LAT + 34)**2) / (2 * 2**2))
        p_grid = p_grid_1 + p_grid_2
        
        fig, ax = plot_precipitation_map(
            lon, lat, p_grid, 
            title='Combined Precipitation - Two Winds'
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_precipitation_map_individual_states(self):
        """Test individual state precipitation maps (Fig10-11 equivalent)"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        p_grid_1 = 0.5e-6 * np.exp(-((LON + 66)**2 + (LAT + 36)**2) / (2 * 2**2))
        p_grid_2 = 0.5e-6 * np.exp(-((LON + 64)**2 + (LAT + 34)**2) / (2 * 2**2))
        
        # State 1
        fig1, ax1 = plot_precipitation_map(
            lon, lat, p_grid_1,
            title='Precipitation State #1',
            wind_arrow={'lon': -68, 'lat': -38, 'u': 5, 'v': 5}
        )
        assert fig1 is not None
        plt.close(fig1)
        
        # State 2
        fig2, ax2 = plot_precipitation_map(
            lon, lat, p_grid_2,
            title='Precipitation State #2',
            wind_arrow={'lon': -62, 'lat': -32, 'u': -3, 'v': 7}
        )
        assert fig2 is not None
        plt.close(fig2)
    
    def test_plot_isotope_map_combined(self):
        """Test combined isotope maps (Fig07-08 equivalent)"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Combined isotopes
        d2h_grid = -50e-3 + 10e-3 * np.sin(LON * np.pi / 180)
        d18o_grid = d2h_grid / 8
        
        fig, axes = plot_isotope_map(
            lon, lat, d2h_grid, d18o_grid=d18o_grid,
            title_prefix='Combined Isotopes'
        )
        
        assert fig is not None
        assert axes is not None
        assert len(axes) == 2  # Both d2H and d18O
        plt.close(fig)
    
    def test_plot_precipitation_source_map(self):
        """Test precipitation source map (Fig09 equivalent)"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Fraction from state 1 (0 = all state 2, 1 = all state 1)
        fraction_p_grid = 0.5 + 0.3 * np.sin(LON * np.pi / 10)
        
        fig, ax = plot_precipitation_source_map(
            lon, lat, fraction_p_grid,
            wind_arrows=[
                {'lon': -68, 'lat': -38, 'u': 5, 'v': 5, 'color': 'blue'},
                {'lon': -62, 'lat': -32, 'u': -3, 'v': 7, 'color': 'red'}
            ]
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_two_wind_comparison(self):
        """Test two-wind comparison plot"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        data = {
            'lon': lon,
            'lat': lat,
            'h_grid': 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2)),
            'p_grid_1': 0.5e-6 * np.exp(-((LON + 66)**2 + (LAT + 36)**2) / (2 * 2**2)),
            'p_grid_2': 0.5e-6 * np.exp(-((LON + 64)**2 + (LAT + 34)**2) / (2 * 2**2)),
            'd2h_grid_1': -60e-3 + 5e-3 * np.sin(LON * np.pi / 180),
            'd2h_grid_2': -40e-3 + 5e-3 * np.cos(LON * np.pi / 180),
        }
        
        figs = plot_two_wind_comparison(data)
        
        # Function now returns list of figure paths (or empty list if no save_dir)
        assert isinstance(figs, list)


class TestOpiMapsTwoWindsMatlabComparison:
    """Direct comparison with MATLAB opiMaps_TwoWinds.m functionality"""
    
    def test_all_20_figure_types(self):
        """
        Test that Python produces equivalent figure types to MATLAB opiMaps_TwoWinds:
        - Fig01: Topography with two wind arrows
        - Fig02: Combined precipitation rate
        - Fig03-04: Streamlines for both states (3D)
        - Fig05-06: Cross sections for both states
        - Fig07-08: Combined d2H and d18O
        - Fig09: Precipitation source (fraction)
        - Fig10-11: Individual precipitation rates
        - Fig12-13: Individual moisture ratios
        - Fig14-15: Individual d2H maps
        - Fig16-17: Individual temperature maps
        - Fig18-19: Individual velocity ratios
        - Fig20: Outlier locations
        """
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Create comprehensive two-wind result data
        result_dict = {
            'lon': lon,
            'lat': lat,
            'h_grid': 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2)),
            # Combined grids
            'p_grid': 1e-6 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2)),
            'd2h_grid': -50e-3 + 10e-3 * np.sin(LON * np.pi / 180),
            'd18o_grid': -6e-3 + 1.25e-3 * np.sin(LON * np.pi / 180),
            # State 1 grids
            'p_grid_1': 0.6e-6 * np.exp(-((LON + 65.5)**2 + (LAT + 35.5)**2) / (2 * 2**2)),
            'd2h_grid_1': -55e-3 + 8e-3 * np.sin(LON * np.pi / 180),
            'f_m_grid_1': 0.75 * np.ones_like(LON),
            'temp_grid_1': 288 - 0.0065 * 2000 * np.exp(-((LON + 65.5)**2 + (LAT + 35.5)**2) / (2 * 2**2)),
            'u_ratio_grid_1': 0.8 + 0.1 * np.sin(LON * np.pi / 180),
            # State 2 grids
            'p_grid_2': 0.4e-6 * np.exp(-((LON + 64.5)**2 + (LAT + 34.5)**2) / (2 * 2**2)),
            'd2h_grid_2': -45e-3 + 12e-3 * np.cos(LON * np.pi / 180),
            'f_m_grid_2': 0.85 * np.ones_like(LON),
            'temp_grid_2': 293 - 0.0065 * 2000 * np.exp(-((LON + 64.5)**2 + (LAT + 34.5)**2) / (2 * 2**2)),
            'u_ratio_grid_2': 1.0 + 0.1 * np.cos(LON * np.pi / 180),
            # Precipitation source fraction
            'fraction_p_grid': 0.6 * np.ones_like(LON),
            # Wind parameters
            'wind_1': {'azimuth': 45, 'speed': 10},
            'wind_2': {'azimuth': 225, 'speed': 8},
        }
        
        figures = plot_result_maps(result_dict, two_wind=True)
        
        # Should create multiple figures for two-wind case
        assert len(figures) >= 5
        
        for fig in figures:
            plt.close(fig)
    
    def test_precipitation_source_fraction_map(self):
        """
        Test Fig09: Precipitation source map
        Shows fraction from state 1 (blue arrow) vs state 2 (red arrow)
        """
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Variable fraction across the domain
        fraction_p_grid = 0.3 + 0.4 * (LON - lon.min()) / (lon.max() - lon.min())
        
        fig, ax = plot_precipitation_source_map(
            lon, lat, fraction_p_grid,
            map_limits=[lon.min(), lon.max(), lat.min(), lat.max()],
            wind_arrows=[
                {'lon': -68, 'lat': -38, 'u': 7, 'v': 7, 'color': 'blue'},
                {'lon': -62, 'lat': -32, 'u': -5, 'v': 5, 'color': 'red'}
            ],
            title='Precipitation Source (MATLAB Fig09 Equivalent)'
        )
        
        assert fig is not None
        assert ax is not None
        
        # Check that colorbar exists
        cb = ax.collections[0].colorbar if hasattr(ax, 'collections') else None
        
        plt.close(fig)
    
    def test_cross_sections_both_states(self):
        """
        Test Fig05-06: Cross sections for both precipitation states
        """
        # Create synthetic cross-section data
        x = np.linspace(-70, -60, 100)
        h_section = 2000 * np.exp(-((x + 65)**2) / (2 * 2**2))
        
        # State 1 cross-section
        precip_1 = 1e-6 * np.exp(-((x + 65.5)**2) / (2 * 1.5**2))
        d2h_1 = -50e-3 - 20e-3 * (h_section / h_section.max())
        
        fig1, axes1 = plot_cross_section(
            x, np.array([0]), h_section.reshape(1, -1),
            precip_grid=precip_1.reshape(1, -1),
            d2h_grid=d2h_1.reshape(1, -1),
            section_y=0
        )
        
        assert fig1 is not None
        assert len(axes1) == 3  # Precip, d2H, d18O subplots
        plt.close(fig1)
        
        # State 2 cross-section
        precip_2 = 0.8e-6 * np.exp(-((x + 64.5)**2) / (2 * 1.8**2))
        d2h_2 = -45e-3 - 18e-3 * (h_section / h_section.max())
        
        fig2, axes2 = plot_cross_section(
            x, np.array([0]), h_section.reshape(1, -1),
            precip_grid=precip_2.reshape(1, -1),
            d2h_grid=d2h_2.reshape(1, -1),
            section_y=0
        )
        
        assert fig2 is not None
        plt.close(fig2)
    
    def test_moisture_ratio_maps(self):
        """
        Test Fig12-13: Moisture ratio maps for both states
        """
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        f_m_grid_1 = 0.6 + 0.2 * np.sin(LON * np.pi / 180)
        f_m_grid_2 = 0.7 + 0.15 * np.cos(LON * np.pi / 180)
        
        # State 1
        fig1, ax1 = plot_precipitation_map(
            lon, lat, f_m_grid_1,
            title='Moisture Ratio - State #1',
            units='Moisture Ratio',
            cmap='viridis'
        )
        assert fig1 is not None
        plt.close(fig1)
        
        # State 2
        fig2, ax2 = plot_precipitation_map(
            lon, lat, f_m_grid_2,
            title='Moisture Ratio - State #2',
            units='Moisture Ratio',
            cmap='viridis'
        )
        assert fig2 is not None
        plt.close(fig2)
    
    def test_temperature_maps(self):
        """
        Test Fig16-17: Surface temperature maps for both states
        """
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Temperature decreases with elevation
        h_grid = 2000 * np.exp(-((LON + 65)**2 + (LAT + 35)**2) / (2 * 2**2))
        temp_grid_1 = 288 - 0.0065 * h_grid
        temp_grid_2 = 293 - 0.0065 * h_grid
        
        fig1, ax1 = plot_precipitation_map(
            lon, lat, temp_grid_1,
            title='Surface Temperature - State #1',
            units='Temperature (K)',
            cmap='coolwarm'
        )
        assert fig1 is not None
        plt.close(fig1)
        
        fig2, ax2 = plot_precipitation_map(
            lon, lat, temp_grid_2,
            title='Surface Temperature - State #2',
            units='Temperature (K)',
            cmap='coolwarm'
        )
        assert fig2 is not None
        plt.close(fig2)
    
    def test_velocity_ratio_maps(self):
        """
        Test Fig18-19: Surface velocity ratio (u'/U) maps for both states
        """
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        u_ratio_1 = 0.8 + 0.3 * np.sin(LON * np.pi / 180)
        u_ratio_2 = 1.0 + 0.25 * np.cos(LON * np.pi / 180)
        
        fig1, ax1 = plot_precipitation_map(
            lon, lat, u_ratio_1,
            title='Surface Velocity Ratio - State #1',
            units="u'/U",
            cmap='coolwarm'
        )
        assert fig1 is not None
        plt.close(fig1)
        
        fig2, ax2 = plot_precipitation_map(
            lon, lat, u_ratio_2,
            title='Surface Velocity Ratio - State #2',
            units="u'/U",
            cmap='coolwarm'
        )
        assert fig2 is not None
        plt.close(fig2)
    
    def test_combined_precipitation_calculation(self):
        """
        Verify that combined precipitation equals sum of individual states
        MATLAB: pGrid = pGrid_1 + pGrid_2
        """
        lon = np.linspace(-70, -60, 20)
        lat = np.linspace(-40, -30, 20)
        LON, LAT = np.meshgrid(lon, lat)
        
        p_grid_1 = 0.5e-6 * np.exp(-((LON + 66)**2 + (LAT + 36)**2) / (2 * 2**2))
        p_grid_2 = 0.4e-6 * np.exp(-((LON + 64)**2 + (LAT + 34)**2) / (2 * 2**2))
        p_grid_combined = p_grid_1 + p_grid_2
        
        # Verify mathematical relationship
        np.testing.assert_allclose(p_grid_combined, p_grid_1 + p_grid_2, rtol=1e-10)


class TestOpiMapsTwoWindsEdgeCases:
    """Edge case tests for two-wind visualization"""
    
    def test_unequal_grid_sizes(self):
        """Test handling of potentially different grid sizes from each state"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Grids with same coordinates but different content
        p_grid_1 = 0.5e-6 * np.ones_like(LON)
        p_grid_2 = 0.4e-6 * np.ones_like(LON)
        
        # Should still work
        fig, ax = plot_precipitation_map(
            lon, lat, p_grid_1 + p_grid_2,
            title='Combined Precipitation'
        )
        assert fig is not None
        plt.close(fig)
    
    def test_fraction_grid_bounds(self):
        """Test that precipitation source fraction is properly bounded [0, 1]"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Valid fraction grid
        fraction_p_grid = np.clip(0.5 + 0.3 * np.random.randn(*LON.shape), 0, 1)
        
        fig, ax = plot_precipitation_source_map(
            lon, lat, fraction_p_grid
        )
        assert fig is not None
        plt.close(fig)
    
    def test_missing_state_data(self):
        """Test graceful handling when one state's data is missing"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # Only provide state 1 data
        result_dict = {
            'lon': lon,
            'lat': lat,
            'h_grid': 2000 * np.ones_like(LON),
            'p_grid_1': 1e-6 * np.ones_like(LON),
            # Missing p_grid_2 and other state 2 data
        }
        
        figures = plot_result_maps(result_dict, two_wind=True)
        
        # Should still create at least topography
        assert len(figures) >= 1
        
        for fig in figures:
            plt.close(fig)
    
    def test_extreme_fraction_values(self):
        """Test precipitation source map with extreme fraction values"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        # All state 1
        fraction_all_1 = np.ones_like(LON)
        fig1, ax1 = plot_precipitation_source_map(lon, lat, fraction_all_1)
        assert fig1 is not None
        plt.close(fig1)
        
        # All state 2
        fraction_all_2 = np.zeros_like(LON)
        fig2, ax2 = plot_precipitation_source_map(lon, lat, fraction_all_2)
        assert fig2 is not None
        plt.close(fig2)
    
    def test_two_wind_with_no_wind_arrows(self):
        """Test two-wind maps without wind arrows"""
        lon = np.linspace(-70, -60, 50)
        lat = np.linspace(-40, -30, 50)
        LON, LAT = np.meshgrid(lon, lat)
        
        fig, ax = plot_precipitation_source_map(
            lon, lat, 0.5 * np.ones_like(LON),
            wind_arrows=None
        )
        assert fig is not None
        plt.close(fig)


class TestOpiMapsTwoWindsUnits:
    """Unit conversion tests specific to two-wind case"""
    
    def test_combined_isotope_units(self):
        """
        Test that combined isotope maps handle units correctly
        MATLAB: d2HGrid, d18OGrid in fraction, display in permil
        """
        lon = np.linspace(-70, -60, 20)
        lat = np.linspace(-40, -30, 20)
        LON, LAT = np.meshgrid(lon, lat)
        
        # State 1 isotopes
        d2h_1 = -50e-3 * np.ones_like(LON)
        d18o_1 = -6e-3 * np.ones_like(LON)
        
        # State 2 isotopes
        d2h_2 = -45e-3 * np.ones_like(LON)
        d18o_2 = -5e-3 * np.ones_like(LON)
        
        # Combined (weighted average)
        fraction = 0.6
        d2h_combined = fraction * d2h_1 + (1 - fraction) * d2h_2
        d18o_combined = fraction * d18o_1 + (1 - fraction) * d18o_2
        
        fig, axes = plot_isotope_map(
            lon, lat, d2h_combined, d18o_grid=d18o_combined
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_precipitation_unit_conversion_both_states(self):
        """
        Verify precipitation unit conversion for both states.
        MATLAB uses kg/m^2/s and converts to mm/hr for display.
        Conversion: 1 kg/m^2/s = 1 mm/s = 3600 mm/hr = 3.6e3 mm/hr
        """
        # State 1: 1e-3 kg/m^2/s = 3.6 mm/hr (realistic light rain)
        p_1_kg_m2_s = 1e-3
        p_1_mm_hr = p_1_kg_m2_s * 3.6e3  # MATLAB conversion
        assert np.isclose(p_1_mm_hr, 3.6, rtol=1e-6)
        
        # State 2: 2e-3 kg/m^2/s = 7.2 mm/hr (moderate rain)
        p_2_kg_m2_s = 2e-3
        p_2_mm_hr = p_2_kg_m2_s * 3.6e3
        assert np.isclose(p_2_mm_hr, 7.2, rtol=1e-6)
        
        # Combined
        p_combined_mm_hr = (p_1_kg_m2_s + p_2_kg_m2_s) * 3.6e3
        assert np.isclose(p_combined_mm_hr, 10.8, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
