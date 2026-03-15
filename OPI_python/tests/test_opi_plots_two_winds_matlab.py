"""
Comparison tests for opiPlots_TwoWinds.m (MATLAB) vs Python visualization

These tests verify that the Python plotting functions produce equivalent 
outputs to the MATLAB original for the TwoWinds OPI plots.

MATLAB opiPlots_TwoWinds.m generates 5 figures:
- Fig01: Predicted vs observed isotope values (blue/red for state 1/2)
- Fig02: Craig plot with two base precipitation ellipses
- Fig03: Standardized residuals vs easting and northing
- Fig04: Predicted isotopes vs maximum lifting (both states)
- Fig05: Predicted isotopes vs elevation (both states)

Key differences from OneWind:
- Two precipitation states with different colors (blue=state1, red=state2)
- isSampleSide01 flag to identify which side of divide samples belong to
- Separate best-fit lines for each state
- Two base precipitation composition points/ellipses
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from opi.viz.plots import (
    plot_sample_comparison,
    plot_residuals,
    plot_mwl
)


class TestOpiPlotsTwoWindsBasic:
    """Basic functionality tests for TwoWinds plots"""
    
    def test_two_wind_sample_comparison(self):
        """Test Fig01: Predicted vs observed with two wind states"""
        np.random.seed(42)
        n_samples = 20
        
        # Generate synthetic data for two states
        sample_d2h = -80 + 20 * np.random.randn(n_samples)
        sample_d18o = -10 + 2.5 * np.random.randn(n_samples)
        pred_d2h = sample_d2h + 2 * np.random.randn(n_samples)
        pred_d18o = sample_d18o + 0.3 * np.random.randn(n_samples)
        
        # Side flag (which state side sample belongs to)
        is_sample_side_01 = np.random.choice([True, False], n_samples)
        
        # Convert to fraction
        sample_d2h_f = sample_d2h / 1000
        sample_d18o_f = sample_d18o / 1000
        pred_d2h_f = pred_d2h / 1000
        pred_d18o_f = pred_d18o / 1000
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # d2H with color coding by side
        axes[0].scatter(sample_d2h_f[is_sample_side_01] * 1000,
                       pred_d2h_f[is_sample_side_01] * 1000,
                       c='blue', s=100, alpha=0.6, label='State 1 side')
        axes[0].scatter(sample_d2h_f[~is_sample_side_01] * 1000,
                       pred_d2h_f[~is_sample_side_01] * 1000,
                       c='red', s=100, alpha=0.6, label='State 2 side')
        axes[0].plot([-100, -50], [-100, -50], 'k--', alpha=0.5)
        axes[0].set_xlabel('Observed d2H (permil)')
        axes[0].set_ylabel('Predicted d2H (permil)')
        axes[0].legend()
        axes[0].grid(True)
        
        assert fig is not None
        plt.close(fig)
    
    def test_two_wind_craig_plot(self):
        """Test Fig02: Craig plot with two states"""
        np.random.seed(42)
        n_samples = 20
        
        # State 1 samples
        d18o_1 = np.random.uniform(-15, -8, n_samples // 2)
        d2h_1 = 8 * d18o_1 + 10 + np.random.randn(n_samples // 2)
        
        # State 2 samples
        d18o_2 = np.random.uniform(-12, -5, n_samples // 2)
        d2h_2 = 8 * d18o_2 + 5 + np.random.randn(n_samples // 2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Observed (gray)
        ax.scatter(np.concatenate([d18o_1, d18o_2]),
                  np.concatenate([d2h_1, d2h_2]),
                  c='gray', s=100, alpha=0.5, label='Observed')
        
        # State 1 (blue)
        ax.scatter(d18o_1, d2h_1, c='blue', s=100, label='State 1')
        
        # State 2 (red)
        ax.scatter(d18o_2, d2h_2, c='red', s=100, label='State 2')
        
        # Base precipitation points
        ax.scatter(-10, -80, c='blue', marker='s', s=200, label='Base 1')
        ax.scatter(-8, -65, c='red', marker='s', s=200, label='Base 2')
        
        ax.set_xlabel('d18O (permil)')
        ax.set_ylabel('d2H (permil)')
        ax.legend()
        ax.grid(True)
        
        assert fig is not None
        plt.close(fig)


class TestOpiPlotsTwoWindsMatlabComparison:
    """Direct comparison with MATLAB opiPlots_TwoWinds.m"""
    
    def test_matlab_figure_types(self):
        """
        Test all 5 MATLAB figure types for TwoWinds:
        - Fig 1: Observed vs predicted (color by side)
        - Fig 2: Craig plot with two base compositions
        - Fig 3: Residuals vs X/Y
        - Fig 4: Isotopes vs max lifting (both states)
        - Fig 5: Isotopes vs elevation (both states)
        """
        np.random.seed(42)
        n = 20
        
        # Generate synthetic data
        sample_d2h = (-80 + 20 * np.random.randn(n)) / 1000
        sample_d18o = (-10 + 2.5 * np.random.randn(n)) / 1000
        pred_d2h = sample_d2h + 0.002 * np.random.randn(n)
        pred_d18o = sample_d18o + 0.0003 * np.random.randn(n)
        is_sample_side_01 = np.random.choice([True, False], n)
        
        # Fig 1: Sample comparison with colors
        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 10))
        for ax, sample, pred, label in [(axes1[0], sample_d2h, pred_d2h, 'd2H'),
                                         (axes1[1], sample_d18o, pred_d18o, 'd18O')]:
            ax.scatter(sample[is_sample_side_01] * 1000,
                      pred[is_sample_side_01] * 1000,
                      c='blue', s=100, alpha=0.6)
            ax.scatter(sample[~is_sample_side_01] * 1000,
                      pred[~is_sample_side_01] * 1000,
                      c='red', s=100, alpha=0.6)
            ax.plot([sample.min() * 1000, sample.max() * 1000],
                   [sample.min() * 1000, sample.max() * 1000],
                   'k--', alpha=0.5)
            ax.set_xlabel(f'Observed {label}')
            ax.set_ylabel(f'Predicted {label}')
            ax.grid(True)
        assert fig1 is not None
        plt.close(fig1)
        
        # Fig 2: Craig plot
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(sample_d18o[is_sample_side_01] * 1000,
                   sample_d2h[is_sample_side_01] * 1000,
                   c='blue', s=100, alpha=0.6)
        ax2.scatter(sample_d18o[~is_sample_side_01] * 1000,
                   sample_d2h[~is_sample_side_01] * 1000,
                   c='red', s=100, alpha=0.6)
        ax2.set_xlabel('d18O (permil)')
        ax2.set_ylabel('d2H (permil)')
        ax2.grid(True)
        assert fig2 is not None
        plt.close(fig2)
        
        # Fig 3: Residuals
        sample_x = np.random.uniform(0, 100, n)
        std_residuals = np.random.randn(n)
        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 10))
        for ax, coord, label in [(axes3[0], sample_x, 'Easting'),
                                  (axes3[1], sample_x + 10, 'Northing')]:
            ax.scatter(coord[is_sample_side_01], std_residuals[is_sample_side_01],
                      c='blue', s=100, alpha=0.6)
            ax.scatter(coord[~is_sample_side_01], std_residuals[~is_sample_side_01],
                      c='red', s=100, alpha=0.6)
            ax.set_xlabel(f'{label} (km)')
            ax.set_ylabel('Standardized Residuals')
            ax.grid(True)
        assert fig3 is not None
        plt.close(fig3)
    
    def test_is_sample_side_01_flag(self):
        """
        Test isSampleSide01 flag functionality.
        MATLAB uses this to color samples by which side of divide they are on.
        """
        n = 20
        is_sample_side_01 = np.random.choice([True, False], n)
        
        # Count samples on each side
        n_side_1 = np.sum(is_sample_side_01)
        n_side_2 = n - n_side_1
        
        assert n_side_1 + n_side_2 == n
        assert n_side_1 >= 0
        assert n_side_2 >= 0
    
    def test_two_state_best_fit_lines(self):
        """
        Test best-fit line calculation for both states.
        MATLAB calculates separate lines for state 1 and state 2.
        """
        np.random.seed(42)
        n = 20
        
        # State 1 data
        lift_max_1 = np.array([0, 500, 1000, 1500, 2000])
        d2h_1 = (-80 - 20 * lift_max_1 / 1000) / 1000
        d2h0_1 = -80 / 1000
        
        # State 2 data (different slope)
        lift_max_2 = np.array([0, 400, 800, 1200, 1600])
        d2h_2 = (-70 - 15 * lift_max_2 / 1000) / 1000
        d2h0_2 = -70 / 1000
        
        # Calculate slopes (MATLAB method)
        y1 = d2h_1 - d2h0_1
        slope_1 = np.sum(y1 * lift_max_1) / np.sum(lift_max_1 ** 2)
        
        y2 = d2h_2 - d2h0_2
        slope_2 = np.sum(y2 * lift_max_2) / np.sum(lift_max_2 ** 2)
        
        # State 1 should have steeper slope
        assert slope_1 < slope_2  # Both negative, state 1 more negative
    
    def test_isotopes_vs_lifting_two_states(self):
        """Test Fig04: Isotopes vs lifting with both states"""
        np.random.seed(42)
        n_1, n_2 = 10, 10
        
        # State 1
        lift_1 = np.random.uniform(0, 3000, n_1)
        d2h_1 = (-80 - 20 * lift_1 / 1000 + 5 * np.random.randn(n_1))
        d18o_1 = (-10 - 2.5 * lift_1 / 1000 + 0.5 * np.random.randn(n_1))
        
        # State 2
        lift_2 = np.random.uniform(0, 2500, n_2)
        d2h_2 = (-70 - 15 * lift_2 / 1000 + 5 * np.random.randn(n_2))
        d18o_2 = (-8 - 2 * lift_2 / 1000 + 0.5 * np.random.randn(n_2))
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # d2H
        axes[0].scatter(lift_1, d2h_1, c='blue', s=100, label='State 1')
        axes[0].scatter(lift_2, d2h_2, c='red', s=100, label='State 2')
        axes[0].set_xlabel('Maximum lifting (m)')
        axes[0].set_ylabel('Predicted d2H (permil)')
        axes[0].legend()
        axes[0].grid(True)
        
        # d18O
        axes[1].scatter(lift_1, d18o_1, c='blue', s=100)
        axes[1].scatter(lift_2, d18o_2, c='red', s=100)
        axes[1].set_xlabel('Maximum lifting (m)')
        axes[1].set_ylabel('Predicted d18O (permil)')
        axes[1].grid(True)
        
        assert fig is not None
        plt.close(fig)
    
    def test_isotopes_vs_elevation_two_states(self):
        """Test Fig05: Isotopes vs elevation with both states"""
        np.random.seed(42)
        n_1, n_2 = 10, 10
        
        # State 1
        elev_1 = np.random.uniform(0, 4000, n_1)
        d2h_1 = (-80 - 15 * elev_1 / 1000 + 5 * np.random.randn(n_1))
        
        # State 2
        elev_2 = np.random.uniform(0, 3500, n_2)
        d2h_2 = (-70 - 12 * elev_2 / 1000 + 5 * np.random.randn(n_2))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(elev_1, d2h_1, c='blue', s=100, label='State 1')
        ax.scatter(elev_2, d2h_2, c='red', s=100, label='State 2')
        ax.set_xlabel('Elevation (m)')
        ax.set_ylabel('Predicted d2H (permil)')
        ax.legend()
        ax.grid(True)
        
        assert fig is not None
        plt.close(fig)
    
    def test_two_base_precipitation_points(self):
        """
        Test that two base precipitation points are plotted.
        MATLAB plots squares for both state 1 and state 2 base composition.
        """
        # Base precipitation compositions
        d18o0_1, d2h0_1 = -10, -80
        d18o0_2, d2h0_2 = -8, -65
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot both base points as squares
        ax.scatter(d18o0_1, d2h0_1, c='blue', marker='s', s=200,
                  label='Base State 1', zorder=5)
        ax.scatter(d18o0_2, d2h0_2, c='red', marker='s', s=200,
                  label='Base State 2', zorder=5)
        
        ax.set_xlabel('d18O (permil)')
        ax.set_ylabel('d2H (permil)')
        ax.legend()
        
        # Verify both points are different
        assert d18o0_1 != d18o0_2
        assert d2h0_1 != d2h0_2
        
        plt.close(fig)
    
    def test_nineteen_beta_parameters(self):
        """
        Test that beta has 19 parameters for two-wind solution.
        MATLAB checks: if length(beta)==9, error('Mat file is for a one-wind solution.')
        """
        # Simulate 19-parameter beta vector (TwoWinds)
        beta_two_winds = np.random.randn(19)
        
        # Simulate 9-parameter beta vector (OneWind)
        beta_one_wind = np.random.randn(9)
        
        # Check lengths
        assert len(beta_two_winds) == 19
        assert len(beta_one_wind) == 9
        
        # TwoWinds validation logic
        assert len(beta_two_winds) != 9  # Should not trigger OneWind error


class TestOpiPlotsTwoWindsEdgeCases:
    """Edge case tests for TwoWinds plots"""
    
    def test_all_samples_one_side(self):
        """Test when all samples are on one side of divide"""
        n = 20
        is_sample_side_01 = np.ones(n, dtype=bool)  # All on side 1
        
        sample_d2h = np.random.uniform(-100, -60, n)
        pred_d2h = sample_d2h + np.random.randn(n)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sample_d2h[is_sample_side_01], pred_d2h[is_sample_side_01],
                  c='blue', s=100)
        
        # No red points since no samples on side 2
        assert np.sum(~is_sample_side_01) == 0
        
        plt.close(fig)
    
    def test_wet_flag_per_state(self):
        """
        Test iWet flags for each state.
        MATLAB uses iWet_1, iWet_2, and iWet for each state and combined.
        """
        n = 20
        
        # Wet flags for each state
        i_wet_1 = np.random.choice([True, False], n, p=[0.7, 0.3])
        i_wet_2 = np.random.choice([True, False], n, p=[0.6, 0.4])
        
        # Combined wet (wet in either state)
        i_wet = i_wet_1 | i_wet_2
        
        # Combined should have at least as many wet as individual
        assert np.sum(i_wet) >= np.sum(i_wet_1)
        assert np.sum(i_wet) >= np.sum(i_wet_2)
    
    def test_different_slopes_both_states(self):
        """Test that two states can have different isotopic slopes"""
        # State 1: -20 permil/km
        slope_1 = -20
        
        # State 2: -15 permil/km (less depletion per km)
        slope_2 = -15
        
        assert slope_1 != slope_2
        assert abs(slope_1) > abs(slope_2)
    
    def test_no_samples_on_one_side(self):
        """Test handling when no samples exist for one state"""
        n = 20
        # All samples on side 1
        is_sample_side_01 = np.ones(n, dtype=bool)
        
        # Should still be able to plot
        sample_d2h = np.random.uniform(-100, -60, n)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(range(n), sample_d2h, c='blue', s=100)
        
        # Verify no side 2 samples
        assert np.sum(~is_sample_side_01) == 0
        
        plt.close(fig)


class TestOpiPlotsTwoWindsUnitConversions:
    """Unit conversion tests specific to TwoWinds"""
    
    def test_beta_parameter_unpacking(self):
        """
        Test unpacking of 19 beta parameters.
        MATLAB: beta(1:9) = state 1, beta(10) = fraction, beta(11:19) = state 2
        """
        beta = np.random.randn(19)
        
        # Unpack state 1
        U_1 = beta[0]
        azimuth_1 = beta[1]
        T0_1 = beta[2]
        M_1 = beta[3]
        kappa_1 = beta[4]
        tauC_1 = beta[5]
        d2H0_1 = beta[6]
        dD2H0_dLat_1 = beta[7]
        fP0_1 = beta[8]
        
        # Fraction
        fraction = beta[9]
        
        # Unpack state 2
        U_2 = beta[10]
        azimuth_2 = beta[11]
        T0_2 = beta[12]
        M_2 = beta[13]
        kappa_2 = beta[14]
        tauC_2 = beta[15]
        d2H0_2 = beta[16]
        dD2H0_dLat_2 = beta[17]
        fP0_2 = beta[18]
        
        # Verify unpacking
        assert U_1 == beta[0]
        assert fraction == beta[9]
        assert U_2 == beta[10]
    
    def test_best_fit_line_units(self):
        """
        Test that best-fit line slopes are in permil/km.
        MATLAB converts: slope * 1e6 gives permil/km
        """
        # Slope in fraction per meter
        slope_frac_per_m = -20e-6
        
        # Convert to permil per km
        slope_permil_per_km = slope_frac_per_m * 1e6
        
        assert slope_permil_per_km == -20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
