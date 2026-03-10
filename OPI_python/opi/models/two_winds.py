"""
Two-Wind Model

Two moisture source model for orographic precipitation and isotopes.
Corresponds to MATLAB's opiCalc_TwoWinds and calc_TwoWinds functions.

This model combines two distinct wind regimes with a fractional weighting
to represent mixed moisture sources or seasonal wind patterns.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from ..core.base_state import base_state
from ..core.precipitation import precipitation_grid
from ..core.isotopes import isotope_grid


@dataclass
class TwoWindsParameters:
    """Parameters for two-wind OPI model."""
    # Wind regime 1
    U_1: float
    azimuth_1: float
    T0_1: float
    M_1: float
    kappa_1: float
    tau_c_1: float
    d2H0_1: float
    d_d2H0_d_lat_1: float
    f_p0_1: float
    
    # Wind regime 2
    U_2: float
    azimuth_2: float
    T0_2: float
    M_2: float
    kappa_2: float
    tau_c_2: float
    d2H0_2: float
    d_d2H0_d_lat_2: float
    f_p0_2: float
    
    # Fraction of precipitation from regime 1
    fraction_1: float
    
    def to_array(self) -> np.ndarray:
        """Convert to parameter array."""
        return np.array([
            self.U_1, self.azimuth_1, self.T0_1, self.M_1, self.kappa_1,
            self.tau_c_1, self.d2H0_1, self.d_d2H0_d_lat_1, self.f_p0_1,
            self.fraction_1,
            self.U_2, self.azimuth_2, self.T0_2, self.M_2, self.kappa_2,
            self.tau_c_2, self.d2H0_2, self.d_d2H0_d_lat_2, self.f_p0_2
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TwoWindsParameters':
        """Create from parameter array."""
        return cls(
            U_1=arr[0], azimuth_1=arr[1], T0_1=arr[2], M_1=arr[3],
            kappa_1=arr[4], tau_c_1=arr[5], d2H0_1=arr[6],
            d_d2H0_d_lat_1=arr[7], f_p0_1=arr[8],
            fraction_1=arr[9],
            U_2=arr[10], azimuth_2=arr[11], T0_2=arr[12], M_2=arr[13],
            kappa_2=arr[14], tau_c_2=arr[15], d2H0_2=arr[16],
            d_d2H0_d_lat_2=arr[17], f_p0_2=arr[18]
        )


@dataclass
class TwoWindsResult:
    """Results from two-wind OPI calculation."""
    # Quality of fit
    chi_r2: float
    nu: int
    std_residuals: np.ndarray
    
    # Individual wind regime results
    p_grid_1: np.ndarray
    p_grid_2: np.ndarray
    d2H_grid_1: np.ndarray
    d2H_grid_2: np.ndarray
    d18O_grid_1: np.ndarray
    d18O_grid_2: np.ndarray
    
    # Combined results
    p_grid: np.ndarray
    d2H_grid: np.ndarray
    d18O_grid: np.ndarray
    fraction_p_grid: np.ndarray  # Fraction from regime 1
    
    # Predictions
    i_wet: np.ndarray
    d2H_pred: np.ndarray
    d18O_pred: np.ndarray


class TwoWindModel:
    """
    Two moisture source model for orographic precipitation and isotopes.
    
    This model combines two distinct wind regimes to represent:
    - Mixed moisture sources
    - Seasonal wind patterns
    - Transitional wind regimes
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid' arrays
    constants : dict, optional
        Model constants
    """
    
    def __init__(self, topography: Dict, constants: Optional[Dict] = None):
        self.x = topography['x']
        self.y = topography['y']
        self.h_grid = topography['h_grid']
        self.lat = topography.get('lat', None)
        
        self.f_c = constants.get('f_c', 1e-4) if constants else 1e-4
        self.h_r = constants.get('h_r', 540.0) if constants else 540.0
        
        self.h_max = np.max(self.h_grid)
        
        if self.lat is None:
            self.lat = np.zeros_like(self.h_grid)
            self.lat0 = 0.0
        else:
            self.lat0 = np.mean(self.lat)
    
    def run(self,
            params: Union[TwoWindsParameters, np.ndarray],
            samples: Optional[Dict] = None,
            is_fit: bool = False) -> TwoWindsResult:
        """
        Run the two-wind OPI calculation.
        
        Parameters
        ----------
        params : TwoWindsParameters or ndarray
            Model parameters
        samples : dict, optional
            Sample data
        is_fit : bool
            If True, skip diagnostic calculations
            
        Returns
        -------
        TwoWindsResult
            Complete model results
        """
        if isinstance(params, np.ndarray):
            params = TwoWindsParameters.from_array(params)
        
        # Run first wind regime
        result_1 = self._run_regime(
            params.U_1, params.azimuth_1, params.T0_1, params.M_1,
            params.kappa_1, params.tau_c_1, params.d2H0_1,
            params.d_d2H0_d_lat_1, params.f_p0_1, is_fit
        )
        
        # Run second wind regime
        result_2 = self._run_regime(
            params.U_2, params.azimuth_2, params.T0_2, params.M_2,
            params.kappa_2, params.tau_c_2, params.d2H0_2,
            params.d_d2H0_d_lat_2, params.f_p0_2, is_fit
        )
        
        # Scale precipitation grids
        p_grid_1 = params.fraction_1 * result_1['p_grid']
        p_grid_2 = (1 - params.fraction_1) * result_2['p_grid']
        
        # Combine precipitation
        p_grid = p_grid_1 + p_grid_2
        
        # Combine isotope grids (precipitation-weighted)
        with np.errstate(divide='ignore', invalid='ignore'):
            d2H_grid = np.where(
                p_grid > 0,
                (p_grid_1 * result_1['d2H_grid'] + 
                 p_grid_2 * result_2['d2H_grid']) / p_grid,
                np.nan
            )
            d18O_grid = np.where(
                p_grid > 0,
                (p_grid_1 * result_1['d18O_grid'] + 
                 p_grid_2 * result_2['d18O_grid']) / p_grid,
                np.nan
            )
        
        # Fraction from regime 1
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction_p_grid = np.where(
                p_grid > 0, p_grid_1 / p_grid, np.nan
            )
        
        # Process samples
        chi_r2 = np.nan
        nu = 0
        std_residuals = np.array([])
        i_wet = np.array([], dtype=bool)
        d2H_pred = np.array([])
        d18O_pred = np.array([])
        
        if samples is not None:
            d2H_pred, d18O_pred, i_wet = self._predict_samples(
                samples, p_grid, d2H_grid, d18O_grid
            )
            
            if 'd2H' in samples and 'd18O' in samples:
                chi_r2, nu, std_residuals = self._calculate_fit_statistics(
                    samples['d2H'], samples['d18O'],
                    d2H_pred, d18O_pred, i_wet, 19  # 19 free parameters
                )
        
        return TwoWindsResult(
            chi_r2=chi_r2, nu=nu, std_residuals=std_residuals,
            p_grid_1=p_grid_1, p_grid_2=p_grid_2,
            d2H_grid_1=result_1['d2H_grid'], d2H_grid_2=result_2['d2H_grid'],
            d18O_grid_1=result_1['d18O_grid'], d18O_grid_2=result_2['d18O_grid'],
            p_grid=p_grid, d2H_grid=d2H_grid, d18O_grid=d18O_grid,
            fraction_p_grid=fraction_p_grid,
            i_wet=i_wet, d2H_pred=d2H_pred, d18O_pred=d18O_pred
        )
    
    def _run_regime(self, U, azimuth, T0, M, kappa, tau_c, d2H0, 
                    d_d2H0_d_lat, f_p0, is_fit):
        """Run calculation for a single wind regime."""
        NM = M * U / self.h_max
        
        base = base_state(NM, T0)
        
        mwl_slope = 8.0
        mwl_intercept = 0.01
        d18O0 = (d2H0 - mwl_intercept) / mwl_slope
        d_d18O0_d_lat = d_d2H0_d_lat / mwl_slope
        
        precip = precipitation_grid(
            self.x, self.y, self.h_grid, U, azimuth, NM, self.f_c,
            kappa, tau_c, base['h_rho'], base['z_bar'], base['T'],
            base['gamma_env'], base['gamma_sat'], base['gamma_ratio'],
            base['rho_s0'], base['h_s'], f_p0
        )
        
        iso = isotope_grid(
            precip['s'], precip['t'], precip['Sxy'], precip['Txy'],
            self.lat, self.lat0, precip['h_wind'], precip['f_m_wind'],
            precip['r_h_wind'], precip['f_p_wind'], precip['z223_wind'],
            precip['z258_wind'], precip['tau_f'], U, base['T'],
            base['gamma_sat'], base['h_s'], self.h_r, d2H0, d18O0,
            d_d2H0_d_lat, d_d18O0_d_lat, is_fit
        )
        
        return {
            'p_grid': precip['p_grid'],
            'd2H_grid': iso['d2H_grid'],
            'd18O_grid': iso['d18O_grid']
        }
    
    def _predict_samples(self, samples, p_grid, d2H_grid, d18O_grid):
        """Predict isotope values for samples."""
        sample_x = samples['x']
        sample_y = samples['y']
        n_samples = len(sample_x)
        
        d2H_pred = np.full(n_samples, np.nan)
        d18O_pred = np.full(n_samples, np.nan)
        i_wet = np.zeros(n_samples, dtype=bool)
        
        for k in range(n_samples):
            i = np.argmin(np.abs(self.y - sample_y[k]))
            j = np.argmin(np.abs(self.x - sample_x[k]))
            
            if p_grid[i, j] > 0:
                i_wet[k] = True
                d2H_pred[k] = d2H_grid[i, j]
                d18O_pred[k] = d18O_grid[i, j]
        
        return d2H_pred, d18O_pred, i_wet
    
    def _calculate_fit_statistics(self, obs_d2H, obs_d18O, pred_d2H, 
                                   pred_d18O, i_wet, n_params):
        """Calculate fit statistics."""
        n_wet = np.sum(i_wet)
        nu = n_wet - n_params
        
        if nu > 0 and n_wet > 0:
            r_d2H = obs_d2H[i_wet] - pred_d2H[i_wet]
            r_d18O = obs_d18O[i_wet] - pred_d18O[i_wet]
            chi_sq = np.sum(r_d2H**2) + np.sum(r_d18O**2)
            t_val = stats.t.ppf(0.84134, nu)
            chi_r2 = t_val * chi_sq / nu
            std_residuals = np.sqrt(r_d2H**2 + r_d18O**2)
        else:
            chi_r2 = np.nan
            std_residuals = np.full(len(obs_d2H), np.nan)
        
        return chi_r2, nu, std_residuals


def run_two_winds(topography: Dict,
                  parameters: Union[Dict, np.ndarray, TwoWindsParameters],
                  samples: Optional[Dict] = None,
                  constants: Optional[Dict] = None) -> TwoWindsResult:
    """
    Convenience function to run a two-wind OPI calculation.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid'
    parameters : dict, ndarray, or TwoWindsParameters
        Model parameters
    samples : dict, optional
        Sample data
    constants : dict, optional
        Model constants
        
    Returns
    -------
    TwoWindsResult
        Complete model results
    """
    if isinstance(parameters, dict):
        parameters = TwoWindsParameters(**parameters)
    
    model = TwoWindModel(topography, constants)
    return model.run(parameters, samples)
