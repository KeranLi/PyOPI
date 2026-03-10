"""
One-Wind Model

Single moisture source model for orographic precipitation and isotopes.
Corresponds to MATLAB's opiCalc_OneWind and calc_OneWind functions.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from ..core.base_state import base_state
from ..core.precipitation import precipitation_grid
from ..core.isotopes import isotope_grid
from ..core.catchment import catchment_indices


@dataclass
class OneWindParameters:
    """Parameters for single-wind OPI model."""
    U: float          # Wind speed (m/s)
    azimuth: float    # Wind direction (degrees from North)
    T0: float         # Sea-level temperature (K)
    M: float          # Mountain-height number
    kappa: float      # Eddy diffusivity (m²/s)
    tau_c: float      # Cloud water residence time (s)
    d2H0: float       # Base d2H value (fraction)
    d_d2H0_d_lat: float  # d2H latitudinal gradient (fraction/degree)
    f_p0: float       # Residual precipitation fraction
    
    def to_array(self) -> np.ndarray:
        """Convert to parameter array."""
        return np.array([
            self.U, self.azimuth, self.T0, self.M, self.kappa,
            self.tau_c, self.d2H0, self.d_d2H0_d_lat, self.f_p0
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'OneWindParameters':
        """Create from parameter array."""
        return cls(*arr[:9])


@dataclass 
class OneWindResult:
    """Results from single-wind OPI calculation."""
    # Quality of fit
    chi_r2: float
    nu: int
    std_residuals: np.ndarray
    
    # Atmospheric profiles
    z_bar: np.ndarray
    T: np.ndarray
    gamma_env: np.ndarray
    gamma_sat: np.ndarray
    gamma_ratio: float
    rho_s0: float
    h_s: float
    rho0: float
    h_rho: float
    
    # Isotope parameters
    d18O0: float
    d_d18O0_d_lat: float
    tau_f: float
    
    # Grids
    p_grid: np.ndarray
    f_m_grid: np.ndarray
    r_h_grid: np.ndarray
    d2H_grid: np.ndarray
    d18O_grid: np.ndarray
    
    # Predictions
    i_wet: np.ndarray
    d2H_pred: np.ndarray
    d18O_pred: np.ndarray


class OneWindModel:
    """
    Single moisture source model for orographic precipitation and isotopes.
    
    This class provides a high-level interface for running OPI calculations
    with a single wind regime.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid' (elevation in meters)
    constants : dict, optional
        Model constants including 'f_c' (Coriolis), 'h_r' (exchange distance)
    
    Example
    -------
    >>> model = OneWindModel(topography={'x': x, 'y': y, 'h_grid': h})
    >>> params = OneWindParameters(U=10, azimuth=270, T0=288, M=0.8, ...)
    >>> result = model.run(params, samples)
    """
    
    def __init__(self, topography: Dict, constants: Optional[Dict] = None):
        self.x = topography['x']
        self.y = topography['y']
        self.h_grid = topography['h_grid']
        self.lat = topography.get('lat', None)
        self.lon = topography.get('lon', None)
        
        # Default constants
        self.f_c = constants.get('f_c', 1e-4) if constants else 1e-4
        self.h_r = constants.get('h_r', 540.0) if constants else 540.0
        self.sd_res_ratio = constants.get('sd_res_ratio', 28.3) if constants else 28.3
        
        # Derived quantities
        self.h_max = np.max(self.h_grid)
        self.dA = abs((self.x[1] - self.x[0]) * (self.y[1] - self.y[0]))
        
        # If no lat provided, create synthetic grid
        if self.lat is None:
            self.lat = np.zeros_like(self.h_grid)
            self.lat0 = 0.0
        else:
            self.lat0 = np.mean(self.lat)
    
    def run(self, 
            params: Union[OneWindParameters, np.ndarray],
            samples: Optional[Dict] = None,
            is_fit: bool = False) -> Union[OneWindResult, Dict]:
        """
        Run the single-wind OPI calculation.
        
        Parameters
        ----------
        params : OneWindParameters or ndarray
            Model parameters
        samples : dict, optional
            Sample data for comparison including:
            - 'x', 'y': sample coordinates
            - 'd2H', 'd18O': observed isotope values
            - 'type': 'L' for local or 'C' for catchment samples
        is_fit : bool
            If True, skip expensive diagnostic calculations
            
        Returns
        -------
        OneWindResult or dict
            Full result object or simplified dict if samples is None
        """
        # Convert array to parameter object if needed
        if isinstance(params, np.ndarray):
            params = OneWindParameters.from_array(params)
        
        # Unpack parameters
        U = params.U
        azimuth = params.azimuth
        T0 = params.T0
        M = params.M
        kappa = params.kappa
        tau_c = params.tau_c
        d2H0 = params.d2H0
        d_d2H0_d_lat = params.d_d2H0_d_lat
        f_p0 = params.f_p0
        
        # Convert mountain-height number to buoyancy frequency
        NM = M * U / self.h_max
        
        # Calculate base state
        base = base_state(NM, T0)
        
        # Calculate d18O from d2H using meteoric water line
        # Default MWL: d2H = 8*d18O + 10 (per mil)
        mwl_slope = 8.0
        mwl_intercept = 0.01  # 10 per mil in fraction
        d18O0 = (d2H0 - mwl_intercept) / mwl_slope
        d_d18O0_d_lat = d_d2H0_d_lat / mwl_slope
        
        # Calculate precipitation grid
        precip = precipitation_grid(
            self.x, self.y, self.h_grid, U, azimuth, NM, self.f_c,
            kappa, tau_c, base['h_rho'], base['z_bar'], base['T'],
            base['gamma_env'], base['gamma_sat'], base['gamma_ratio'],
            base['rho_s0'], base['h_s'], f_p0
        )
        
        # Calculate isotope grids
        iso = isotope_grid(
            precip['s'], precip['t'], precip['Sxy'], precip['Txy'],
            self.lat, self.lat0, precip['h_wind'], precip['f_m_wind'],
            precip['r_h_wind'], precip['f_p_wind'], precip['z223_wind'],
            precip['z258_wind'], precip['tau_f'], U, base['T'],
            base['gamma_sat'], base['h_s'], self.h_r, d2H0, d18O0,
            d_d2H0_d_lat, d_d18O0_d_lat, is_fit
        )
        
        # Transform moisture ratio to geographic grid
        from scipy.interpolate import RegularGridInterpolator
        F = RegularGridInterpolator(
            (precip['s'], precip['t']), precip['f_m_wind'],
            method='linear', bounds_error=False, fill_value=None
        )
        f_m_grid = F(np.column_stack([
            precip['Txy'].ravel(), precip['Sxy'].ravel()
        ])).reshape(precip['Sxy'].shape)
        
        # Transform relative humidity if needed
        if not is_fit and precip['r_h_wind'].size > 1:
            F = RegularGridInterpolator(
                (precip['s'], precip['t']), precip['r_h_wind'],
                method='linear', bounds_error=False, fill_value=None
            )
            r_h_grid = F(np.column_stack([
                precip['Txy'].ravel(), precip['Sxy'].ravel()
            ])).reshape(precip['Sxy'].shape)
        else:
            r_h_grid = np.ones_like(self.h_grid)
        
        # Process sample predictions if provided
        chi_r2 = np.nan
        nu = 0
        std_residuals = np.array([])
        i_wet = np.array([], dtype=bool)
        d2H_pred = np.array([])
        d18O_pred = np.array([])
        
        if samples is not None:
            d2H_pred, d18O_pred, i_wet = self._predict_samples(
                samples, precip['p_grid'], iso['d2H_grid'], iso['d18O_grid']
            )
            
            # Calculate statistics if observations provided
            if 'd2H' in samples and 'd18O' in samples:
                chi_r2, nu, std_residuals = self._calculate_fit_statistics(
                    samples['d2H'], samples['d18O'],
                    d2H_pred, d18O_pred, i_wet
                )
        
        return OneWindResult(
            chi_r2=chi_r2, nu=nu, std_residuals=std_residuals,
            z_bar=base['z_bar'], T=base['T'], gamma_env=base['gamma_env'],
            gamma_sat=base['gamma_sat'], gamma_ratio=base['gamma_ratio'],
            rho_s0=base['rho_s0'], h_s=base['h_s'],
            rho0=base['rho0'], h_rho=base['h_rho'],
            d18O0=d18O0, d_d18O0_d_lat=d_d18O0_d_lat, tau_f=precip['tau_f'],
            p_grid=precip['p_grid'], f_m_grid=f_m_grid, r_h_grid=r_h_grid,
            d2H_grid=iso['d2H_grid'], d18O_grid=iso['d18O_grid'],
            i_wet=i_wet, d2H_pred=d2H_pred, d18O_pred=d18O_pred
        )
    
    def _predict_samples(self, samples, p_grid, d2H_grid, d18O_grid):
        """Predict isotope values for sample locations."""
        sample_x = samples['x']
        sample_y = samples['y']
        sample_type = samples.get('type', ['L'] * len(sample_x))
        
        n_samples = len(sample_x)
        d2H_pred = np.full(n_samples, np.nan)
        d18O_pred = np.full(n_samples, np.nan)
        i_wet = np.zeros(n_samples, dtype=bool)
        
        # Find nearest grid nodes for each sample
        for k in range(n_samples):
            i = np.argmin(np.abs(self.y - sample_y[k]))
            j = np.argmin(np.abs(self.x - sample_x[k]))
            
            if sample_type[k] == 'L':
                # Local sample - use single grid node
                if p_grid[i, j] > 0:
                    i_wet[k] = True
                    d2H_pred[k] = d2H_grid[i, j]
                    d18O_pred[k] = d18O_grid[i, j]
            else:
                # Catchment sample - would need full catchment delineation
                # Simplified: use local value for now
                if p_grid[i, j] > 0:
                    i_wet[k] = True
                    d2H_pred[k] = d2H_grid[i, j]
                    d18O_pred[k] = d18O_grid[i, j]
        
        return d2H_pred, d18O_pred, i_wet
    
    def _calculate_fit_statistics(self, obs_d2H, obs_d18O, 
                                   pred_d2H, pred_d18O, i_wet):
        """Calculate chi-squared and other fit statistics."""
        n_wet = np.sum(i_wet)
        n_params = 9  # Number of free parameters
        nu = n_wet - n_params
        
        if nu > 0 and n_wet > 0:
            # Residuals for wet samples
            r_d2H = obs_d2H[i_wet] - pred_d2H[i_wet]
            r_d18O = obs_d18O[i_wet] - pred_d18O[i_wet]
            
            # Simple chi-squared (assuming uncorrelated, unit variance)
            chi_sq = np.sum(r_d2H**2) + np.sum(r_d18O**2)
            
            # Reduced chi-squared with t-distribution correction
            t_val = stats.t.ppf(0.84134, nu)
            chi_r2 = t_val * chi_sq / nu
            
            # Standardized residuals
            std_residuals = np.sqrt(r_d2H**2 + r_d18O**2)
        else:
            chi_r2 = np.nan
            std_residuals = np.full(len(obs_d2H), np.nan)
        
        return chi_r2, nu, std_residuals


def run_one_wind(topography: Dict, 
                 parameters: Union[Dict, np.ndarray, OneWindParameters],
                 samples: Optional[Dict] = None,
                 constants: Optional[Dict] = None) -> OneWindResult:
    """
    Convenience function to run a single-wind OPI calculation.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid' arrays
    parameters : dict, ndarray, or OneWindParameters
        Model parameters
    samples : dict, optional
        Sample data for comparison
    constants : dict, optional
        Model constants
        
    Returns
    -------
    OneWindResult
        Complete model results
    """
    # Convert dict parameters to object
    if isinstance(parameters, dict):
        parameters = OneWindParameters(**parameters)
    
    model = OneWindModel(topography, constants)
    return model.run(parameters, samples)
