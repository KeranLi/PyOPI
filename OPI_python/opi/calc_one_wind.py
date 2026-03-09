"""
One-wind calculation function for OPI model
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .base_state import base_state
from .constants import G, L, RD, EPSILON
from .precipitation_grid import precipitation_grid
from .isotope_grid import isotope_grid
from .catchment_indices import catchment_indices


def calc_one_wind(beta, f_c, h_r, x, y, lat, lat0, h_grid, b_mwl_sample, 
                  ij_catch, ptr_catch, sample_d2h, sample_d18o, cov, 
                  n_parameters_free, is_fit):
    """
    Calculate meteoric and isotope values for a specified solution vector, beta,
    for one moisture source (OneWind).
    
    Parameters:
    -----------
    beta : array-like
        Solution vector with 9 parameters
    f_c : float
        Coriolis parameter
    h_r : float
        Characteristic distance for isotopic exchange
    x, y : array-like
        Grid coordinates
    lat : array-like
        Latitudes
    lat0 : float
        Reference latitude
    h_grid : 2D array
        Elevation grid
    b_mwl_sample : array-like
        MWL parameters for samples
    ij_catch, ptr_catch : list
        Catchment indices
    sample_d2h, sample_d18o : array-like
        Sample isotope values
    cov : 2D array
        Covariance matrix
    n_parameters_free : int
        Number of free parameters
    is_fit : bool
        Whether this is part of a fitting procedure
    
    Returns:
    --------
    chi_r2 : float
        Reduced chi-square
    nu : int
        Degrees of freedom
    std_residuals : array
        Standardized residuals
    z_bar, T, gamma_env, gamma_sat, gamma_ratio : array/float
        Atmospheric profile parameters
    rho_s0, h_s, rho0, h_rho : float
        Density parameters
    d18o0, d_d18o0_d_lat : float
        Isotope parameters
    tau_f : float
        Residence time for falling precipitation
    p_grid : 2D array
        Precipitation grid
    f_m_grid : 2D array
        Moisture grid
    r_h_grid : 2D array
        Relative humidity grid
    evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid : 2D array
        Isotope grids
    d2h_grid, d18o_grid : 2D array
        Final isotope grids
    i_wet : array
        Boolean indicating wet locations
    d2h_pred, d18o_pred : array
        Predicted isotope values
    """
    
    # Initialize outputs
    chi_r2 = []
    nu = []
    std_residuals = []
    i_wet = []
    d2h_pred = []
    d18o_pred = []

    # Unpack solution vector
    U = beta[0]           # wind speed (m/s)
    azimuth = beta[1]     # azimuth (degrees)
    T0 = beta[2]          # sea-level temperature (K)
    M = beta[3]           # mountain-height number (dimensionless)
    kappa = beta[4]       # eddy diffusion (m/s^2)
    tau_c = beta[5]       # condensation time (s)
    d2h0 = beta[6]        # d2H of base precipitation (per unit)
    d_d2h0_d_lat = beta[7]  # latitudinal gradient of base-prec d2H (1/deg lat)
    f_p0 = beta[8]        # residual precipitation after evaporation (fraction)

    # Convert mountain-height number to buoyancy frequency (rad/s)
    h_max = np.max(h_grid)
    NM = M * U / h_max

    # Number of locations for predictions
    n_samples = 0
    if ij_catch and ptr_catch:
        n_samples = len(ptr_catch) - 1

    # Calculate environmental profile for atmosphere upwind of topography
    z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)

    # Set d18O isotopic composition for base precipitation
    d18o0 = (d2h0 - b_mwl_sample[0]) / b_mwl_sample[1]
    d_d18o0_d_lat = d_d2h0_d_lat / b_mwl_sample[1]

    # Calculate precipitation rate and related grids
    precip_result = precipitation_grid(
        x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c, h_rho, 
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, f_p0
    )
    
    s = precip_result['s']
    t = precip_result['t']
    s_xy = precip_result['Sxy']
    t_xy = precip_result['Txy']
    p_grid = precip_result['p_grid']
    h_wind = precip_result['h_wind']
    f_m_wind = precip_result['f_m_wind']
    r_h_wind = precip_result['r_h_wind']
    f_p_wind = precip_result['f_p_wind']
    z223_wind = precip_result['z223_wind']
    z258_wind = precip_result['z258_wind']
    tau_f = precip_result['tau_f']

    # Calculate grids for precipitation isotopes and moisture ratio
    iso_result = isotope_grid(
        s, t, s_xy, t_xy, lat, lat0, h_wind, f_m_wind, r_h_wind, f_p_wind,
        z223_wind, z258_wind, tau_f,
        U, T, gamma_sat, h_s, h_r, d2h0, d18o0, d_d2h0_d_lat, d_d18o0_d_lat, is_fit
    )
    
    d2h_grid = iso_result['d2h_grid']
    d18o_grid = iso_result['d18o_grid']
    evap_d2h_grid = iso_result['evap_d2h_grid']
    u_evap_d2h_grid = iso_result['u_evap_d2h_grid']
    evap_d18o_grid = iso_result['evap_d18o_grid']
    u_evap_d18o_grid = iso_result['u_evap_d18o_grid']

    # Clear temporary variables
    del h_wind, f_p_wind

    # Interpolate f_m_wind back to geographic grid
    interpolator = RegularGridInterpolator((s, t), f_m_wind, method='linear', bounds_error=False, fill_value=None)
    f_m_grid = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
    del f_m_wind

    # Calculate r_h_grid if needed
    if not is_fit:
        if r_h_wind.size > 1:
            # Using evaporation recycling, so calculate r_h_grid
            interpolator = RegularGridInterpolator((s, t), r_h_wind, method='linear', bounds_error=False, fill_value=None)
            r_h_grid = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
            del r_h_wind
        else:
            # No evaporation, so r_h_grid = 1
            r_h_grid = np.ones_like(h_grid)
            del r_h_wind
    else:
        r_h_grid = []

    # Calculations for sample locations
    if ij_catch and ptr_catch:
        d2h_pred = np.full(n_samples, np.nan)
        d18o_pred = np.full(n_samples, np.nan)
        i_wet = np.zeros(n_samples, dtype=bool)

        for k in range(n_samples):
            # Extract indices for sample catchment
            ij = catchment_indices(k, ij_catch, ptr_catch)
            
            # Calculate weights using predicted precipitation
            if len(ij) > 0:
                p_sum = np.sum([p_grid[idx[0], idx[1]] for idx in ij])
            
            # Calculate precipitation-weighted isotope compositions for catchment
            if p_sum > 0:
                # Normalize weights
                wt_prec = np.array([p_grid[idx[0], idx[1]]/p_sum for idx in ij])
                
                # Calculate catchment-weighted composition of water isotopes
                i_wet[k] = True
                d2h_pred[k] = sum(wt_prec[i] * d2h_grid[ij[i][0], ij[i][1]] for i in range(len(ij)))
                d18o_pred[k] = sum(wt_prec[i] * d18o_grid[ij[i][0], ij[i][1]] for i in range(len(ij)))
            else:
                # Use simple mean for dry sites
                d2h_pred[k] = np.mean([d2h_grid[idx[0], idx[1]] for idx in ij])
                d18o_pred[k] = np.mean([d18o_grid[idx[0], idx[1]] for idx in ij])

    # Skip chiR2 calculation for simulation case
    if np.all(np.isnan(np.concatenate([sample_d18o, sample_d2h])) if sample_d18o.size > 0 and sample_d2h.size > 0 else True):
        return chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio, \
               rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f, p_grid, f_m_grid, r_h_grid, \
               evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid, \
               d2h_grid, d18o_grid, i_wet, d2h_pred, d18o_pred

    # Calculate reduced chi square and standardized residuals
    n_samples_wet = np.sum(i_wet)
    nu = n_samples_wet - n_parameters_free

    if nu > 0:
        # Calculate chiR2 using samples that come from wet locations
        r = np.vstack([
            (sample_d2h[i_wet] - d2h_pred[i_wet]),
            (sample_d18o[i_wet] - d18o_pred[i_wet])
        ])
        
        # Reduced chi square, with correction for small-sample bias
        from scipy.stats import t
        t_val = t.ppf(0.84134, nu)  # t value for probability at +1 sigma for normal distribution
        chi_r2 = t_val * np.sum(np.dot(r, np.linalg.solve(cov, r))) / nu
        
        # Calculate standardized residuals for all samples
        r_all = np.vstack([
            (sample_d2h - d2h_pred),
            (sample_d18o - d18o_pred)
        ])
        std_residuals = np.sqrt(np.dot(r_all, np.linalg.solve(cov, r_all)))
    else:
        # Set to nan for cases where degrees of freedom < 1
        chi_r2 = np.nan
        std_residuals = np.full(n_samples, np.nan)

    return chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio, \
           rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f, p_grid, f_m_grid, r_h_grid, \
           evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid, \
           d2h_grid, d18o_grid, i_wet, d2h_pred, d18o_pred


def catchment_indices_python(sample_idx, ij_catch, ptr_catch):
    """
    Helper function to extract catchment indices for a specific sample
    """
    start_idx = ptr_catch[sample_idx]
    end_idx = ptr_catch[sample_idx + 1]
    return ij_catch[start_idx:end_idx]