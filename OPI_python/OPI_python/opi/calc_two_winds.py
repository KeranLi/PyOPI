"""
Two-wind calculation function for OPI model

This module implements the core calculation for the two-wind model,
combining results from two distinct moisture sources.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .base_state import base_state
from .precipitation_grid import precipitation_grid
from .isotope_grid import isotope_grid
from .catchment_indices import catchment_indices


def calc_two_winds(beta, f_c, h_r, x, y, lat, lat0, h_grid, b_mwl_sample,
                   ij_catch, ptr_catch, sample_d2h, sample_d18o, cov,
                   n_parameters_free, is_fit):
    """
    Calculate meteoric and isotope values for a specified solution vector, beta,
    for two moisture sources (TwoWinds).
    
    Parameters:
    -----------
    beta : array-like
        Solution vector with 19 parameters
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
    tuple
        All calculation results for both wind fields and combined values
    """
    
    # Unpack solution vector
    # Wind 1 parameters
    U_1 = beta[0]
    azimuth_1 = beta[1]
    T0_1 = beta[2]
    M_1 = beta[3]
    kappa_1 = beta[4]
    tau_c_1 = beta[5]
    d2h0_1 = beta[6]
    d_d2h0_d_lat_1 = beta[7]
    f_p0_1 = beta[8]
    fraction = beta[9]  # Fraction of wind 1
    
    # Wind 2 parameters
    U_2 = beta[10]
    azimuth_2 = beta[11]
    T0_2 = beta[12]
    M_2 = beta[13]
    kappa_2 = beta[14]
    tau_c_2 = beta[15]
    d2h0_2 = beta[16]
    d_d2h0_d_lat_2 = beta[17]
    f_p0_2 = beta[18]
    
    # Convert mountain-height number to buoyancy frequency
    h_max = np.max(h_grid)
    NM_1 = M_1 * U_1 / h_max
    NM_2 = M_2 * U_2 / h_max
    
    # Number of samples
    n_samples = 0
    if ij_catch and ptr_catch:
        n_samples = len(ptr_catch) - 1
    
    # ========================================
    # Calculate precipitation state 1
    # ========================================
    z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1, \
        rho_s0_1, h_s_1, rho0_1, h_rho_1 = base_state(NM_1, T0_1)
    
    # Set d18O isotopic composition for base precipitation
    d18o0_1 = (d2h0_1 - b_mwl_sample[0]) / b_mwl_sample[1]
    d_d18o0_d_lat_1 = d_d2h0_d_lat_1 / b_mwl_sample[1]
    
    # Calculate precipitation rate and related grids
    precip_result_1 = precipitation_grid(
        x, y, h_grid, U_1, azimuth_1, NM_1, f_c, kappa_1, tau_c_1, h_rho_1,
        z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1, rho_s0_1, h_s_1, f_p0_1
    )
    
    s = precip_result_1['s']
    t = precip_result_1['t']
    s_xy = precip_result_1['Sxy']
    t_xy = precip_result_1['Txy']
    p_grid_1 = precip_result_1['p_grid'] * fraction  # Scale by fraction
    h_wind = precip_result_1['h_wind']
    f_m_wind_1 = precip_result_1['f_m_wind']
    r_h_wind_1 = precip_result_1['r_h_wind']
    f_p_wind_1 = precip_result_1['f_p_wind']
    z223_wind_1 = precip_result_1['z223_wind']
    z258_wind_1 = precip_result_1['z258_wind']
    tau_f_1 = precip_result_1['tau_f']
    
    # Calculate isotope grids
    iso_result_1 = isotope_grid(
        s, t, s_xy, t_xy, lat, lat0, h_wind, f_m_wind_1, r_h_wind_1, f_p_wind_1,
        z223_wind_1, z258_wind_1, tau_f_1,
        U_1, T_1, gamma_sat_1, h_s_1, h_r, d2h0_1, d18o0_1, 
        d_d2h0_d_lat_1, d_d18o0_d_lat_1, is_fit
    )
    
    d2h_grid_1 = iso_result_1['d2h_grid']
    d18o_grid_1 = iso_result_1['d18o_grid']
    
    del f_p_wind_1
    
    # Interpolate f_m_wind back to geographic grid
    interpolator = RegularGridInterpolator((s, t), f_m_wind_1, method='linear', 
                                           bounds_error=False, fill_value=None)
    f_m_grid_1 = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
    del f_m_wind_1
    
    # Calculate r_h_grid
    if not is_fit:
        if r_h_wind_1.size > 1:
            interpolator = RegularGridInterpolator((s, t), r_h_wind_1, method='linear',
                                                   bounds_error=False, fill_value=None)
            r_h_grid_1 = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
            del r_h_wind_1
        else:
            r_h_grid_1 = np.ones_like(h_grid)
    else:
        r_h_grid_1 = None
    
    # Predictions for samples (state 1)
    d2h_pred_1 = np.zeros(n_samples)
    d18o_pred_1 = np.zeros(n_samples)
    p_sum_pred_1 = np.zeros(n_samples)
    
    for k in range(n_samples):
        if k < len(ptr_catch) - 1:
            ij = ij_catch[ptr_catch[k]:ptr_catch[k+1]]
        else:
            ij = ij_catch[ptr_catch[k]:]
        
        p_sum_pred_1[k] = np.sum([p_grid_1[idx[0], idx[1]] for idx in ij])
        
        if p_sum_pred_1[k] > 0:
            wt_prec = np.array([p_grid_1[idx[0], idx[1]]/p_sum_pred_1[k] for idx in ij])
            d2h_pred_1[k] = sum(wt_prec[i] * d2h_grid_1[ij[i][0], ij[i][1]] for i in range(len(ij)))
            d18o_pred_1[k] = sum(wt_prec[i] * d18o_grid_1[ij[i][0], ij[i][1]] for i in range(len(ij)))
    
    # ========================================
    # Calculate precipitation state 2
    # ========================================
    z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2, \
        rho_s0_2, h_s_2, rho0_2, h_rho_2 = base_state(NM_2, T0_2)
    
    # Set d18O isotopic composition
    d18o0_2 = (d2h0_2 - b_mwl_sample[0]) / b_mwl_sample[1]
    d_d18o0_d_lat_2 = d_d2h0_d_lat_2 / b_mwl_sample[1]
    
    # Calculate precipitation rate
    precip_result_2 = precipitation_grid(
        x, y, h_grid, U_2, azimuth_2, NM_2, f_c, kappa_2, tau_c_2, h_rho_2,
        z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2, rho_s0_2, h_s_2, f_p0_2
    )
    
    p_grid_2 = precip_result_2['p_grid'] * (1 - fraction)  # Scale by (1 - fraction)
    h_wind_2 = precip_result_2['h_wind']
    f_m_wind_2 = precip_result_2['f_m_wind']
    r_h_wind_2 = precip_result_2['r_h_wind']
    f_p_wind_2 = precip_result_2['f_p_wind']
    z223_wind_2 = precip_result_2['z223_wind']
    z258_wind_2 = precip_result_2['z258_wind']
    tau_f_2 = precip_result_2['tau_f']
    
    # Calculate isotope grids
    iso_result_2 = isotope_grid(
        s, t, s_xy, t_xy, lat, lat0, h_wind_2, f_m_wind_2, r_h_wind_2, f_p_wind_2,
        z223_wind_2, z258_wind_2, tau_f_2,
        U_2, T_2, gamma_sat_2, h_s_2, h_r, d2h0_2, d18o0_2,
        d_d2h0_d_lat_2, d_d18o0_d_lat_2, is_fit
    )
    
    d2h_grid_2 = iso_result_2['d2h_grid']
    d18o_grid_2 = iso_result_2['d18o_grid']
    
    del f_p_wind_2
    
    # Interpolate f_m_wind back to geographic grid
    interpolator = RegularGridInterpolator((s, t), f_m_wind_2, method='linear',
                                           bounds_error=False, fill_value=None)
    f_m_grid_2 = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
    del f_m_wind_2
    
    # Calculate r_h_grid
    if not is_fit:
        if r_h_wind_2.size > 1:
            interpolator = RegularGridInterpolator((s, t), r_h_wind_2, method='linear',
                                                   bounds_error=False, fill_value=None)
            r_h_grid_2 = np.reshape(interpolator((s_xy.flatten(), t_xy.flatten())), s_xy.shape)
            del r_h_wind_2
        else:
            r_h_grid_2 = np.ones_like(h_grid)
    else:
        r_h_grid_2 = None
    
    # Predictions for samples (state 2)
    d2h_pred_2 = np.zeros(n_samples)
    d18o_pred_2 = np.zeros(n_samples)
    p_sum_pred_2 = np.zeros(n_samples)
    
    for k in range(n_samples):
        if k < len(ptr_catch) - 1:
            ij = ij_catch[ptr_catch[k]:ptr_catch[k+1]]
        else:
            ij = ij_catch[ptr_catch[k]:]
        
        p_sum_pred_2[k] = np.sum([p_grid_2[idx[0], idx[1]] for idx in ij])
        
        if p_sum_pred_2[k] > 0:
            wt_prec = np.array([p_grid_2[idx[0], idx[1]]/p_sum_pred_2[k] for idx in ij])
            d2h_pred_2[k] = sum(wt_prec[i] * d2h_grid_2[ij[i][0], ij[i][1]] for i in range(len(ij)))
            d18o_pred_2[k] = sum(wt_prec[i] * d18o_grid_2[ij[i][0], ij[i][1]] for i in range(len(ij)))
    
    # ========================================
    # Combine precipitation states
    # ========================================
    p_grid = p_grid_1 + p_grid_2
    
    # Calculate combined isotope grids (precipitation-weighted)
    with np.errstate(divide='ignore', invalid='ignore'):
        d2h_grid = (p_grid_1 * d2h_grid_1 + p_grid_2 * d2h_grid_2) / p_grid
        d18o_grid = (p_grid_1 * d18o_grid_1 + p_grid_2 * d18o_grid_2) / p_grid
    
    # Set NaN where no precipitation
    d2h_grid = np.where(p_grid > 0, d2h_grid, np.nan)
    d18o_grid = np.where(p_grid > 0, d18o_grid, np.nan)
    
    # Grid of fraction from state 1
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction_p_grid = p_grid_1 / p_grid
        fraction_p_grid = np.where(p_grid > 0, fraction_p_grid, np.nan)
    
    # Combined predictions for samples
    p_sum_pred = p_sum_pred_1 + p_sum_pred_2
    i_wet = p_sum_pred > 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d2h_pred = (p_sum_pred_1 * d2h_pred_1 + p_sum_pred_2 * d2h_pred_2) / p_sum_pred
        d18o_pred = (p_sum_pred_1 * d18o_pred_1 + p_sum_pred_2 * d18o_pred_2) / p_sum_pred
    
    d2h_pred = np.where(i_wet, d2h_pred, np.nan)
    d18o_pred = np.where(i_wet, d18o_pred, np.nan)
    d2h_pred_1 = np.where(p_sum_pred_1 > 0, d2h_pred_1, np.nan)
    d18o_pred_1 = np.where(p_sum_pred_1 > 0, d18o_pred_1, np.nan)
    d2h_pred_2 = np.where(p_sum_pred_2 > 0, d2h_pred_2, np.nan)
    d18o_pred_2 = np.where(p_sum_pred_2 > 0, d18o_pred_2, np.nan)
    
    # ========================================
    # Calculate chi-square
    # ========================================
    if np.all(np.isnan(sample_d18o)) or np.all(np.isnan(sample_d2h)):
        chi_r2 = np.nan
        nu = 0
        std_residuals = np.full(n_samples, np.nan)
    else:
        n_samples_wet = np.sum(i_wet)
        nu = n_samples_wet - n_parameters_free
        
        if nu > 0:
            # Calculate residuals for wet samples
            r = np.vstack([
                (sample_d2h[i_wet] - d2h_pred[i_wet]),
                (sample_d18o[i_wet] - d18o_pred[i_wet])
            ])
            
            # Reduced chi-square
            from scipy.stats import t
            t_val = t.ppf(0.84134, nu)
            chi_r2 = t_val * np.sum(np.dot(r.T, np.linalg.solve(cov, r))) / nu
            
            # Standardized residuals for all samples
            r_all = np.vstack([
                (sample_d2h - d2h_pred),
                (sample_d18o - d18o_pred)
            ])
            std_residuals = np.sqrt(np.sum(r_all * np.linalg.solve(cov, r_all), axis=0))
        else:
            chi_r2 = np.nan
            std_residuals = np.full(n_samples, np.nan)
    
    # Return all results
    return (chi_r2, nu, std_residuals,
            z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1,
            rho_s0_1, h_s_1, rho0_1, h_rho_1,
            d18o0_1, d_d18o0_d_lat_1, tau_f_1, p_grid_1, f_m_grid_1, r_h_grid_1,
            d2h_grid_1, d18o_grid_1, p_sum_pred_1, d2h_pred_1, d18o_pred_1,
            z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2,
            rho_s0_2, h_s_2, rho0_2, h_rho_2,
            d18o0_2, d_d18o0_d_lat_2, tau_f_2, p_grid_2, f_m_grid_2, r_h_grid_2,
            d2h_grid_2, d18o_grid_2, p_sum_pred_2, d2h_pred_2, d18o_pred_2,
            p_grid, d2h_grid, d18o_grid, p_sum_pred, d2h_pred, d18o_pred,
            fraction_p_grid)
