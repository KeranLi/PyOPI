"""
Isotope Grid Calculations

This module calculates precipitation isotope grids (d2H and d18O) including:
1. Fractionation during precipitation formation and fall
2. Evaporative recycling effects
3. Integration along wind paths

The calculation accounts for the Bergeron-Findeisen process (mixed-phase 
precipitation) using the WBF (Wegener-Bergeron-Findeisen) zone defined 
by the 268-248 K temperature range.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .fractionation import (
    fractionation_hydrogen, 
    fractionation_oxygen,
    kinetic_fractionation_hydrogen,
    kinetic_fractionation_oxygen
)


def isotope_grid(s, t, Sxy, Txy, lat, lat0, h_wind, f_m_wind, r_h_wind, 
                 f_p_wind, z223_wind, z258_wind, tau_f, U, T, gamma_sat, 
                 h_s, h_r, d2H0, d18O0, d_d2H0_d_lat, d_d18O0_d_lat, is_fit):
    """
    Calculate precipitation isotope grids.
    
    Computes d2H and d18O values for precipitation accounting for:
    - Equilibrium fractionation during condensation
    - Mixed-phase precipitation (ice/water)
    - Evaporative recycling
    - Latitudinal gradients
    
    Parameters
    ----------
    s, t : ndarray
        Wind grid coordinate vectors
    Sxy, Txy : ndarray
        Geographic coordinates in wind system
    lat : ndarray
        Latitude vector or grid
    lat0 : float
        Reference latitude
    h_wind : ndarray
        Topography in wind coordinates (m)
    f_m_wind : ndarray
        Moisture ratio field
    r_h_wind : ndarray
        Relative humidity field (0-1)
    f_p_wind : ndarray
        Residual precipitation fraction
    z223_wind, z258_wind : ndarray
        Heights of 223K and 258K isotherms (m)
    tau_f : float
        Precipitation fall time (s)
    U : float
        Wind speed (m/s)
    T : ndarray
        Temperature profile (K)
    gamma_sat : ndarray
        Saturation lapse rate (K/m)
    h_s : float
        Water vapor scale height (m)
    h_r : float
        Isotope exchange distance (m)
    d2H0, d18O0 : float
        Base isotope values (fraction, not per mil)
    d_d2H0_d_lat, d_d18O0_d_lat : float
        Latitudinal gradients (fraction per degree)
    is_fit : bool
        If True, skip calculation of evaporation grids (for fitting)
        
    Returns
    -------
    dict : Dictionary containing:
        - 'd2H_grid', 'd18O_grid': Isotope grids in geographic coordinates
        - 'evap_d2H_grid', 'u_evap_d2H_grid': Evaporation grids for d2H
        - 'evap_d18O_grid', 'u_evap_d18O_grid': Evaporation grids for d18O
        
    References
    ----------
    Ciais and Jouzel (1994) - Deuterium and oxygen 18 in precipitation:
        Isotopic model, including mixed cloud processes
    Criss (1999) - Principles of Stable Isotope Distribution
    """
    # Shear for transforming fall paths to vertical
    shear = U * tau_f / h_s
    
    # Diffusivity ratios from Merlivat (1978)
    D_ratio_2H = 0.9755
    D_ratio_18O = 0.9723
    n_exp = 1.0  # Exponent for evaporation regime
    
    # Grid parameters
    d_s = s[1] - s[0]
    n_s, n_t = h_wind.shape
    
    # ============================================================
    # Set up wind-direction grid with shear transformation
    # ============================================================
    
    # Surface shear coordinates
    s_surf_shear = s.reshape(-1, 1) + shear * h_wind
    
    # Temperature at land surface
    T_ls = T[0] - gamma_sat[0] * h_wind
    
    # Transform 223K isotherm to surface-relative coordinates
    s_shear = s.reshape(-1, 1) + shear * z223_wind
    for j in range(n_t):
        col = s_shear[:, j]
        cummax_vals = np.maximum.accumulate(col)
        i_mono = col >= np.concatenate([[col[0] - 1], cummax_vals[:-1]])
        
        if np.any(i_mono):
            z223_wind[:, j] = np.interp(
                s_surf_shear[:, j], col[i_mono], z223_wind[i_mono, j],
                left=z223_wind[0, j], right=z223_wind[0, j]
            )
    
    z_bar_223 = z223_wind - h_wind
    del z223_wind
    
    # Transform 258K isotherm
    s_shear = s.reshape(-1, 1) + shear * z258_wind
    for j in range(n_t):
        col = s_shear[:, j]
        cummax_vals = np.maximum.accumulate(col)
        i_mono = col >= np.concatenate([[col[0] - 1], cummax_vals[:-1]])
        
        if np.any(i_mono):
            z258_wind[:, j] = np.interp(
                s_surf_shear[:, j], col[i_mono], z258_wind[i_mono, j],
                left=z258_wind[0, j], right=z258_wind[0, j]
            )
    
    z_bar_258 = z258_wind - h_wind
    del z258_wind, s_shear
    
    # Freezing surface height above land surface
    z_bar_fs = z_bar_258.copy()
    z_bar_fs[z_bar_fs < 0] = 0
    
    # Log derivative of moisture ratio
    d_ln_fm_ds = np.gradient(np.log(f_m_wind), d_s, axis=0)
    
    # ============================================================
    # Calculate d2H grid
    # ============================================================
    
    # Fractionation factors
    a_ls = fractionation_hydrogen(T_ls)
    a_258 = fractionation_hydrogen(258.0)
    a_223 = fractionation_hydrogen(223.0)
    
    # Vertical averaging coefficients
    b_a = (a_223 - a_258) / (z_bar_223 - z_bar_258)
    b_a[np.isnan(b_a)] = 0
    b_b = (a_258 - a_ls) / z_bar_258
    b_b[np.isnan(b_b)] = 0
    
    # Precipitation fractionation factor (R_prec/R_vapor)
    a_prec = (
        ((a_258 + b_a * (h_s + z_bar_fs - z_bar_258)) * np.exp(-z_bar_fs / h_r)
         + a_ls * (1 - np.exp(-z_bar_fs / h_r))) * np.exp(-z_bar_fs / h_s)
        + b_b * (h_r**2 * h_s / (h_s + h_r)**2)
        * (1 - (1 + (1/h_s + 1/h_r) * z_bar_fs)
           * np.exp(-(1/h_s + 1/h_r) * z_bar_fs))
        + a_ls * (1 - np.exp(-z_bar_fs / h_s))
    )
    del b_a, b_b
    
    # Evaporative recycling
    a1_evap = fractionation_hydrogen(T_ls)
    a0_evap = a1_evap * D_ratio_2H**(-n_exp)
    a_evap = a1_evap * r_h_wind / (1 - a0_evap * (1 - r_h_wind))
    u_evap = 1.0 / (a0_evap * (1 - r_h_wind)) - 1
    u_evap[r_h_wind == 1] = 0
    
    a_residual = (f_p_wind**u_evap * a_prec + 
                  (1 - f_p_wind**u_evap) * a_evap)
    del a0_evap, a1_evap
    
    # Integrate along wind direction
    R_prec = (a_prec / a_prec[0, :] *
              np.exp(np.cumsum((a_residual - 1) * d_ln_fm_ds, axis=0) * d_s))
    
    if is_fit:
        del a_prec, a_residual
    
    # Calculate evaporation grids if not fitting
    evap_d2H_grid = None
    u_evap_d2H_grid = None
    
    if not is_fit:
        d2H_evap = (a_evap / a_prec) * R_prec - 1
        F = RegularGridInterpolator((s, t), d2H_evap, method='linear',
                                     bounds_error=False, fill_value=None)
        evap_d2H_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        
        F = RegularGridInterpolator((s, t), u_evap, method='linear',
                                     bounds_error=False, fill_value=None)
        u_evap_d2H_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
    
    # Transform to geographic grid
    F = RegularGridInterpolator((s, t), R_prec, method='linear',
                                 bounds_error=False, fill_value=None)
    del R_prec
    
    if lat.ndim == 1:
        lat = lat.reshape(-1, 1)
    
    R_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
    d2H_grid = ((1 + d2H0 + d_d2H0_d_lat * (np.abs(lat) - np.abs(lat0))) 
                * R_grid - 1)
    
    # ============================================================
    # Calculate d18O grid
    # ============================================================
    
    a_ls = fractionation_oxygen(T_ls)
    a_258 = fractionation_oxygen(258.0)
    a_223 = fractionation_oxygen(223.0)
    
    b_a = (a_223 - a_258) / (z_bar_223 - z_bar_258)
    b_a[np.isnan(b_a)] = 0
    b_b = (a_258 - a_ls) / z_bar_258
    b_b[np.isnan(b_b)] = 0
    del z_bar_223
    
    a_prec = (
        ((a_258 + b_a * (h_s + z_bar_fs - z_bar_258)) * np.exp(-z_bar_fs / h_r)
         + a_ls * (1 - np.exp(-z_bar_fs / h_r))) * np.exp(-z_bar_fs / h_s)
        + b_b * (h_r**2 * h_s / (h_s + h_r)**2)
        * (1 - (1 + (1/h_s + 1/h_r) * z_bar_fs)
           * np.exp(-(1/h_s + 1/h_r) * z_bar_fs))
        + a_ls * (1 - np.exp(-z_bar_fs / h_s))
    )
    del b_a, b_b, z_bar_258
    
    # Evaporative recycling for oxygen
    a1_evap = fractionation_oxygen(T_ls)
    a0_evap = a1_evap * D_ratio_18O**(-n_exp)
    a_evap = a1_evap * r_h_wind / (1 - a0_evap * (1 - r_h_wind))
    u_evap_o = 1.0 / (a0_evap * (1 - r_h_wind)) - 1
    u_evap_o[r_h_wind == 1] = 0
    
    a_residual = (f_p_wind**u_evap_o * a_prec + 
                  (1 - f_p_wind**u_evap_o) * a_evap)
    del a0_evap, a1_evap
    
    if is_fit:
        del a_evap, u_evap_o
    
    R_prec = (a_prec / a_prec[0, :] *
              np.exp(np.cumsum((a_residual - 1) * d_ln_fm_ds, axis=0) * d_s))
    del a_residual, d_ln_fm_ds
    
    if is_fit:
        del a_prec
    
    # Calculate evaporation grids for oxygen
    evap_d18O_grid = None
    u_evap_d18O_grid = None
    
    if not is_fit:
        d18O_evap = (a_evap / a_prec) * R_prec - 1
        F = RegularGridInterpolator((s, t), d18O_evap, method='linear',
                                     bounds_error=False, fill_value=None)
        evap_d18O_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        
        F = RegularGridInterpolator((s, t), u_evap_o, method='linear',
                                     bounds_error=False, fill_value=None)
        u_evap_d18O_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
    
    # Transform to geographic grid
    F = RegularGridInterpolator((s, t), R_prec, method='linear',
                                 bounds_error=False, fill_value=None)
    del R_prec
    
    R_grid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
    d18O_grid = ((1 + d18O0 + d_d18O0_d_lat * (np.abs(lat) - np.abs(lat0))) 
                 * R_grid - 1)
    
    return {
        'd2H_grid': d2H_grid,
        'd18O_grid': d18O_grid,
        'evap_d2H_grid': evap_d2H_grid,
        'u_evap_d2H_grid': u_evap_d2H_grid,
        'evap_d18O_grid': evap_d18O_grid,
        'u_evap_d18O_grid': u_evap_d18O_grid
    }
