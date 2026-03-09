"""
Isotope Grid Calculation

Calculates the spatial distribution of precipitation isotopes (delta^2H and delta^18O)
using the mixed cloud isotopic model (MCIM) of Ciais and Jouzel (1994).

Matches MATLAB isotopeGrid.m implementation exactly.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from .fractionation import fractionation_hydrogen, fractionation_oxygen


def isotope_grid(s, t, s_xy, t_xy, lat, lat0, h_wind, f_m_wind, r_h_wind, f_p_wind,
                 z223_wind, z258_wind, tau_f, U, T, gamma_sat, h_s, h_r,
                 d2h0, d18o0, d_d2h0_d_lat, d_d18o0_d_lat, is_fit):
    """
    Calculate precipitation isotope (delta^2H, delta^18O) spatial distribution.
    
    Matches MATLAB isotopeGrid.m implementation.
    
    Parameters
    ----------
    s, t : ndarray
        Wind grid coordinate vectors
    s_xy, t_xy : ndarray
        Geographic grids in wind coordinates
    lat : ndarray
        Latitude of sample points
    lat0 : float
        Reference latitude
    h_wind : ndarray
        Topography in wind coordinates
    f_m_wind : ndarray
        Moisture ratio field
    r_h_wind : ndarray
        Relative humidity field
    f_p_wind : ndarray
        Residual precipitation fraction field
    z223_wind, z258_wind : ndarray
        Heights of 223K and 258K isotherms
    tau_f : float
        Precipitation fall time
    U : float
        Wind speed
    T : ndarray
        Temperature profile
    gamma_sat : ndarray
        Saturated lapse rate
    h_s : float
        Water vapor scale height
    h_r : float
        Isotope exchange distance (540 m)
    d2h0, d18o0 : float
        Base isotope values (fraction, not permil)
    d_d2h0_d_lat, d_d18o0_d_lat : float
        Latitudinal gradients (fraction per degree)
    is_fit : bool
        Whether this is a fitting calculation
    
    Returns
    -------
    dict : Dictionary containing isotope grids
    """
    # Constants
    d_ratio_2h = 0.9755  # Diffusivity ratio for hydrogen
    d_ratio_18o = 0.9723  # Diffusivity ratio for oxygen
    n = 1  # Exponent for evaporation fractionation
    
    # Grid parameters
    d_s = s[1] - s[0]
    n_s, n_t = h_wind.shape
    
    # Shear for bringing fall path to vertical
    shear = U * tau_f / h_s
    
    # Surface position with shear
    s_surface_shear_wind = s[:, np.newaxis] + shear * h_wind
    
    # Surface temperature
    t_ls_wind = T[0] - gamma_sat[0] * h_wind
    
    # Calculate zBar223Wind (height of 223K isotherm above surface)
    z223_wind = z223_wind.copy()
    s_shear_wind = s[:, np.newaxis] + shear * z223_wind
    for j in range(n_t):
        # Use first crossing where surface is steeper than fall path
        s_col = s_shear_wind[:, j]
        # Find monotonic increasing section
        cummax_vals = np.maximum.accumulate(np.concatenate([[s_col[0] - 1], s_col[:-1]]))
        i_monotonic = s_col > cummax_vals
        if np.sum(i_monotonic) >= 2:
            z223_wind[:, j] = np.interp(
                s_surface_shear_wind[:, j],
                s_shear_wind[i_monotonic, j],
                z223_wind[i_monotonic, j],
                left=z223_wind[0, j],
                right=z223_wind[i_monotonic, j][-1] if np.any(i_monotonic) else z223_wind[0, j]
            )
    z_bar_223_wind = z223_wind - h_wind
    
    # Calculate zBar258Wind (height of 258K isotherm above surface)
    z258_wind = z258_wind.copy()
    s_shear_wind = s[:, np.newaxis] + shear * z258_wind
    for j in range(n_t):
        s_col = s_shear_wind[:, j]
        cummax_vals = np.maximum.accumulate(np.concatenate([[s_col[0] - 1], s_col[:-1]]))
        i_monotonic = s_col > cummax_vals
        if np.sum(i_monotonic) >= 2:
            z258_wind[:, j] = np.interp(
                s_surface_shear_wind[:, j],
                s_shear_wind[i_monotonic, j],
                z258_wind[i_monotonic, j],
                left=z258_wind[0, j],
                right=z258_wind[i_monotonic, j][-1] if np.any(i_monotonic) else z258_wind[0, j]
            )
    z_bar_258_wind = z258_wind - h_wind
    
    # Height of freezing surface above land surface (set to 0 where below surface)
    z_bar_fs_wind = z_bar_258_wind.copy()
    z_bar_fs_wind[z_bar_fs_wind < 0] = 0
    
    # Differentiate ln(f_m) in wind direction
    d_ln_f_m_d_s_wind = np.gradient(np.log(f_m_wind), d_s, axis=0)
    
    # ==================== HYDROGEN ISOTOPES ====================
    
    # Get specific equilibrium factors
    a_ls_wind = fractionation_hydrogen(t_ls_wind)
    a_258 = fractionation_hydrogen(258.0)
    a_223 = fractionation_hydrogen(223.0)
    
    # Calculate fractionation factors by vertical averaging
    # Subscripts A and B refer to above and below 258 K point
    with np.errstate(divide='ignore', invalid='ignore'):
        b_a = (a_223 - a_258) / (z_bar_223_wind - z_bar_258_wind)
        b_a = np.nan_to_num(b_a, nan=0.0, posinf=0.0, neginf=0.0)
        
        b_b = (a_258 - a_ls_wind) / z_bar_258_wind
        b_b = np.nan_to_num(b_b, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fractionation factor for precipitation (R_prec/R_vapor)
    # Complex vertical averaging formula from MATLAB
    a_prec_wind = (
        ((a_258 + b_a * (h_s + z_bar_fs_wind - z_bar_258_wind)) * np.exp(-z_bar_fs_wind / h_r)
         + a_ls_wind * (1 - np.exp(-z_bar_fs_wind / h_r))) * np.exp(-z_bar_fs_wind / h_s)
        + b_b * (h_r**2 * h_s / (h_s + h_r)**2)
        * (1 - (1 + (1/h_s + 1/h_r) * z_bar_fs_wind)
           * np.exp(-(1/h_s + 1/h_r) * z_bar_fs_wind))
        + a_ls_wind * (1 - np.exp(-z_bar_fs_wind / h_s))
    )
    
    # Evaporative recycling
    a_1_evap_wind = fractionation_hydrogen(t_ls_wind)
    a_0_evap_wind = a_1_evap_wind * d_ratio_2h**(-n)
    a_evap_wind = a_1_evap_wind * r_h_wind / (1 - a_0_evap_wind * (1 - r_h_wind))
    
    # Exponent for integration of evaporative fractionation
    u_evap_d2h_wind = 1 / (a_0_evap_wind * (1 - r_h_wind)) - 1
    u_evap_d2h_wind[r_h_wind == 1] = 0
    
    # Combine to get fractionation factor for residual precipitation
    a_residual_vapor_wind = (
        f_p_wind**u_evap_d2h_wind * a_prec_wind
        + (1 - f_p_wind**u_evap_d2h_wind) * a_evap_wind
    )
    
    # Integrate fractionation along wind direction
    integrand = (a_residual_vapor_wind - 1) * d_ln_f_m_d_s_wind
    r_prec_wind = (
        a_prec_wind / a_prec_wind[0, :]
        * np.exp(cumulative_trapezoid(integrand, dx=d_s, axis=0, initial=0))
    )
    
    # Calculate evaporation grids if not fitting
    evap_d2h_grid = None
    u_evap_d2h_grid = None
    if not is_fit:
        d2h_evap_wind = (a_evap_wind / a_prec_wind) * r_prec_wind - 1
        u_evap_d2h_grid = _interp_to_geo(u_evap_d2h_wind, s, t, s_xy, t_xy)
        evap_d2h_grid = _interp_to_geo(d2h_evap_wind, s, t, s_xy, t_xy)
    
    # Transform to geographic grid and apply regional variation
    d2h_grid = _finalize_isotope_grid(
        r_prec_wind, s, t, s_xy, t_xy, d2h0, d_d2h0_d_lat, lat, lat0
    )
    
    # ==================== OXYGEN ISOTOPES ====================
    
    # Get specific equilibrium factors
    a_ls_wind = fractionation_oxygen(t_ls_wind)
    a_258 = fractionation_oxygen(258.0)
    a_223 = fractionation_oxygen(223.0)
    
    # Calculate fractionation factors by vertical averaging
    with np.errstate(divide='ignore', invalid='ignore'):
        b_a = (a_223 - a_258) / (z_bar_223_wind - z_bar_258_wind)
        b_a = np.nan_to_num(b_a, nan=0.0, posinf=0.0, neginf=0.0)
        
        b_b = (a_258 - a_ls_wind) / z_bar_258_wind
        b_b = np.nan_to_num(b_b, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fractionation factor for precipitation
    a_prec_wind = (
        ((a_258 + b_a * (h_s + z_bar_fs_wind - z_bar_258_wind)) * np.exp(-z_bar_fs_wind / h_r)
         + a_ls_wind * (1 - np.exp(-z_bar_fs_wind / h_r))) * np.exp(-z_bar_fs_wind / h_s)
        + b_b * (h_r**2 * h_s / (h_s + h_r)**2)
        * (1 - (1 + (1/h_s + 1/h_r) * z_bar_fs_wind)
           * np.exp(-(1/h_s + 1/h_r) * z_bar_fs_wind))
        + a_ls_wind * (1 - np.exp(-z_bar_fs_wind / h_s))
    )
    
    # Evaporative recycling
    a_1_evap_wind = fractionation_oxygen(t_ls_wind)
    a_0_evap_wind = a_1_evap_wind * d_ratio_18o**(-n)
    a_evap_wind = a_1_evap_wind * r_h_wind / (1 - a_0_evap_wind * (1 - r_h_wind))
    
    # Exponent for integration
    u_evap_d18o_wind = 1 / (a_0_evap_wind * (1 - r_h_wind)) - 1
    u_evap_d18o_wind[r_h_wind == 1] = 0
    
    # Combine fractionation factors
    a_residual_vapor_wind = (
        f_p_wind**u_evap_d18o_wind * a_prec_wind
        + (1 - f_p_wind**u_evap_d18o_wind) * a_evap_wind
    )
    
    # Integrate along wind direction
    integrand = (a_residual_vapor_wind - 1) * d_ln_f_m_d_s_wind
    r_prec_wind = (
        a_prec_wind / a_prec_wind[0, :]
        * np.exp(cumulative_trapezoid(integrand, dx=d_s, axis=0, initial=0))
    )
    
    # Calculate evaporation grids if not fitting
    evap_d18o_grid = None
    u_evap_d18o_grid = None
    if not is_fit:
        d18o_evap_wind = (a_evap_wind / a_prec_wind) * r_prec_wind - 1
        u_evap_d18o_grid = _interp_to_geo(u_evap_d18o_wind, s, t, s_xy, t_xy)
        evap_d18o_grid = _interp_to_geo(d18o_evap_wind, s, t, s_xy, t_xy)
    
    # Transform to geographic grid
    d18o_grid = _finalize_isotope_grid(
        r_prec_wind, s, t, s_xy, t_xy, d18o0, d_d18o0_d_lat, lat, lat0
    )
    
    return {
        'd2h_grid': d2h_grid,
        'd18o_grid': d18o_grid,
        'evap_d2h_grid': evap_d2h_grid if evap_d2h_grid is not None else np.zeros_like(d2h_grid),
        'u_evap_d2h_grid': u_evap_d2h_grid if u_evap_d2h_grid is not None else np.zeros_like(d2h_grid),
        'evap_d18o_grid': evap_d18o_grid if evap_d18o_grid is not None else np.zeros_like(d18o_grid),
        'u_evap_d18o_grid': u_evap_d18o_grid if u_evap_d18o_grid is not None else np.zeros_like(d18o_grid)
    }


def _interp_to_geo(wind_data, s, t, s_xy, t_xy):
    """Interpolate from wind grid to geographic grid."""
    interp_func = RegularGridInterpolator(
        (s, t), wind_data,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    return interp_func(np.column_stack([s_xy.ravel(), t_xy.ravel()])).reshape(s_xy.shape)


def _finalize_isotope_grid(r_prec_wind, s, t, s_xy, t_xy, iso0, d_iso0_d_lat, lat, lat0):
    """
    Finalize isotope calculation:
    1) Transform to geographic grid
    2) Convert from isotope ratio to delta representation
    3) Account for regional variation
    """
    # Interpolate to geographic grid
    r_prec_grid = _interp_to_geo(r_prec_wind, s, t, s_xy, t_xy)
    
    # Apply regional variation and convert to delta
    iso_grid = (1 + iso0 + d_iso0_d_lat * (np.abs(lat.mean()) - np.abs(lat0))) * r_prec_grid - 1
    
    return iso_grid


if __name__ == "__main__":
    print("Testing isotope_grid module...")
    print("Run test_isotope.py for full tests")
