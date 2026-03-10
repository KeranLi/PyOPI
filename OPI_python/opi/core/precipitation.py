"""
Precipitation Grid Calculations

This module calculates precipitation rate and moisture ratio grids using
a modified version of the Linear Theory of Orographic Precipitation (LTOP)
algorithm from Smith and Barstad (2004).

Modifications include:
- Coriolis forcing
- Moisture balance with evaporative recycling
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .fourier_solution import fourier_solution
from ..constants import GAMMA_D


def isotherm(TR, z_bar, T, gamma_env, gamma_sat, h_rho, n_s, n_t, h_hat, k_z):
    """
    Calculate height array for an isothermal surface.
    
    Determines the height of a specified temperature surface using the
    linearized solution for airflow over topography.
    
    Parameters
    ----------
    TR : float
        Target isotherm temperature (K)
    z_bar : ndarray
        Base-state elevation vector (m)
    T : ndarray
        Base-state temperature profile (K)
    gamma_env : ndarray
        Environmental lapse rate (K/m)
    gamma_sat : ndarray
        Saturation lapse rate (K/m)
    h_rho : float
        Density scale height (m)
    n_s, n_t : int
        Output grid dimensions
    h_hat : ndarray
        Fourier coefficients of topography
    k_z : ndarray
        Vertical wavenumber grid
        
    Returns
    -------
    z_R : ndarray
        Height of temperature surface TR (m)
    """
    # Interpolate to find base-state elevation of isotherm
    # T typically decreases with height, so we may need to flip
    if T[0] > T[-1]:
        T_flip = T[::-1]
        z_bar_flip = z_bar[::-1]
    else:
        T_flip = T
        z_bar_flip = z_bar
    
    z_bar_R = np.interp(TR, T_flip, z_bar_flip, 
                        left=z_bar_flip[0], right=z_bar_flip[-1])
    
    # Get lapse rates at this elevation
    if z_bar[0] > z_bar[-1]:
        z_bar_flip = z_bar[::-1]
        gamma_env_flip = gamma_env[::-1]
        gamma_sat_flip = gamma_sat[::-1]
    else:
        z_bar_flip = z_bar
        gamma_env_flip = gamma_env
        gamma_sat_flip = gamma_sat
    
    gamma_env_R = np.interp(z_bar_R, z_bar_flip, gamma_env_flip,
                           left=gamma_env_flip[0], right=gamma_env_flip[-1])
    gamma_sat_R = np.interp(z_bar_R, z_bar_flip, gamma_sat_flip,
                           left=gamma_sat_flip[0], right=gamma_sat_flip[-1])
    
    # Calculate perturbation height using Fourier solution
    exponent = (1j * k_z + 1/(2*h_rho)) * z_bar_R
    z_prime = np.fft.ifft2(h_hat * np.exp(exponent))
    z_prime = np.real(z_prime)  # Should be real due to symmetry
    z_prime = z_prime[:n_s, :n_t]
    
    # Total height with perturbation
    # Perturbation is scaled by (1 - gamma_sat/gamma_env)
    if abs(gamma_env_R) < 1e-10:
        gamma_env_R = 1e-10
    
    z_R = z_bar_R + z_prime * (1 - gamma_sat_R / gamma_env_R)
    
    return z_R


def precipitation_grid(x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c,
                       h_rho, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
                       rho_s0, h_s, f_p0):
    """
    Calculate precipitation rate and moisture ratio grids.
    
    Uses the modified LTOP (Linear Theory of Orographic Precipitation)
    algorithm with Coriolis forcing and moisture balance.
    
    Parameters
    ----------
    x, y : ndarray
        Geographic grid vectors (m)
    h_grid : ndarray
        Topography grid (m)
    U : float
        Wind speed (m/s)
    azimuth : float
        Wind direction (degrees from North)
    NM : float
        Saturated buoyancy frequency (rad/s)
    f_c : float
        Coriolis parameter (rad/s)
    kappa : float
        Eddy diffusion coefficient (m²/s)
    tau_c : float
        Cloud water residence time (s)
    h_rho : float
        Density scale height (m)
    z_bar : ndarray
        Base-state elevation vector (m)
    T : ndarray
        Temperature profile (K)
    gamma_env : ndarray
        Environmental lapse rate (K/m)
    gamma_sat : ndarray
        Saturation lapse rate (K/m)
    gamma_ratio : float
        Weighted mean of gamma_sat/gamma_env
    rho_s0 : float
        Surface water vapor density (kg/m³)
    h_s : float
        Water vapor scale height (m)
    f_p0 : float
        Residual precipitation fraction (evaporative recycling)
        
    Returns
    -------
    dict : Dictionary containing:
        - 's', 't': Wind grid coordinate vectors
        - 'Sxy', 'Txy': Geographic coordinates in wind system
        - 'p_grid': Precipitation rate grid (kg/m²/s)
        - 'h_wind': Topography in wind coordinates
        - 'f_m_wind': Moisture ratio field
        - 'r_h_wind': Relative humidity field
        - 'f_p_wind': Residual precipitation fraction field
        - 'z223_wind', 'z258_wind': Heights of isotherms (m)
        - 'tau_f': Mean precipitation fall time (s)
        
    References
    ----------
    Smith and Barstad (2004) - A linear theory of orographic precipitation
    """
    # Get Fourier solution
    fourier = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    
    s = fourier['s']
    t = fourier['t']
    Sxy = fourier['Sxy']
    Txy = fourier['Txy']
    h_wind = fourier['h_wind']
    k_s = fourier['k_s']
    k_t = fourier['k_t']
    h_hat = fourier['h_hat']
    k_z = fourier['k_z']
    n_s_pad = fourier['n_s_pad']
    n_t_pad = fourier['n_t_pad']
    d_s = fourier['d_s']
    
    n_s, n_t = h_wind.shape
    k_s_col = k_s.reshape(-1, 1)
    
    # Calculate heights of key isotherms
    z258_wind = isotherm(258.0, z_bar, T, gamma_env, gamma_sat, h_rho,
                         n_s, n_t, h_hat, k_z)
    
    # Calculate mean fall time
    # Fall velocities: w_f_snow = -1 m/s, w_f_rain = -6 m/s
    w_f_snow = -1.0
    w_f_rain = -6.0
    
    tau_f_grid = np.where(
        z258_wind <= h_wind,
        -h_s / w_f_snow,
        -(z258_wind - h_wind) / w_f_rain - 
        h_s * np.exp(-(z258_wind - h_wind) / h_s) / w_f_snow
    )
    tau_f = np.mean(tau_f_grid)
    
    if np.isnan(tau_f):
        raise ValueError("Mean fall time tau_f is NaN")
    
    # Calculate additional isotherm heights for isotope calculations
    z223_wind = isotherm(223.0, z_bar, T, gamma_env, gamma_sat, h_rho,
                         n_s, n_t, h_hat, k_z)
    z258_wind = isotherm(258.0, z_bar, T, gamma_env, gamma_sat, h_rho,
                         n_s, n_t, h_hat, k_z)
    
    # Calculate reference precipitation rate (wave domain)
    # Green's functions for source, cloud, and fallout
    GS_hat = (gamma_ratio * rho_s0 * 1j * k_s_col * U / 
              (1 - h_s * (1j * k_z + 1/(2*h_rho))))
    GC_hat = 1.0 / (tau_c * (kappa * (k_s_col**2 + k_t**2) + 
                              1j * k_s_col * U) + 1)
    GF_hat = 1.0 / (tau_f * (kappa * (k_s_col**2 + k_t**2) + 
                              1j * k_s_col * U) + 1)
    
    p_star_hat = GS_hat * GC_hat * GF_hat * h_hat
    
    # Transform to space domain
    p_star_wind = np.fft.ifft2(p_star_hat)
    p_star_wind = np.real(p_star_wind)
    p_star_wind = p_star_wind[:n_s, :n_t]
    p_star_wind[p_star_wind < 0] = 0
    
    # Calculate column-density fields for moisture balance
    QC_star = np.fft.ifft2(tau_c * GS_hat * GC_hat * h_hat)
    QC_star = np.real(QC_star)[:n_s, :n_t]
    QC_star[QC_star < 0] = 0
    
    QF_star = np.fft.ifft2(tau_f * GS_hat * GC_hat * GF_hat * h_hat)
    QF_star = np.real(QF_star)[:n_s, :n_t]
    QF_star[QF_star < 0] = 0
    
    QT_star = rho_s0 * h_s + QC_star + QF_star
    
    # Handle evaporative recycling
    if f_p0 == 1.0:
        r_h_wind = np.ones((n_s, n_t))
        f_p_wind = np.ones((n_s, n_t))
    else:
        # Calculate cloud water density at surface
        rho_c_hat = ((gamma_ratio * rho_s0 * 1j * k_s_col * U / h_s) * h_hat /
                     ((kappa * (k_s_col**2 + k_t**2) + 1j * k_s_col * U) + 
                      1/tau_c))
        rho_c_wind = np.real(np.fft.ifft2(rho_c_hat))[:n_s, :n_t]
        
        # Relative humidity (0 to 1)
        r_h_wind = 1.0 + (GAMMA_D / gamma_sat[0]) * rho_c_wind / rho_s0
        r_h_wind = np.clip(r_h_wind, 0, 1)
        
        # Residual precipitation fraction
        f_p_wind = np.ones((n_s, n_t))
        f_p_wind[r_h_wind < 1] = f_p0
    
    # Integrate along wind direction for moisture balance
    integrand = f_p_wind * p_star_wind * d_s / QT_star
    f_v_wind = (rho_s0 * h_s / QT_star) * np.exp(
        -(1.0/U) * np.cumsum(integrand, axis=0) * d_s
    )
    
    # Calculate precipitation rate
    p_wind = f_v_wind * p_star_wind
    
    # Transform back to geographic grid
    interp_func = RegularGridInterpolator(
        (s, t), p_wind,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    points = np.column_stack([Txy.ravel(), Sxy.ravel()])
    p_grid = interp_func(points).reshape(Sxy.shape)
    
    # Moisture ratio field
    f_m_wind = f_v_wind * QT_star / (rho_s0 * h_s)
    
    return {
        's': s,
        't': t,
        'Sxy': Sxy,
        'Txy': Txy,
        'p_grid': p_grid,
        'h_wind': h_wind,
        'f_m_wind': f_m_wind,
        'r_h_wind': r_h_wind,
        'f_p_wind': f_p_wind,
        'z223_wind': z223_wind,
        'z258_wind': z258_wind,
        'tau_f': tau_f
    }
