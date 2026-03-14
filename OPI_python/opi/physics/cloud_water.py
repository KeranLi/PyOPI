"""
Cloud Water Content Calculations

Calculates cloudwater density for specified section.
Matches MATLAB's cloudWater.m
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from typing import Tuple

from .fourier_solution import fourier_solution
from .precipitation_grid import isotherm


def calculate_cloud_water_section(
    x_path: np.ndarray,
    y_path: np.ndarray,
    z_max: float,
    x: np.ndarray,
    y: np.ndarray,
    h_grid: np.ndarray,
    U: float,
    azimuth: float,
    NM: float,
    f_c: float,
    kappa: float,
    tau_c: float,
    tau_f: float,
    h_rho: float,
    z_bar: np.ndarray,
    T: np.ndarray,
    gamma_env: float,
    gamma_sat: float,
    gamma_ratio: float,
    rho_s0: float,
    h_v: float,
    n_z: int = 100
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Calculate cloudwater density for specified section.
    
    Matches MATLAB's cloudWater.m function.
    
    Parameters
    ----------
    x_path, y_path : ndarray
        Eastings and northings for section path (vectors, m)
    z_max : float
        Sea level elevation for top of rhoC grid (scalar, m)
    x, y : ndarray
        Grid vectors for geographic coordinates (m)
    h_grid : ndarray
        Topography grid (m)
    U : float
        Base-state wind speed (m/s)
    azimuth : float
        Base-state wind direction (down wind) (degrees)
    NM : float
        Saturation buoyancy frequency (rad/s)
    f_c : float
        Coriolis frequency (rad/s)
    kappa : float
        Horizontal eddy diffusivity (m^2/s)
    tau_c : float
        Mean residence-time for cloud water (s)
    tau_f : float
        Mean residence-time for falling precipitation (s)
    h_rho : float
        Scale height for density (m)
    z_bar : ndarray
        Base state elevation vector (m)
    T : ndarray
        Temperature profile (K)
    gamma_env : float
        Environmental lapse rate (K/m)
    gamma_sat : float
        Saturated adiabatic lapse rate (K/m)
    gamma_ratio : float
        Ratio of lapse rates (dimensionless)
    rho_s0 : float
        Saturated water vapor density at sea level (kg/m^3)
    h_v : float
        Mean height of water vapor (m)
    n_z : int, optional
        Number of vertical levels (default: 100)
    
    Returns
    -------
    z_rho_c : ndarray
        Grid vector with sea level elevations for rhoC grid (m), shape (n_z, n_path)
    rho_c : ndarray
        Grid indicating cloudwater density (kg/m^3), shape (n_z, n_path)
    z_c_mean : float
        Mean height above ground surface of cloud water (m)
    z_248_path : ndarray
        Height along path of the 248 K isotherm (m)
    z_268_path : ndarray
        Height along path of the 268 K isotherm (m)
    """
    # Compute reference solution
    # Get Fourier solution for Euler equations
    fs = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    s = fs['s']
    t = fs['t']
    k_s = fs['k_s']
    k_t = fs['k_t']
    h_hat = fs['h_hat']
    k_z = fs['k_z']
    
    # Parameters for wind grids
    d_s = s[1] - s[0]
    n_s = len(s)
    n_t = len(t)
    
    # Reshape k_s and k_t to match grid shape for broadcasting
    k_s_grid = k_s.reshape(-1, 1) if k_s.ndim == 1 else k_s
    k_t_grid = k_t.reshape(1, -1) if k_t.ndim == 1 else k_t
    
    # Calculate PStarHat, reference precipitation rate for wave domain
    g_s_hat = gamma_ratio * rho_s0 * 1j * k_s_grid * U / (1 - 1j * k_z * h_v)
    g_c_hat = 1.0 / (tau_c * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)
    g_f_hat = 1.0 / (tau_f * (kappa * (k_s_grid**2 + k_t_grid**2) - 1j * k_s_grid * U) + 1)
    p_star_hat = g_s_hat * g_c_hat * g_f_hat * h_hat
    
    # Transform back to space domain, remove padding
    p_star_pos_wind = np.fft.ifft2(p_star_hat).real
    p_star_pos_wind = p_star_pos_wind[:n_s, :n_t]
    p_star_pos_wind[p_star_pos_wind < 0] = 0
    
    # Calculate vapor ratio, fV
    # Calculate column-density fields for cloud water QC, falling precipitation QF
    q_c_star_pos_wind = np.fft.ifft2(tau_c * g_s_hat * g_c_hat * h_hat).real
    q_c_star_pos_wind = q_c_star_pos_wind[:n_s, :n_t]
    q_c_star_pos_wind[q_c_star_pos_wind < 0] = 0
    
    q_f_star_pos_wind = np.fft.ifft2(tau_f * g_s_hat * g_c_hat * g_f_hat * h_hat).real
    q_f_star_pos_wind = q_f_star_pos_wind[:n_s, :n_t]
    q_f_star_pos_wind[q_f_star_pos_wind < 0] = 0
    
    q_t_star_pos_wind = rho_s0 * h_v + q_c_star_pos_wind + q_f_star_pos_wind
    
    # Integrate along the columns of the s,t grid to calculate water-vapor ratio
    # Use scipy.integrate.cumulative_trapezoid (replaces deprecated np.cumtrapz)
    f_v_integrand = p_star_pos_wind * d_s / (U * q_t_star_pos_wind)
    f_v_integral = cumulative_trapezoid(f_v_integrand, axis=0, initial=0)
    f_v_wind = (rho_s0 * h_v / q_t_star_pos_wind) * np.exp(-f_v_integral)
    
    # Compute cloud-water density in section along path
    # Reference cloud-water density field at zBar = 0
    rho_c_star_hat_0 = (gamma_ratio * rho_s0 * 1j * k_s_grid * U / h_v) * h_hat / \
                       (kappa * (k_s_grid**2 + k_t_grid**2) + 1j * k_s_grid * U + 1.0/tau_c)
    
    # Initialize variables
    z_bar_rho_c = np.linspace(0, z_max, n_z)
    n_path = len(x_path)
    z_rho_c = np.zeros((n_z, n_path))
    rho_c = np.zeros((n_z, n_path))
    
    # Calculate s,t coordinates for path (maintains right-handed s,t,z)
    s_path = x_path * np.sin(np.deg2rad(azimuth)) + y_path * np.cos(np.deg2rad(azimuth))
    t_path = -x_path * np.cos(np.deg2rad(azimuth)) + y_path * np.sin(np.deg2rad(azimuth))
    
    # Calculate rhoC at increments of zBar
    for i in range(n_z):
        # Calculate rhoC for current zBar
        rho_c_for_z_bar = np.fft.ifft2(
            rho_c_star_hat_0 * np.exp((1j * k_z + 1/(2*h_rho) - 1/h_v) * z_bar_rho_c[i])
        ).real
        rho_c_for_z_bar = f_v_wind * rho_c_for_z_bar[:n_s, :n_t]
        
        # Interpolate to path points
        interp = RegularGridInterpolator(
            (s, t), rho_c_for_z_bar,
            method='linear', bounds_error=False, fill_value=0
        )
        rho_c[i, :] = interp(np.column_stack([s_path, t_path]))
        
        # Calculate elevation z for path points at current zBar
        z_for_z_bar = z_bar_rho_c[i] + np.fft.ifft2(
            h_hat * np.exp((1j * k_z - 1/(2*h_rho)) * z_bar_rho_c[i])
        ).real
        z_for_z_bar = z_for_z_bar[:n_s, :n_t]
        
        interp_z = RegularGridInterpolator(
            (s, t), z_for_z_bar,
            method='linear', bounds_error=False, fill_value=z_bar_rho_c[i]
        )
        z_rho_c[i, :] = interp_z(np.column_stack([s_path, t_path]))
    
    # rhoC is now in terms of zBar (height above base of model)
    # Save positive part for weighted mean calculation
    rho_c_pos_for_z_bar = rho_c.copy()
    
    # Interpolate rhoC results to a regular grid
    for j in range(n_path):
        valid = ~np.isnan(z_rho_c[:, j]) & ~np.isnan(rho_c[:, j])
        if np.sum(valid) > 1:
            rho_c[:, j] = np.interp(
                z_bar_rho_c,
                z_rho_c[valid, j],
                rho_c[valid, j],
                left=rho_c[valid, j][0] if len(valid) > 0 else 0,
                right=rho_c[valid, j][-1] if len(valid) > 0 else 0
            )
    
    z_rho_c = z_bar_rho_c.reshape(-1, 1) * np.ones((1, n_path))
    
    # Weighted mean height of cloud water relative to the land surface
    rho_c_pos_for_z_bar[rho_c_pos_for_z_bar <= 0] = np.nan
    valid = ~np.isnan(rho_c_pos_for_z_bar)
    if np.any(valid):
        z_c_mean = np.nansum(z_bar_rho_c.reshape(-1, 1) * rho_c_pos_for_z_bar) / \
                   np.nansum(rho_c_pos_for_z_bar)
    else:
        z_c_mean = np.nan
    
    # Calculate height along path of the 248 K (-25°C) and 268 K (-5°C) isotherms
    # Note: MATLAB uses Celsius in comments but function receives Kelvin
    z_248 = isotherm(248.0, z_bar, T, gamma_env, gamma_sat, h_rho, n_s, n_t, h_hat, k_z)
    interp_248 = RegularGridInterpolator(
        (t, s), z_248,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    z_248_path = interp_248(np.column_stack([t_path, s_path]))
    
    z_268 = isotherm(268.0, z_bar, T, gamma_env, gamma_sat, h_rho, n_s, n_t, h_hat, k_z)
    interp_268 = RegularGridInterpolator(
        (t, s), z_268,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    z_268_path = interp_268(np.column_stack([t_path, s_path]))
    
    return z_rho_c, rho_c, z_c_mean, z_248_path, z_268_path


def calculate_cloud_water(
    z_bar: np.ndarray,
    T: np.ndarray,
    gamma_sat: np.ndarray,
    p_grid: np.ndarray,
    q_s0: float,
    h_s: float
) -> np.ndarray:
    """
    Calculate cloud water content from precipitation and temperature.
    
    Simplified version for basic calculations.
    
    Parameters
    ----------
    z_bar : ndarray
        Base state elevation vector (m)
    T : ndarray
        Temperature profile (K)
    gamma_sat : ndarray
        Saturated adiabatic lapse rate (K/m)
    p_grid : ndarray
        Precipitation rate (m/s)
    q_s0 : float
        Surface saturation mixing ratio (kg/kg)
    h_s : float
        Water vapor scale height (m)
    
    Returns
    -------
    q_cloud : ndarray
        Cloud water mixing ratio (kg/kg)
    """
    # Saturation mixing ratio at each level
    q_s = q_s0 * np.exp(-z_bar / h_s)
    
    # Lapse rate effect
    stability_factor = np.maximum(0, gamma_sat / np.mean(gamma_sat))
    
    # Cloud water mixing ratio
    q_cloud = q_s * stability_factor * (p_grid / np.max(p_grid)) ** 0.5
    
    # Clip to realistic values
    q_cloud = np.clip(q_cloud, 0, 0.01)
    
    return q_cloud


def calculate_relative_humidity(
    s: np.ndarray,
    t: np.ndarray,
    h_wind: np.ndarray,
    z_isotherm_223: np.ndarray,
    z_isotherm_258: np.ndarray,
    z_bar: np.ndarray,
    T: np.ndarray,
    gamma_sat: np.ndarray,
    q_s0: float,
    h_s: float
) -> np.ndarray:
    """
    Calculate relative humidity field at the surface.
    
    Parameters
    ----------
    s, t : ndarray
        Wind coordinate vectors
    h_wind : ndarray
        Topography in wind coordinates (m)
    z_isotherm_223, z_isotherm_258 : ndarray
        Heights of isotherms (m)
    z_bar : ndarray
        Base state elevation vector (m)
    T : ndarray
        Temperature profile (K)
    gamma_sat : ndarray
        Saturated adiabatic lapse rate (K/m)
    q_s0 : float
        Surface saturation mixing ratio (kg/kg)
    h_s : float
        Water vapor scale height (m)
    
    Returns
    -------
    rh_grid : ndarray
        Relative humidity at surface (fraction, 0-1)
    """
    n_s, n_t = h_wind.shape
    rh_grid = np.zeros((n_s, n_t))
    
    # Interpolate base state to surface
    t_interp = interp1d(z_bar, T, fill_value='extrapolate')
    
    for i in range(n_s):
        for j in range(n_t):
            h_surf = h_wind[i, j]
            t_surf = t_interp(h_surf)
            
            # Calculate saturation mixing ratio at surface
            q_s_surf = q_s0 * np.exp(-h_surf / h_s)
            
            # Relative humidity based on temperature and lifting
            z_223 = z_isotherm_223[i, j] if hasattr(z_isotherm_223, 'shape') else 5000
            z_258 = z_isotherm_258[i, j] if hasattr(z_isotherm_258, 'shape') else 3000
            
            # RH increases with depth of moist layer
            moist_depth = z_223 - h_surf
            rh = 0.6 + 0.3 * np.tanh(moist_depth / 3000)
            
            # Adjust for temperature (colder = higher RH)
            rh += 0.1 * (290 - t_surf) / 20
            
            rh_grid[i, j] = np.clip(rh, 0.3, 0.98)
    
    return rh_grid


def calculate_ice_water_content(
    z_bar: np.ndarray,
    T: np.ndarray,
    q_cloud: np.ndarray,
    z_isotherm_223: np.ndarray
) -> np.ndarray:
    """
    Calculate ice water content from cloud water and temperature.
    
    Parameters
    ----------
    z_bar : ndarray
        Base state elevation vector (m)
    T : ndarray
        Temperature profile (K)
    q_cloud : ndarray
        Cloud water mixing ratio (kg/kg)
    z_isotherm_223 : ndarray or float
        Height of -50°C isotherm (m)
    
    Returns
    -------
    q_ice : ndarray
        Ice water mixing ratio (kg/kg)
    """
    from scipy.interpolate import interp1d
    
    # Temperature threshold for ice
    t_freeze = 273.15
    t_ice = 253.15  # All ice below this temperature
    
    # Ice fraction increases as temperature decreases
    ice_fraction = np.clip((t_freeze - T) / (t_freeze - t_ice), 0, 1)
    
    # Ice water content
    q_ice = q_cloud * ice_fraction
    
    return q_ice
