"""
Cloud Water Content Calculations

Calculates cloud liquid water content and related quantities.
"""

import numpy as np
from scipy.interpolate import interp1d


def calculate_cloud_water(z_bar, T, gamma_sat, p_grid, q_s0, h_s):
    """
    Calculate cloud water content from precipitation and temperature.
    
    Based on the relationship between saturation mixing ratio,
    temperature, and precipitation rate.
    
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
    # q_s = q_s0 * exp(-z / h_s)
    q_s = q_s0 * np.exp(-z_bar / h_s)
    
    # Cloud water is related to the excess of saturation
    # and the precipitation rate
    # Simplified: q_cloud proportional to precipitation and temperature
    
    # Lapse rate effect: more unstable = more cloud water
    stability_factor = np.maximum(0, gamma_sat / np.mean(gamma_sat))
    
    # Cloud water mixing ratio (kg/kg)
    q_cloud = q_s * stability_factor * (p_grid / np.max(p_grid)) ** 0.5
    
    # Clip to realistic values
    q_cloud = np.clip(q_cloud, 0, 0.01)
    
    return q_cloud


def calculate_relative_humidity(s, t, h_wind, z_isotherm_223, z_isotherm_258,
                                z_bar, T, gamma_sat, q_s0, h_s):
    """
    Calculate relative humidity field at the surface.
    
    Parameters
    ----------
    s, t : ndarray
        Wind coordinate vectors
    h_wind : ndarray
        Topography in wind coordinates (m)
    z_isotherm_223, z_isotherm_258 : ndarray
        Heights of isotherms (K)
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
    T_interp = interp1d(z_bar, T, fill_value='extrapolate')
    
    for i in range(n_s):
        for j in range(n_t):
            h_surf = h_wind[i, j]
            T_surf = T_interp(h_surf)
            
            # Calculate saturation mixing ratio at surface
            q_s_surf = q_s0 * np.exp(-h_surf / h_s)
            
            # Relative humidity based on temperature and lifting
            # Warmer and more lifting = higher RH
            z_223 = z_isotherm_223[i, j] if hasattr(z_isotherm_223, 'shape') else 5000
            z_258 = z_isotherm_258[i, j] if hasattr(z_isotherm_258, 'shape') else 3000
            
            # RH increases with depth of moist layer
            moist_depth = z_223 - h_surf
            rh = 0.6 + 0.3 * np.tanh(moist_depth / 3000)
            
            # Adjust for temperature (colder = higher RH)
            rh += 0.1 * (290 - T_surf) / 20
            
            rh_grid[i, j] = np.clip(rh, 0.3, 0.98)
    
    return rh_grid


def calculate_ice_water_content(z_bar, T, q_cloud, z_isotherm_223):
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
    # Temperature threshold for ice
    T_freeze = 273.15
    T_ice = 253.15  # All ice below this temperature
    
    # Ice fraction increases as temperature decreases
    ice_fraction = np.clip((T_freeze - T) / (T_freeze - T_ice), 0, 1)
    
    # Ice water content
    q_ice = q_cloud * ice_fraction
    
    return q_ice
