"""
Thermodynamic calculations for atmospheric processes.

Includes:
- Saturated vapor pressure (Tetens equation)
- Base state atmospheric profiles
"""

import numpy as np
from scipy.optimize import fsolve


def saturated_vapor_pressure(temperature):
    """
    Calculate saturated vapor pressure using Tetens equation.
    
    Parameters
    ----------
    temperature : float or array-like
        Temperature in Kelvin
    
    Returns
    -------
    e_s : float or ndarray
        Saturated vapor pressure in Pa
    """
    temp_celsius = temperature - 273.15
    e_s = 6.1078 * np.power(10.0, (7.5 * temp_celsius) / (237.3 + temp_celsius))
    return e_s * 100.0  # Convert hPa to Pa


def base_state(NM, T0, z_max=12e3, dz=100, 
               G=9.81, CPD=1004.0, CPV=1885.0, RD=287.0, 
               L=2.5e6, P0=1e5, EPSILON=0.622):
    """
    Calculate base state atmospheric properties without topography.
    
    Reference: Durran and Klemp, 1982
    
    Parameters
    ----------
    NM : float
        Saturated buoyancy frequency (rad/s)
    T0 : float
        Sea-level temperature (K)
    z_max : float, optional
        Maximum elevation (m), default is 12 km
    dz : float, optional
        Elevation increment (m), default is 100 m
    G, CPD, CPV, RD, L, P0, EPSILON : float
        Physical constants
    
    Returns
    -------
    z_bar : ndarray
        Elevation relative to sea level (m)
    T : ndarray
        Environmental temperature (K)
    gamma_env : ndarray
        Environmental lapse rate (K/m)
    gamma_sat : ndarray
        Saturation lapse rate (K/m)
    gamma_ratio : float
        Weighted mean of gamma_sat/gamma_env
    rho_s0 : float
        Water-vapor density at z=0 (kg/m^3)
    h_s : float
        Scale height for saturated vapor (m)
    rho0 : float
        Total air density at z=0 (kg/m^3)
    h_rho : float
        Scale height for total air density (m)
    """
    n = int(np.round(z_max / dz)) + 1
    z_bar = np.arange(n) * dz
    T = np.zeros(n)
    T[0] = T0
    p = np.zeros(n)
    p[0] = P0
    r_s = np.zeros(n)
    gamma_env = np.zeros(n)
    gamma_sat = np.zeros(n)
    rho = np.zeros(n)

    def f_diff(gamma_env_est, i):
        """Calculate difference for gamma_env estimation"""
        d_rs_dz = r_s[i] * (1 + r_s[i] / EPSILON) / (RD * T[i]) * (
            -EPSILON * L * gamma_env_est / T[i] +
            G * (1 + r_s[i]) / (1 + r_s[i] / EPSILON)
        )
        
        ns2_est = (G / T[i]) * (gamma_sat[i] - gamma_env_est) * (
            1 + L * r_s[i] / (RD * T[i])
        ) - G / (1 + r_s[i]) * d_rs_dz
        
        return NM**2 - ns2_est

    # Calculate profile by integrating upward from sea level
    for i in range(n):
        e_s = saturated_vapor_pressure(T[i])
        r_s[i] = EPSILON * e_s / (p[i] - e_s)
        
        # Saturated adiabatic lapse rate
        gamma_sat[i] = G * (1 + r_s[i]) * (1 + L * r_s[i] / (RD * T[i])) / (
            CPD + CPV * r_s[i] + L**2 * r_s[i] * (EPSILON + r_s[i]) / (RD * T[i]**2)
        )
        
        # Use fsolve to get gamma_env
        gamma_env_guess = gamma_sat[i] - NM**2 * T[i] / G
        gamma_env[i] = fsolve(lambda g: f_diff(g, i), gamma_env_guess)[0]
        
        # Total density
        rho[i] = p[i] * (1 + r_s[i]) / (RD * T[i] * (1 + r_s[i] / EPSILON))
        
        # Forward difference for next level
        if i < n - 1:
            T[i + 1] = T[i] - gamma_env[i] * dz
            d_ln_p_dz = -G * rho[i] / p[i]
            p[i + 1] = p[i] * np.exp(d_ln_p_dz * dz)

    # Check gamma_env
    if np.any(gamma_env < 1e-3):
        raise ValueError(
            "Selected NM and T0 result in regions where gammaEnv < 1 C/km"
        )

    # Calculate water-vapor density
    rho_s = p * r_s / (RD * T * (1 + r_s / EPSILON))

    # Weighted mean of gamma_sat/gamma_env
    weights = rho_s / np.sum(rho_s)
    gamma_ratio = np.sum(weights * gamma_sat / gamma_env)

    # Exponential fit for rho_s0 and h_s
    A = np.column_stack([np.ones(n), z_bar])
    weights = rho_s**2 / np.sum(rho_s**2)
    W = np.diag(np.sqrt(weights))
    b = np.linalg.lstsq(W @ A, W @ np.log(rho_s), rcond=None)[0]
    rho_s0 = np.exp(b[0])
    h_s = -1 / b[1]

    # Exponential fit for rho0 and h_rho
    rho_total = p * (1 + r_s) / (RD * T * (1 + r_s / EPSILON))
    weights = rho_total / np.sum(rho_total)
    W = np.diag(np.sqrt(weights))
    b = np.linalg.lstsq(W @ A, W @ np.log(rho_total), rcond=None)[0]
    rho0 = np.exp(b[0])
    h_rho = -1 / b[1]

    return z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho
