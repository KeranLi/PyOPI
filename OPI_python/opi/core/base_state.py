"""
Atmospheric Base State Calculations

This module calculates the vertical structure of the atmosphere
in the absence of topography, following Durran and Klemp (1982).
"""

import numpy as np
from scipy.optimize import fsolve
from ..constants import G, CPD, CPV, RD, L, P0, EPSILON


def saturated_vapor_pressure(T):
    """
    Calculate saturated vapor pressure over water using the Clausius-Clapeyron equation.
    
    Uses the Magnus formula for accurate water vapor pressure calculation
    over a wide temperature range.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    float or ndarray
        Saturated vapor pressure in Pa
        
    References
    ----------
    Buck (1981) - New equations for computing vapor pressure
    """
    # Convert to Celsius for the formula
    TC = T - 273.15
    
    # Buck (1981) formula for saturation vapor pressure over water
    # Valid for -80°C to 50°C
    e_s = 6.1121 * np.exp((18.678 - TC/234.5) * TC / (257.14 + TC))  # hPa
    
    return e_s * 100  # Convert to Pa


def base_state(NM, T0, z_max=12e3, dz=100):
    """
    Calculate atmospheric base state properties without topography.
    
    The base state represents a saturated atmosphere with a specified 
    surface temperature and constant buoyancy frequency. This calculation
    solves for the vertical structure using the anelastic equations of
    Durran and Klemp (1982).
    
    Parameters
    ----------
    NM : float
        Saturated buoyancy frequency (rad/s)
    T0 : float
        Sea-level temperature (K)
    z_max : float, optional
        Maximum elevation (m), default 12 km
    dz : float, optional
        Vertical resolution (m), default 100 m
        
    Returns
    -------
    dict : Dictionary containing base state variables:
        - 'z_bar' : Elevation relative to sea level (m)
        - 'T' : Environmental temperature (K)
        - 'gamma_env' : Environmental lapse rate (K/m)
        - 'gamma_sat' : Saturation lapse rate (K/m)
        - 'gamma_ratio' : Weighted mean of gamma_sat/gamma_env
        - 'rho_s0' : Water-vapor density at z=0 (kg/m³)
        - 'h_s' : Scale height for saturated vapor (m)
        - 'rho0' : Total air density at z=0 (kg/m³)
        - 'h_rho' : Scale height for total air density (m)
        - 'p' : Pressure profile (Pa)
        - 'r_s' : Saturation mixing ratio
        
    Raises
    ------
    ValueError
        If gamma_env < 1e-3 K/m (unstable atmosphere)
        
    Notes
    -----
    The water-vapor density profile is characterized by an exponential
    relationship: rho_s(z) = rho_s0 * exp(-z/h_s)
    
    The total air density profile follows:
    rho(z) = rho0 * exp(-z/h_rho)
    
    References
    ----------
    Durran and Klemp (1982) - On the effects of moisture on the 
        Brunt-Väisälä frequency
    Wallace and Hobbs (2006) - Atmospheric Science: An Introductory Survey
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
        """
        Calculate the difference between target and estimated buoyancy frequency.
        
        This is the residual function for solving gamma_env using the
        relationship from Durran and Klemp (1982), equation 5.
        """
        # Vertical derivative of saturation mixing ratio (DK82, eq. 12)
        d_rs_dz = r_s[i] * (1 + r_s[i] / EPSILON) / (RD * T[i]) * (
            -EPSILON * L * gamma_env_est / T[i] +
            G * (1 + r_s[i]) / (1 + r_s[i] / EPSILON)
        )
        
        # Saturated buoyancy frequency using DK82, eq 5, with rT = rS
        ns2_est = (G / T[i]) * (gamma_sat[i] - gamma_env_est) * (
            1 + L * r_s[i] / (RD * T[i])
        ) - G / (1 + r_s[i]) * d_rs_dz
        
        return NM**2 - ns2_est

    # Calculate profile by integrating upward from sea level
    for i in range(n):
        # Partial pressure and mixing ratio for saturated water vapor
        e_s = saturated_vapor_pressure(T[i])
        r_s[i] = EPSILON * e_s / (p[i] - e_s)
        
        # Saturated adiabatic lapse rate, DK82, eq. 19
        gamma_sat[i] = G * (1 + r_s[i]) * (1 + L * r_s[i] / (RD * T[i])) / (
            CPD + CPV * r_s[i] + 
            L**2 * r_s[i] * (EPSILON + r_s[i]) / (RD * T[i]**2)
        )
        
        # Solve for gamma_env using equation 5 from Durran and Klemp, 1982
        gamma_env_guess = gamma_sat[i] - NM**2 * T[i] / G
        gamma_env[i] = fsolve(lambda g: f_diff(g, i), gamma_env_guess)[0]
        
        # Total density, eq. 3.15 in Wallace and Hobbs (2006)
        rho[i] = p[i] * (1 + r_s[i]) / (RD * T[i] * (1 + r_s[i] / EPSILON))
        
        # Forward difference for next elevation
        if i < n - 1:
            T[i + 1] = T[i] - gamma_env[i] * dz
            d_ln_p_dz = -G * rho[i] / p[i]
            p[i + 1] = p[i] * np.exp(d_ln_p_dz * dz)

    # Check stability
    if np.any(gamma_env < 1e-3):
        raise ValueError(
            f"Selected NM={NM*1000:.2f} mrad/s and T0={T0:.1f} K result in "
            "regions where gamma_env < 1 C/km. Atmosphere is too unstable."
        )

    # Calculate water-vapor density
    rho_s = p * r_s / (RD * T * (1 + r_s / EPSILON))

    # Weighted mean lapse rate ratio
    weights = rho_s / np.sum(rho_s)
    gamma_ratio = np.sum(weights * gamma_sat / gamma_env)

    # Exponential fit for water-vapor density parameters
    A = np.column_stack([np.ones(n), z_bar])
    weights = rho_s**2 / np.sum(rho_s**2)
    W = np.diag(np.sqrt(weights))
    Aw = W @ A
    bw = W @ np.log(rho_s)
    b = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    rho_s0 = np.exp(b[0])
    h_s = -1 / b[1]

    # Exponential fit for total density parameters
    rho_total = p * (1 + r_s) / (RD * T * (1 + r_s / EPSILON))
    weights = rho_total / np.sum(rho_total)
    W = np.diag(np.sqrt(weights))
    Aw = W @ A
    bw = W @ np.log(rho_total)
    b = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    rho0 = np.exp(b[0])
    h_rho = -1 / b[1]

    return {
        'z_bar': z_bar,
        'T': T,
        'p': p,
        'r_s': r_s,
        'gamma_env': gamma_env,
        'gamma_sat': gamma_sat,
        'gamma_ratio': gamma_ratio,
        'rho_s0': rho_s0,
        'h_s': h_s,
        'rho0': rho0,
        'h_rho': h_rho,
        'rho_s': rho_s,
        'rho_total': rho_total
    }
