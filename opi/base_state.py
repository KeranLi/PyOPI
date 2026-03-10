"""
Base state calculation for the atmosphere in the absence of topography
"""

import numpy as np
from scipy.optimize import fsolve
from .saturated_vapor_pressure import saturated_vapor_pressure
from .constants import G, CPD, CPV, RD, L, P0, EPSILON


def base_state(NM, T0, z_max=12e3, dz=100):
    """
    Calculate base state atmospheric properties without topography
    
    Parameters:
    -----------
    NM : float
        Saturated buoyancy frequency (rad/s)
    T0 : float
        Sea-level temperature (K)
    z_max : float, optional
        Maximum elevation (m), default is 12 km
    dz : float, optional
        Elevation increment (m), default is 100 m
    
    Returns:
    --------
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
        # Vertical derivative of saturation mixing ratio (Durran and Klemp, 1982, eq. 12)
        d_rs_dz = r_s[i] * (1 + r_s[i] / EPSILON) / (RD * T[i]) * (
            -EPSILON * L * gamma_env_est / T[i] +
            G * (1 + r_s[i]) / (1 + r_s[i] / EPSILON)
        )
        
        # Saturated buoyancy frequency using DK82, eq 5, with rT = rS
        ns2_est = (G / T[i]) * (gamma_sat[i] - gamma_env_est) * (
            1 + L * r_s[i] / (RD * T[i])
        ) - G / (1 + r_s[i]) * d_rs_dz
        
        diff = NM**2 - ns2_est
        return diff

    # Calculate profile by integrating upward from sea level
    for i in range(n):
        # Partial pressure and mixing ratio for saturated water vapor
        e_s = saturated_vapor_pressure(T[i])
        r_s[i] = EPSILON * e_s / (p[i] - e_s)
        
        # Saturated adiabatic lapse rate, DK82, eq. 19, with 2`Q
        gamma_sat[i] = G * (1 + r_s[i]) * (1 + L * r_s[i] / (RD * T[i])) / (
            CPD + CPV * r_s[i] + L**2 * r_s[i] * (EPSILON + r_s[i]) / (RD * T[i]**2)
        )
        
        # Use fsolve to get gamma_env using eq. 5 from Durran and Klemp, 1982
        # Start with guess based on eq. 3 from Durran and Klemp, 1982
        gamma_env_guess = gamma_sat[i] - NM**2 * T[i] / G
        gamma_env[i] = fsolve(lambda g: f_diff(g, i), gamma_env_guess)[0]
        
        # Total density, eq. 3.15 in Wallace and Hobbs, 2006
        rho[i] = p[i] * (1 + r_s[i]) / (RD * T[i] * (1 + r_s[i] / EPSILON))
        
        # Forward difference for temperature, pressure, and saturated adiabat at next elevation
        if i < n - 1:
            T[i + 1] = T[i] - gamma_env[i] * dz
            d_ln_p_dz = -G * rho[i] / p[i]
            p[i + 1] = p[i] * np.exp(d_ln_p_dz * dz)

    # Check gamma_env
    if np.any(gamma_env < 1e-3):
        raise ValueError(
            "Not allowed: Selected NM and T0 for the base state "
            "result in local regions where gammaEnv < 1 C/km."
        )

    # Calculate water-vapor density
    rho_s = p * r_s / (RD * T * (1 + r_s / EPSILON))

    # Estimate gamma_ratio, which is the mean of the
    # ratio of the saturated lapse rate over the environmental lapse rate.
    # The mean is weighted proportional to the saturated-vapor density.
    weights = rho_s / np.sum(rho_s)
    gamma_ratio = np.sum(weights * gamma_sat / gamma_env)

    # Estimate rho_s0 and h_s for vertical water-vapor density.
    # These parameters are estimated using an exponential fit, with
    # weighting set proportional to the saturated-vapor density, and
    # to account for the log transformation used for the fit.
    A = np.column_stack([np.ones(n), z_bar])
    weights = rho_s**2 / np.sum(rho_s**2)
    W = np.diag(np.sqrt(weights))
    Aw = W @ A
    bw = W @ np.log(rho_s)
    b = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    rho_s0 = np.exp(b[0])
    h_s = -1 / b[1]

    # Exponential fit to estimate rho0 and h_rho for total density
    # Total density is based on eq. 3.15 in Wallace and Hobbs, 2006.
    # Weights account for log transformation used for the fit.
    rho_total = p * (1 + r_s) / (RD * T * (1 + r_s / EPSILON))
    A = np.column_stack([np.ones(n), z_bar])
    weights = rho_total / np.sum(rho_total)
    W = np.diag(np.sqrt(weights))
    Aw = W @ A
    bw = W @ np.log(rho_total)
    b = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    rho0 = np.exp(b[0])
    h_rho = -1 / b[1]

    return z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho