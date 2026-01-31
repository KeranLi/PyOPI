"""
Isotope Fractionation

Calculates the fractionation factors for hydrogen (delta^2H) and oxygen (delta^18O) 
isotopes based on the Mixed Cloud Isotope Model (MCIM) of Ciais and Jouzel (1994).

The model considers:
1. Equilibrium fractionation (water-vapor and ice-vapor)
2. Kinetic fractionation (diffusion effects)
3. Mixed-phase region (WBF zone: 248-268 K)

Reference: Ciais and Jouzel, 1994
"""

import numpy as np


# Temperature thresholds
T_L = 273.15         # Liquid water freezing point
T_WBF_LOW = 248.0    # Lower bound of WBF zone
T_WBF_HIGH = 268.0   # Upper bound of WBF zone


def fractionation_hydrogen(T, h_r=1.0, is_kinetic=True):
    """
    Calculate hydrogen isotope fractionation factor.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin. Can be a scalar or array.
    h_r : float or ndarray, optional
        Relative humidity (0 to 1). Default is 1.0 (saturated).
    is_kinetic : bool, optional
        Whether to include kinetic fractionation. Default is True.
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (R_precip / R_vapor).
        For delta notation: delta = (alpha - 1) * 1000 permil
    """
    T = np.asarray(T)
    alpha = np.zeros_like(T, dtype=float)
    
    # Equilibrium fractionation coefficients (Majoube, 1971)
    # ln(alpha) = A1/T^2 + A2/T^3 for liquid
    A1 = -52.612
    A2 = 6282.0
    
    # ln(alpha) = B2/T^2 + B3/T^3 for ice
    B2 = 16288.0
    B3 = 0.0930
    
    # Region 1: T > T_WBF_HIGH (liquid water)
    mask_liquid = T > T_WBF_HIGH
    if np.any(mask_liquid):
        alpha[mask_liquid] = np.exp((A2 / T[mask_liquid] + A1) / T[mask_liquid])
    
    # Region 2: T < T_WBF_LOW (ice)
    mask_ice = T < T_WBF_LOW
    if np.any(mask_ice):
        alpha[mask_ice] = np.exp((B2 / T[mask_ice] + B3) / T[mask_ice])
    
    # Region 3: WBF zone (mixed phase)
    mask_wbf = (T >= T_WBF_LOW) & (T <= T_WBF_HIGH)
    if np.any(mask_wbf):
        T_wbf = T[mask_wbf]
        f_ice = (T_WBF_HIGH - T_wbf) / (T_WBF_HIGH - T_WBF_LOW)
        alpha_eq_liq = np.exp((A2 / T_WBF_HIGH + A1) / T_WBF_HIGH)
        alpha_eq_ice = np.exp((B2 / T_WBF_LOW + B3) / T_WBF_LOW)
        alpha[mask_wbf] = alpha_eq_liq + f_ice * (alpha_eq_ice - alpha_eq_liq)
    
    # Apply kinetic fractionation
    if is_kinetic:
        diff_ratio = 0.975  # D_diff / 18O_diff (Merlivat, 1978)
        n = 1.0  # For turbulent conditions
        alpha_kin_factor = 1.0 + (diff_ratio**n - 1.0) * (1.0 - h_r)**n
        alpha = alpha * alpha_kin_factor
    
    return alpha


def fractionation_oxygen(T, h_r=1.0, is_kinetic=True):
    """
    Calculate oxygen isotope fractionation factor.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin. Can be a scalar or array.
    h_r : float or ndarray, optional
        Relative humidity (0 to 1). Default is 1.0 (saturated).
    is_kinetic : bool, optional
        Whether to include kinetic fractionation. Default is True.
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (R_precip / R_vapor).
    """
    T = np.asarray(T)
    alpha = np.zeros_like(T, dtype=float)
    
    # Equilibrium fractionation coefficients (Majoube, 1971)
    # ln(alpha) = A1/T^2 + A2/T^3 for liquid
    A1 = -2.0667
    A2 = 0.0
    
    # ln(alpha) = B2/T^2 + B3/T^3 for ice
    B2 = 1136.0
    B3 = 0.4156
    
    # Region 1: Liquid water
    mask_liquid = T > T_WBF_HIGH
    if np.any(mask_liquid):
        alpha[mask_liquid] = np.exp(A1 / T[mask_liquid] + A2 / T[mask_liquid]**2)
    
    # Region 2: Ice
    mask_ice = T < T_WBF_LOW
    if np.any(mask_ice):
        alpha[mask_ice] = np.exp((B2 / T[mask_ice] + B3) / T[mask_ice])
    
    # Region 3: WBF zone (mixed phase)
    mask_wbf = (T >= T_WBF_LOW) & (T <= T_WBF_HIGH)
    if np.any(mask_wbf):
        T_wbf = T[mask_wbf]
        f_ice = (T_WBF_HIGH - T_wbf) / (T_WBF_HIGH - T_WBF_LOW)
        alpha_eq_liq = np.exp(A1 / T_WBF_HIGH + A2 / T_WBF_HIGH**2)
        alpha_eq_ice = np.exp((B2 / T_WBF_LOW + B3) / T_WBF_LOW)
        alpha[mask_wbf] = alpha_eq_liq + f_ice * (alpha_eq_ice - alpha_eq_liq)
    
    # Apply kinetic fractionation
    if is_kinetic:
        n = 1.0
        alpha_kin_factor = 1.0 + 0.5 * (1.0 - h_r)**n
        alpha = alpha * alpha_kin_factor
    
    return alpha


def fractionation_hydrogen_simple(T):
    """Simplified hydrogen fractionation without kinetic effects."""
    return fractionation_hydrogen(T, h_r=1.0, is_kinetic=False)


def fractionation_oxygen_simple(T):
    """Simplified oxygen fractionation without kinetic effects."""
    return fractionation_oxygen(T, h_r=1.0, is_kinetic=False)
