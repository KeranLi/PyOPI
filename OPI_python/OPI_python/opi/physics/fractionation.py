"""
Isotope Fractionation

Based on MATLAB OPI implementation using:
- Merlivat and Nief (1967) for ice-vapor and water-vapor (258-273 K)
- Majoube (1971) for water-vapor (T >= 273.15 K)
- Majoube (1970) for ice-vapor (oxygen)
- Ciais and Jouzel (1994) for kinetic fractionation and WBF zone

All formulas return alpha = R_condensate / R_vapor (> 1)
"""

import numpy as np


def fractionation_hydrogen(T):
    """
    Calculate hydrogen isotope fractionation factor (liquid or ice / vapor).
    
    Matches MATLAB fractionationHydrogen.m exactly.
    Returns alpha = R_precip / R_vapor (> 1)
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (always > 1)
    """
    T = np.asarray(T, dtype=float)
    TC2K = 273.15
    
    # Ice-vapor calculation (Merlivat and Nief, 1967)
    bivp3 = 0
    bivp2 = 0
    bivp1 = 0
    biv0 = -9.45e-2
    bivm1 = 0
    bivm2 = 16289
    bivm3 = 0
    
    alpha_iv = np.exp(
        bivp3 * T**3 + bivp2 * T**2 + bivp1 * T
        + biv0
        + bivm1 * T**-1 + bivm2 * T**-2 + bivm3 * T**-3
    )
    
    # Kinetic effect for ice-vapor (Ciais and Jouzel, 1994)
    sI = 1.02 - 0.0038 * (T - TC2K)
    diffusivity_ratio = 0.9755
    alpha_iv = alpha_iv * sI / ((alpha_iv * (sI - 1) / diffusivity_ratio) + 1)
    
    # Water-vapor calculation (Majoube, 1971) for T >= 273.15 K
    blvp3 = 0
    blvp2 = 0
    blvp1 = 0
    blv0 = 52.612e-3
    blvm1 = -76.248
    blvm2 = 24.844e3
    blvm3 = 0
    
    alpha_lv_m71 = np.exp(
        blvp3 * T**3 + blvp2 * T**2 + blvp1 * T
        + blv0
        + blvm1 * T**-1 + blvm2 * T**-2 + blvm3 * T**-3
    )
    
    # Water-vapor calculation (Merlivat and Nief, 1967) for 258-273 K
    blvp3 = 0
    blvp2 = 0
    blvp1 = 0
    blv0 = -10.0e-2
    blvm1 = 0
    blvm2 = 15013
    blvm3 = 0
    
    alpha_lv_mn67 = np.exp(
        blvp3 * T**3 + blvp2 * T**2 + blvp1 * T
        + blv0
        + blvm1 * T**-1 + blvm2 * T**-2 + blvm3 * T**-3
    )
    
    # Merge calibration results at T = 273.15 K
    i_switch = (T >= TC2K)
    alpha_lv = i_switch * alpha_lv_m71 + ~i_switch * alpha_lv_mn67
    
    # Combine results for mixed phase (WBF zone: 248-268 K)
    factor = (T - 248) / (268 - 248)
    factor = np.clip(factor, 0, 1)
    
    alpha = alpha_lv * factor + alpha_iv * (1 - factor)
    
    return alpha


def fractionation_oxygen(T):
    """
    Calculate oxygen isotope fractionation factor (liquid or ice / vapor).
    
    Matches MATLAB fractionationOxygen.m exactly.
    Returns alpha = R_precip / R_vapor (> 1)
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (always > 1)
    """
    T = np.asarray(T, dtype=float)
    TC2K = 273.15
    
    # Ice-vapor (Majoube, 1970, Nature)
    bivp3 = 0
    bivp2 = 0
    bivp1 = 0
    biv0 = -28.224e-3
    bivm1 = 11.839
    bivm2 = 0
    bivm3 = 0
    
    alpha_iv = np.exp(
        bivp3 * T**3 + bivp2 * T**2 + bivp1 * T
        + biv0
        + bivm1 * T**-1 + bivm2 * T**-2 + bivm3 * T**-3
    )
    
    # Kinetic effect for ice-vapor
    sI = 1.02 - 0.0038 * (T - TC2K)
    diffusivity_ratio = 0.9723
    alpha_iv = alpha_iv * sI / ((alpha_iv * (sI - 1) / diffusivity_ratio) + 1)
    
    # Water-vapor (Majoube, 1971) for T >= 273.15 K
    blvp3 = 0
    blvp2 = 0
    blvp1 = 0
    blv0 = -2.0667e-3
    blvm1 = -0.4156
    blvm2 = 1.137e3
    blvm3 = 0
    
    alpha_lv = np.exp(
        blvp3 * T**3 + blvp2 * T**2 + blvp1 * T
        + blv0
        + blvm1 * T**-1 + blvm2 * T**-2 + blvm3 * T**-3
    )
    
    # Combine results for mixed phase (WBF zone)
    factor = (T - 248) / (268 - 248)
    factor = np.clip(factor, 0, 1)
    
    alpha = alpha_lv * factor + alpha_iv * (1 - factor)
    
    return alpha


# Keep simple versions for backward compatibility
fractionation_hydrogen_simple = fractionation_hydrogen
fractionation_oxygen_simple = fractionation_oxygen
