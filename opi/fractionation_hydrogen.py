"""
Hydrogen Isotope Fractionation

Calculates the fractionation factor for hydrogen isotopes (delta^2H)
based on the Mixed Cloud Isotope Model (MCIM) of Ciais and Jouzel (1994).

The model considers:
1. Equilibrium fractionation (water-vapor and ice-vapor)
2. Kinetic fractionation (diffusion effects)
3. Mixed-phase region (WBF zone: 248-268 K)

Reference: Ciais and Jouzel, 1994
"""

import numpy as np


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
    
    Notes
    -----
    The fractionation factor alpha relates the isotope ratio in precipitation
    to that in vapor: R_precip = alpha * R_vapor
    
    Temperature ranges:
    - T > 273.15 K: Liquid water-vapor equilibrium
    - 248 K <= T <= 273.15 K: Mixed phase (WBF zone)
    - T < 248 K: Ice-vapor equilibrium
    """
    T = np.asarray(T)
    
    # Temperature thresholds
    T_L = 273.15  # Liquid water freezing point
    T_WBF_low = 248.0  # Lower bound of WBF zone
    T_WBF_high = 268.0  # Upper bound of WBF zone (commonly used)
    
    # Initialize output array
    alpha = np.zeros_like(T, dtype=float)
    
    # Equilibrium fractionation coefficients
    # For liquid water-vapor (Majoube, 1971)
    # ln(alpha) = A1/T^2 + A2/T^3 + A3/T^4
    A1 = -52.612
    A2 = 6282.0
    A3 = 0.0  # Not used in simplified form
    
    # For ice-vapor (Majoube, 1971)
    # ln(alpha) = B1/T + B2/T^2 + B3/T^3
    B1 = 0.0  # Not used
    B2 = 16288.0
    B3 = 0.0930
    
    # Calculate equilibrium fractionation
    # Region 1: T > T_WBF_high (liquid water)
    mask_liquid = T > T_WBF_high
    if np.any(mask_liquid):
        alpha_eq_liquid = np.exp((A2 / T[mask_liquid] + A1) / T[mask_liquid])
        alpha[mask_liquid] = alpha_eq_liquid
    
    # Region 2: T < T_WBF_low (ice)
    mask_ice = T < T_WBF_low
    if np.any(mask_ice):
        alpha_eq_ice = np.exp((B2 / T[mask_ice] + B3) / T[mask_ice])
        alpha[mask_ice] = alpha_eq_ice
    
    # Region 3: T_WBF_low <= T <= T_WBF_high (WBF zone - mixed phase)
    mask_wbf = (T >= T_WBF_low) & (T <= T_WBF_high)
    if np.any(mask_wbf):
        T_wbf = T[mask_wbf]
        # Calculate fraction at this temperature
        f_ice = (T_WBF_high - T_wbf) / (T_WBF_high - T_WBF_low)
        
        # Equilibrium fractionation at boundaries
        alpha_eq_liq_wbf = np.exp((A2 / T_WBF_high + A1) / T_WBF_high)
        alpha_eq_ice_wbf = np.exp((B2 / T_WBF_low + B3) / T_WBF_low)
        
        # Linear interpolation in WBF zone
        alpha_eq_wbf = alpha_eq_liq_wbf + f_ice * (alpha_eq_ice_wbf - alpha_eq_liq_wbf)
        alpha[mask_wbf] = alpha_eq_wbf
    
    # Apply kinetic fractionation if requested
    if is_kinetic:
        # Diffusion ratio for D/H relative to 18O/16O
        # D_diff / 18O_diff ~ 0.975 (Merlivat, 1978)
        diff_ratio = 0.975
        
        # Kinetic fractionation exponent
        n = 1.0  # For turbulent conditions
        
        # Kinetic fractionation factor
        # alpha_kin = alpha_eq * (1 + (diff_ratio - 1) * (1 - h_r)^n)
        # Simplified: alpha_kin = alpha_eq * diff_ratio^n for undersaturated
        alpha_kin_factor = 1.0 + (diff_ratio**n - 1.0) * (1.0 - h_r)**n
        alpha = alpha * alpha_kin_factor
    
    return alpha


def fractionation_hydrogen_simple(T):
    """
    Simplified version without kinetic effects.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
    
    Returns
    -------
    alpha : float or ndarray
        Equilibrium fractionation factor
    """
    return fractionation_hydrogen(T, h_r=1.0, is_kinetic=False)


if __name__ == "__main__":
    print("Testing fractionation_hydrogen module...")
    
    # Test at various temperatures
    T_test = np.array([280, 268, 258, 248, 238])
    
    print("\nTemperature (K) | alpha (equilibrium) | alpha (with kinetic)")
    print("-" * 60)
    
    for T in T_test:
        alpha_eq = fractionation_hydrogen(T, h_r=1.0, is_kinetic=False)
        alpha_kin = fractionation_hydrogen(T, h_r=0.8, is_kinetic=True)
        print(f"{T:13.1f} | {alpha_eq:19.6f} | {alpha_kin:20.6f}")
    
    # Test delta notation
    print("\nDelta notation (equilibrium, permil):")
    for T in T_test:
        alpha = fractionation_hydrogen(T, h_r=1.0, is_kinetic=False)
        delta = (alpha - 1.0) * 1000.0
        print(f"  T = {T:.1f} K: delta = {delta:+.1f} permil")
    
    print("\nTest completed successfully!")
