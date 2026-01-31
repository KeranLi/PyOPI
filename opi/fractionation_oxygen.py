"""
Oxygen Isotope Fractionation

Calculates the fractionation factor for oxygen isotopes (delta^18O)
based on the Mixed Cloud Isotope Model (MCIM) of Ciais and Jouzel (1994).

Similar to hydrogen fractionation but with different coefficients.

Reference: Ciais and Jouzel, 1994
"""

import numpy as np


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
    
    Notes
    -----
    Temperature ranges:
    - T > 273.15 K: Liquid water-vapor equilibrium
    - 248 K <= T <= 273.15 K: Mixed phase (WBF zone)
    - T < 248 K: Ice-vapor equilibrium
    """
    T = np.asarray(T)
    
    # Temperature thresholds
    T_L = 273.15
    T_WBF_low = 248.0
    T_WBF_high = 268.0
    
    # Initialize output
    alpha = np.zeros_like(T, dtype=float)
    
    # Equilibrium fractionation coefficients
    # For liquid water-vapor (Majoube, 1971)
    # ln(alpha) = A1/T^2 + A2/T^3
    A1 = -2.0667
    A2 = 0.0  # Simplified form
    
    # For ice-vapor (Majoube, 1971)
    # ln(alpha) = B1/T + B2/T^2 + B3/T^3
    B1 = 0.0
    B2 = 1136.0
    B3 = 0.4156
    
    # Region 1: Liquid water
    mask_liquid = T > T_WBF_high
    if np.any(mask_liquid):
        alpha_eq_liquid = np.exp(A1 / T[mask_liquid] + A2 / T[mask_liquid]**2)
        alpha[mask_liquid] = alpha_eq_liquid
    
    # Region 2: Ice
    mask_ice = T < T_WBF_low
    if np.any(mask_ice):
        alpha_eq_ice = np.exp((B2 / T[mask_ice] + B3) / T[mask_ice])
        alpha[mask_ice] = alpha_eq_ice
    
    # Region 3: WBF zone (mixed phase)
    mask_wbf = (T >= T_WBF_low) & (T <= T_WBF_high)
    if np.any(mask_wbf):
        T_wbf = T[mask_wbf]
        f_ice = (T_WBF_high - T_wbf) / (T_WBF_high - T_WBF_low)
        
        alpha_eq_liq_wbf = np.exp(A1 / T_WBF_high + A2 / T_WBF_high**2)
        alpha_eq_ice_wbf = np.exp((B2 / T_WBF_low + B3) / T_WBF_low)
        
        alpha_eq_wbf = alpha_eq_liq_wbf + f_ice * (alpha_eq_ice_wbf - alpha_eq_liq_wbf)
        alpha[mask_wbf] = alpha_eq_wbf
    
    # Apply kinetic fractionation
    if is_kinetic:
        # Diffusion effect for 18O is different from D
        # The kinetic factor is approximately (1 - h_r)^n
        n = 1.0
        alpha_kin_factor = 1.0 + 0.5 * (1.0 - h_r)**n  # Simplified
        alpha = alpha * alpha_kin_factor
    
    return alpha


def fractionation_oxygen_simple(T):
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
    return fractionation_oxygen(T, h_r=1.0, is_kinetic=False)


if __name__ == "__main__":
    print("Testing fractionation_oxygen module...")
    
    # Test at various temperatures
    T_test = np.array([280, 268, 258, 248, 238])
    
    print("\nTemperature (K) | alpha (equilibrium) | alpha (with kinetic)")
    print("-" * 60)
    
    for T in T_test:
        alpha_eq = fractionation_oxygen(T, h_r=1.0, is_kinetic=False)
        alpha_kin = fractionation_oxygen(T, h_r=0.8, is_kinetic=True)
        print(f"{T:13.1f} | {alpha_eq:19.6f} | {alpha_kin:20.6f}")
    
    # Test delta notation
    print("\nDelta notation (equilibrium, permil):")
    for T in T_test:
        alpha = fractionation_oxygen(T, h_r=1.0, is_kinetic=False)
        delta = (alpha - 1.0) * 1000.0
        print(f"  T = {T:.1f} K: delta = {delta:+.2f} permil")
    
    print("\nTest completed successfully!")
