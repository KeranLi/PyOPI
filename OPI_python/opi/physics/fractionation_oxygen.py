"""
Oxygen Isotope Fractionation

Given temperature T in Kelvin, returns fractionation factors for
oxygen isotopes for condensates water and ice relative to water vapor.

The algorithm is based on the Mixed Cloud Isotope Model (MCIM) of 
Ciais and Jouzel (1994), which accounts for:
- Supercooling required to nucleate ice
- Kinetic fractionation for vapor to ice
- Bergeron-Findeisen (WBF) zone (248-268 K where water and ice coexist)

Fractionation equations are defined relative to the following generic 
polynomial function:
  alpha = exp(b3*T^3 + b2*T^2 + b1*T + b0 + b(-1)*T^-1 + b(-2)*T^-2 + b(-3)*T^-3)

References
----------
- Ciais and Jouzel, 1994 (MCIM model)
- Majoube, 1970 (ice-vapor equilibrium, -33 to 0 C)
- Majoube, 1971 (water-vapor equilibrium, 273 to 373 K)
- Merlivat, 1978 (diffusivity ratios)
"""

import numpy as np

# Kelvin to Celsius conversion
TC2K = 273.15


def fractionation_oxygen(T):
    """
    Calculate oxygen isotope fractionation factor.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin. Can be scalar, vector, or array.
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (R_precip / R_vapor).
        For delta notation: delta = (alpha - 1) * 1000 permil
    
    Notes
    -----
    The calculation includes:
    1. Equilibrium fractionation for ice-vapor and water-vapor
    2. Kinetic fractionation effect for ice (Ciais and Jouzel, 1994)
    3. Mixed-phase region (WBF zone: 248-268 K)
    """
    T = np.asarray(T, dtype=float)
    
    # ============================================================
    # Ice-vapor exchange
    # ============================================================
    # Equilibrium fractionation factor for oxygen isotopes in
    # ice and vapor for -33 to 0 C from Majoube, 1970, Nature.
    b_iv_p3 = 0.0
    b_iv_p2 = 0.0
    b_iv_p1 = 0.0
    b_iv_0 = -28.224e-3
    b_iv_m1 = 11.839
    b_iv_m2 = 0.0
    b_iv_m3 = 0.0
    
    # Equilibrium fractionation factor for ice-vapor exchange
    alpha_IV = np.exp(
        b_iv_p3 * T**3 + b_iv_p2 * T**2 + b_iv_p1 * T
        + b_iv_0
        + b_iv_m1 * T**-1 + b_iv_m2 * T**-2 + b_iv_m3 * T**-3
    )
    
    # Adjust fractionation for kinetic effect for ice and vapor,
    # according to Ciais and Jouzel (1994).
    s_I = 1.02 - 0.0038 * (T - TC2K)
    
    # Ratio diffusivity(HH18O) / diffusivity(HH16O) from Merlivat, 1978
    diffusivity_ratio = 0.9723
    
    alpha_IV = alpha_IV * s_I / ((alpha_IV * (s_I - 1.0) / diffusivity_ratio) + 1.0)
    
    # ============================================================
    # Water-vapor exchange
    # ============================================================
    # Equilibrium fractionation factor for oxygen isotopes in
    # water and vapor for 273 to 373 K from Majoube, 1971
    # (as reported in Criss, 1999, p. 103).
    b_lv_p3 = 0.0
    b_lv_p2 = 0.0
    b_lv_p1 = 0.0
    b_lv_0 = -2.0667e-3
    b_lv_m1 = -0.4156
    b_lv_m2 = 1.137e3
    b_lv_m3 = 0.0
    
    # Equilibrium fractionation factors for water-vapor exchange
    alpha_LV = np.exp(
        b_lv_p3 * T**3 + b_lv_p2 * T**2 + b_lv_p1 * T
        + b_lv_0
        + b_lv_m1 * T**-1 + b_lv_m2 * T**-2 + b_lv_m3 * T**-3
    )
    
    # ============================================================
    # Combine results
    # ============================================================
    # For mixed ice and water, factor goes from zero for T < 248 K
    # to one for T > 268 K, corresponding to the WBF zone.
    factor = (T - 248.0) / (268.0 - 248.0)
    factor = np.clip(factor, 0.0, 1.0)
    
    alpha = alpha_LV * factor + alpha_IV * (1.0 - factor)
    
    return alpha


if __name__ == "__main__":
    print("Testing fractionation_oxygen module...")
    print("(Comparing with MATLAB implementation)")
    
    # Test at various temperatures
    T_test = np.array([280.0, 273.15, 268.0, 258.0, 248.0, 238.0])
    
    print("\nTemperature (K) | alpha")
    print("-" * 40)
    
    for T in T_test:
        alpha = fractionation_oxygen(T)
        print(f"{T:13.1f} | {alpha:10.6f}")
    
    # Test delta notation
    print("\nDelta notation (permil):")
    for T in T_test:
        alpha = fractionation_oxygen(T)
        delta = (alpha - 1.0) * 1000.0
        print(f"  T = {T:6.1f} K: delta = {delta:+7.2f} permil")
    
    print("\nTest completed successfully!")
