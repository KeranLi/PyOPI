"""
Oxygen Isotope Fractionation

<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
<<<<<<< HEAD:OPI_python/opi/fractionation_oxygen.py
Calculates the fractionation factor for oxygen isotopes (delta^18O)
based on the Mixed Cloud Isotope Model (MCIM) of Ciais and Jouzel (1994).

Similar to hydrogen fractionation but with different coefficients.

Reference: Ciais and Jouzel, 1994
=======
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
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
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
>>>>>>> dev:opi/fractionation_oxygen.py
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
"""

import numpy as np

<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
<<<<<<< HEAD:OPI_python/opi/fractionation_oxygen.py

def fractionation_oxygen(T, h_r=1.0, is_kinetic=True):
=======
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
# Kelvin to Celsius conversion
TC2K = 273.15


def fractionation_oxygen(T):
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
>>>>>>> dev:opi/fractionation_oxygen.py
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
    """
    Calculate oxygen isotope fractionation factor.
    
    Parameters
    ----------
    T : float or ndarray
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
<<<<<<< HEAD:OPI_python/opi/fractionation_oxygen.py
        Temperature in Kelvin. Can be a scalar or array.
    h_r : float or ndarray, optional
        Relative humidity (0 to 1). Default is 1.0 (saturated).
    is_kinetic : bool, optional
        Whether to include kinetic fractionation. Default is True.
=======
        Temperature in Kelvin. Can be scalar, vector, or array.
>>>>>>> dev:opi/fractionation_oxygen.py
=======
        Temperature in Kelvin. Can be scalar, vector, or array.
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (R_precip / R_vapor).
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
<<<<<<< HEAD:OPI_python/opi/fractionation_oxygen.py
    
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
=======
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
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
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
>>>>>>> dev:opi/fractionation_oxygen.py
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
    
    return alpha


<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
<<<<<<< HEAD:OPI_python/opi/fractionation_oxygen.py
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
=======
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
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
<<<<<<< HEAD:OPI_python/OPI_python/opi/fractionation_oxygen.py
>>>>>>> dev:opi/fractionation_oxygen.py
=======
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/fractionation_oxygen.py
    
    print("\nTest completed successfully!")
