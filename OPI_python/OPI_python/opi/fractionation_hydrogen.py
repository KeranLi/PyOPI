"""
Hydrogen Isotope Fractionation

<<<<<<< HEAD:OPI_python/opi/fractionation_hydrogen.py
Calculates the fractionation factor for hydrogen isotopes (delta^2H)
based on the Mixed Cloud Isotope Model (MCIM) of Ciais and Jouzel (1994).

The model considers:
1. Equilibrium fractionation (water-vapor and ice-vapor)
2. Kinetic fractionation (diffusion effects)
3. Mixed-phase region (WBF zone: 248-268 K)

Reference: Ciais and Jouzel, 1994
=======
Given temperature T in Kelvin, returns fractionation factors for
hydrogen isotopes for condensates water and ice relative to water vapor.

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
- Merlivat and Nief, 1967 (ice-vapor equilibrium, -40 to 0 C)
- Majoube, 1971 (water-vapor equilibrium, 273 to 373 K)
- Merlivat, 1978 (diffusivity ratios)
>>>>>>> dev:opi/fractionation_hydrogen.py
"""

import numpy as np

<<<<<<< HEAD:OPI_python/opi/fractionation_hydrogen.py

def fractionation_hydrogen(T, h_r=1.0, is_kinetic=True):
=======
# Kelvin to Celsius conversion
TC2K = 273.15


def fractionation_hydrogen(T):
>>>>>>> dev:opi/fractionation_hydrogen.py
    """
    Calculate hydrogen isotope fractionation factor.
    
    Parameters
    ----------
    T : float or ndarray
<<<<<<< HEAD:OPI_python/opi/fractionation_hydrogen.py
        Temperature in Kelvin. Can be a scalar or array.
    h_r : float or ndarray, optional
        Relative humidity (0 to 1). Default is 1.0 (saturated).
    is_kinetic : bool, optional
        Whether to include kinetic fractionation. Default is True.
=======
        Temperature in Kelvin. Can be scalar, vector, or array.
>>>>>>> dev:opi/fractionation_hydrogen.py
    
    Returns
    -------
    alpha : float or ndarray
        Fractionation factor (R_precip / R_vapor).
        For delta notation: delta = (alpha - 1) * 1000 permil
    
    Notes
    -----
<<<<<<< HEAD:OPI_python/opi/fractionation_hydrogen.py
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
=======
    The calculation includes:
    1. Equilibrium fractionation for ice-vapor and water-vapor
    2. Kinetic fractionation effect for ice (Ciais and Jouzel, 1994)
    3. Mixed-phase region (WBF zone: 248-268 K)
    """
    T = np.asarray(T, dtype=float)
    
    # ============================================================
    # Ice-vapor calculation
    # ============================================================
    # Equilibrium fractionation factor for hydrogen isotopes
    # in ice and vapor for -40 to 0 C from Merlivat and Nief, 1967, Tellus.
    b_iv_p3 = 0.0
    b_iv_p2 = 0.0
    b_iv_p1 = 0.0
    b_iv_0 = -9.45e-2
    b_iv_m1 = 0.0
    b_iv_m2 = 16289.0
    b_iv_m3 = 0.0
    
    # For ice-vapor only
    alpha_IV = np.exp(
        b_iv_p3 * T**3 + b_iv_p2 * T**2 + b_iv_p1 * T
        + b_iv_0
        + b_iv_m1 * T**-1 + b_iv_m2 * T**-2 + b_iv_m3 * T**-3
    )
    
    # Adjust fractionation for kinetic effect for ice and vapor,
    # according to Ciais and Jouzel, 1994.
    s_I = 1.02 - 0.0038 * (T - TC2K)
    
    # Ratio diffusivity(HD16O) / diffusivity(HH16O) from Merlivat, 1978
    diffusivity_ratio = 0.9755
    
    alpha_IV = alpha_IV * s_I / ((alpha_IV * (s_I - 1.0) / diffusivity_ratio) + 1.0)
    
    # ============================================================
    # Water-vapor calculation
    # ============================================================
    # Water-vapor for hydrogen isotopes is represented by two studies,
    # covering two different temperature ranges
    
    # Equilibrium fractionation factor for hydrogen isotopes
    # in water and vapor for 273 to 373 K from Majoube, 1971
    # (as reported in Criss, 1999, p. 103).
    b_lv_p3 = 0.0
    b_lv_p2 = 0.0
    b_lv_p1 = 0.0
    b_lv_0 = 52.612e-3
    b_lv_m1 = -76.248
    b_lv_m2 = 24.844e3
    b_lv_m3 = 0.0
    
    # Equilibrium fractionation factors for water-vapor exchange
    alpha_LV_M71 = np.exp(
        b_lv_p3 * T**3 + b_lv_p2 * T**2 + b_lv_p1 * T
        + b_lv_0
        + b_lv_m1 * T**-1 + b_lv_m2 * T**-2 + b_lv_m3 * T**-3
    )
    
    # Equilibrium fractionation factor for hydrogen isotopes in
    # water and vapor for 258 to 273 K from Merlivat and Nief, 1967, Tellus.
    b_lv_p3 = 0.0
    b_lv_p2 = 0.0
    b_lv_p1 = 0.0
    b_lv_0 = -10.0e-2
    b_lv_m1 = 0.0
    b_lv_m2 = 15013.0
    b_lv_m3 = 0.0
    
    # Equilibrium fractionation factors for water-vapor exchange
    alpha_LV_MN67 = np.exp(
        b_lv_p3 * T**3 + b_lv_p2 * T**2 + b_lv_p1 * T
        + b_lv_0
        + b_lv_m1 * T**-1 + b_lv_m2 * T**-2 + b_lv_m3 * T**-3
    )
    
    # Merge calibration results
    # Use M71 for T >= TC2K (273.15 K), MN67 for T < TC2K
    i_switch = (T >= TC2K)
    alpha_LV = np.where(i_switch, alpha_LV_M71, alpha_LV_MN67)
    
    # ============================================================
    # Combine results
    # ============================================================
    # For mixed ice and water, factor goes from zero for T < 248 K
    # to one for T > 268 K, corresponding to the WBF zone.
    factor = (T - 248.0) / (268.0 - 248.0)
    factor = np.clip(factor, 0.0, 1.0)
    
    alpha = alpha_LV * factor + alpha_IV * (1.0 - factor)
>>>>>>> dev:opi/fractionation_hydrogen.py
    
    return alpha


<<<<<<< HEAD:OPI_python/opi/fractionation_hydrogen.py
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
=======
if __name__ == "__main__":
    print("Testing fractionation_hydrogen module...")
    print("(Comparing with MATLAB implementation)")
    
    # Test at various temperatures
    T_test = np.array([280.0, 273.15, 268.0, 258.0, 248.0, 238.0])
    
    print("\nTemperature (K) | alpha")
    print("-" * 40)
    
    for T in T_test:
        alpha = fractionation_hydrogen(T)
        print(f"{T:13.1f} | {alpha:10.6f}")
    
    # Test delta notation
    print("\nDelta notation (permil):")
    for T in T_test:
        alpha = fractionation_hydrogen(T)
        delta = (alpha - 1.0) * 1000.0
        print(f"  T = {T:6.1f} K: delta = {delta:+8.1f} permil")
>>>>>>> dev:opi/fractionation_hydrogen.py
    
    print("\nTest completed successfully!")
