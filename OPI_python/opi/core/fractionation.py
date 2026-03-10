"""
Isotope Fractionation Factors

This module calculates equilibrium fractionation factors for hydrogen (dD) 
and oxygen (d18O) isotopes as a function of temperature.
"""

import numpy as np


def fractionation_hydrogen(T):
    """
    Calculate hydrogen isotope (dD) equilibrium fractionation factor.
    
    The fractionation factor alpha describes the ratio of isotope ratios
    between two phases: alpha = R_precip / R_vapor
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    float or ndarray
        Equilibrium fractionation factor (dimensionless)
        
    Notes
    -----
    Uses the formulation from Majoube (1971) with coefficients from
    the literature summary in Horita and Wesolowski (1994).
    
    For vapor-liquid equilibrium:
    1000 * ln(alpha) = a/T² + b/T + c
    
    References
    ----------
    Majoube (1971) - Fractionnement en oxygene-18 et en deuterium 
        entre l'eau et sa vapeur
    Horita and Wesolowski (1994) - Liquid-vapor fractionation of 
        oxygen and hydrogen isotopes
    """
    # Coefficients for vapor-liquid equilibrium (Majoube, 1971)
    # 1000*ln(alpha) = a/T² + b/T + c, where T is in Kelvin
    a = 24.844
    b = -76.248
    c = 52.612
    
    # Calculate ln(alpha)
    ln_alpha = (a * 1e6 / T**2 + b * 1e3 / T + c) / 1000.0
    
    return np.exp(ln_alpha)


def fractionation_oxygen(T):
    """
    Calculate oxygen isotope (d18O) equilibrium fractionation factor.
    
    The fractionation factor alpha describes the ratio of isotope ratios
    between two phases: alpha = R_precip / R_vapor
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    float or ndarray
        Equilibrium fractionation factor (dimensionless)
        
    Notes
    -----
    Uses the formulation from Majoube (1971) with coefficients from
    the literature summary in Horita and Wesolowski (1994).
    
    For vapor-liquid equilibrium:
    1000 * ln(alpha) = a/T² + b/T + c
    
    References
    ----------
    Majoube (1971) - Fractionnement en oxygene-18 entre l'eau et sa vapeur
    Horita and Wesolowski (1994) - Liquid-vapor fractionation of 
        oxygen and hydrogen isotopes
    """
    # Coefficients for vapor-liquid equilibrium (Majoube, 1971)
    # 1000*ln(alpha) = a/T² + b/T + c, where T is in Kelvin
    a = 1.137
    b = -0.4156
    c = -2.0667
    
    # Calculate ln(alpha)
    ln_alpha = (a * 1e6 / T**2 + b * 1e3 / T + c) / 1000.0
    
    return np.exp(ln_alpha)


def fractionation_ice_vapor_hydrogen(T):
    """
    Calculate hydrogen isotope fractionation factor for ice-vapor equilibrium.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    float or ndarray
        Equilibrium fractionation factor (dimensionless)
        
    References
    ----------
    Merlivat and Nief (1967) - Fractionnement isotopique lors des 
        changements d'etat solide-vapeur et liquide-vapeur de l'eau
    """
    # Coefficients for ice-vapor equilibrium
    # 1000*ln(alpha) = a/T² + b/T + c
    a = 16288
    b = 0
    c = -9.34e-2
    
    ln_alpha = (a / T**2 + b / T + c) / 1000.0
    
    return np.exp(ln_alpha)


def fractionation_ice_vapor_oxygen(T):
    """
    Calculate oxygen isotope fractionation factor for ice-vapor equilibrium.
    
    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin
        
    Returns
    -------
    float or ndarray
        Equilibrium fractionation factor (dimensionless)
        
    References
    ----------
    Majoube (1971) - Oxygen-18 and deuterium fractionation between 
        ice and water vapor
    """
    # Coefficients for ice-vapor equilibrium
    # 1000*ln(alpha) = a/T² + b/T + c
    a = 8111
    b = 0
    c = -0.19
    
    ln_alpha = (a / T**2 + b / T + c) / 1000.0
    
    return np.exp(ln_alpha)


def kinetic_fractionation_hydrogen(D_ratio=0.9755, n=1.0):
    """
    Calculate kinetic fractionation factor for hydrogen during evaporation.
    
    Parameters
    ----------
    D_ratio : float
        Diffusivity ratio (rare isotopologue / common isotopologue)
        Default: 0.9755 (Merlivat, 1978)
    n : float
        Exponent for diffusive regime (0.5 to 1.0)
        Default: 1.0 (fully turbulent)
        0.58 would be for molecular diffusion (Stewart, 1975)
        
    Returns
    -------
    float
        Kinetic fractionation factor
        
    References
    ----------
    Merlivat (1978) - Molecular diffusivities of H2O16, HDO, and H2O18 
        in gases
    Stewart (1975) - Stable isotope fractionation due to evaporation 
        and isotopic exchange
    Criss (1999) - Principles of Stable Isotope Distribution
    """
    return D_ratio**(-n)


def kinetic_fractionation_oxygen(D_ratio=0.9723, n=1.0):
    """
    Calculate kinetic fractionation factor for oxygen during evaporation.
    
    Parameters
    ----------
    D_ratio : float
        Diffusivity ratio (rare isotopologue / common isotopologue)
        Default: 0.9723 (Merlivat, 1978)
    n : float
        Exponent for diffusive regime (0.5 to 1.0)
        Default: 1.0 (fully turbulent)
        
    Returns
    -------
    float
        Kinetic fractionation factor
        
    References
    ----------
    Merlivat (1978) - Molecular diffusivities of H2O16, HDO, and H2O18 
        in gases
    """
    return D_ratio**(-n)
