"""
Calculate saturated vapor pressure as a function of temperature
Based on the Tetens equation
"""

import numpy as np


def saturated_vapor_pressure(temperature):
    """
    Calculate saturated vapor pressure using Tetens equation
    
    Parameters:
    -----------
    temperature : float or array-like
        Temperature in Kelvin
    
    Returns:
    --------
    e_s : float or ndarray
        Saturated vapor pressure in Pa
    """
    # Convert from Kelvin to Celsius
    temp_celsius = temperature - 273.15
    
    # Tetens equation for temperatures above 0°C
    # e_s = 6.1078 * 10^(a*T / (b+T))
    # Where a=7.5, b=237.3 for water above 0°C
    e_s = 6.1078 * np.power(10.0, (7.5 * temp_celsius) / (237.3 + temp_celsius))
    
    # Convert from hPa to Pa
    e_s_pa = e_s * 100.0
    
    return e_s_pa