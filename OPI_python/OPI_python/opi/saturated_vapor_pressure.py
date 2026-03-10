"""
Calculate saturated vapor pressure as a function of temperature
<<<<<<< HEAD:OPI_python/opi/saturated_vapor_pressure.py
Based on the Tetens equation
=======

Based on Goff and Gratch (1946), as modified by Goff (1965).
See review by Murphy and Koop (2005) for details.

The Wegener-Bergeron-Findeisen (WBF) zone, defined by the temperature
range 268 to 248 K, is where water and ice coexist.
For this zone, eS is set using weighted averages for water and ice components.
>>>>>>> dev:opi/saturated_vapor_pressure.py
"""

import numpy as np


<<<<<<< HEAD:OPI_python/opi/saturated_vapor_pressure.py
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
=======
def saturated_vapor_pressure(T):
    """
    Calculate saturated vapor pressure over water and ice.
    
    Uses the Goff-Gratch equation (1946), as modified by Goff (1965).
    The WBF (Wegener-Bergeron-Findeisen) zone is defined by the temperature
    range 268 to 248 K where water and ice coexist.
    
    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin
    
    Returns
    -------
    eS : float or ndarray
        Saturated vapor pressure in Pa
    
    References
    ----------
    Goff and Gratch (1946)
    Murphy and Koop (2005)
    """
    T = np.asarray(T)
    
    # Saturated vapor pressure (Pa) in equilibrium with water
    # Goff-Gratch equation for water
    eS_Water = 10.0**(
        -7.90298 * (373.16 / T - 1.0)
        + 5.02808 * np.log10(373.16 / T)
        - 1.3816e-7 * (10.0**(11.344 * (1.0 - T / 373.16)) - 1.0)
        + 8.1328e-3 * (10.0**(3.49149 * (1.0 - 373.16 / T)) - 1.0)
        + np.log10(101324.6)
    )
    
    # Saturation vapor pressure (Pa) in equilibrium with ice
    # Goff-Gratch equation for ice
    eS_Ice = 10.0**(
        -9.09718 * ((273.16 / T) - 1.0)
        - 3.56654 * np.log10(273.16 / T)
        + 0.876793 * (1.0 - (T / 273.16))
        + np.log10(610.71)
    )
    
    # Mix values using a factor that goes from zero for T < 248 K
    # to one for T > 268 K, corresponding to the WBF zone.
    factor = (T - 248.0) / (268.0 - 248.0)
    factor = np.clip(factor, 0.0, 1.0)  # All ice when factor=0, all water when factor=1
    
    eS = eS_Ice * (1.0 - factor) + eS_Water * factor
    
    return eS
>>>>>>> dev:opi/saturated_vapor_pressure.py
