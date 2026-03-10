"""
Thermodynamic utility functions for the OPI model.

This module provides a collection of thermodynamic calculations commonly used
in atmospheric science, including lapse rates, vapor pressure, air density,
and various temperature measures.
"""

import numpy as np
from typing import Union

NumberOrArray = Union[float, np.ndarray]


def compute_lapse_rates(
    T: NumberOrArray,
    P: NumberOrArray,
    r_s: NumberOrArray,
    G: float = 9.81,
    CPD: float = 1005.7,
    CPV: float = 1952,
    RD: float = 287.0,
    L: float = 2.501e6,
) -> dict:
    """
    Computes dry, moist, and environmental lapse rates.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    P : float or np.ndarray
        Pressure in Pa.
    r_s : float or np.ndarray
        Saturation mixing ratio (kg/kg).
    G : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81.
    CPD : float, optional
        Specific heat of dry air at constant pressure (J/kg/K). Default is 1005.7.
    CPV : float, optional
        Specific heat of water vapor at constant pressure (J/kg/K). Default is 1952.
    RD : float, optional
        Gas constant for dry air (J/kg/K). Default is 287.0.
    L : float, optional
        Latent heat of vaporization (J/kg). Default is 2.501e6.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dry': Dry adiabatic lapse rate (K/m)
        - 'moist': Moist adiabatic lapse rate (K/m)
        - 'environmental': Environmental lapse rate, approximated as moist (K/m)

    Notes
    -----
    The dry adiabatic lapse rate is given by:
        Gamma_d = g / c_p

    The moist adiabatic lapse rate is given by:
        Gamma_m = (g/c_p) * (1 + (L*r_s)/(R_d*T)) / (1 + (L^2*r_s)/(c_p*R_v*T^2))

    For simplicity, the environmental lapse rate is approximated as equal to
    the moist lapse rate in this implementation.
    """
    # Dry adiabatic lapse rate (K/m)
    gamma_dry = G / CPD

    # Moist adiabatic lapse rate (K/m)
    # Using the approximate formula from Wallace and Hobbs
    epsilon = RD / 461.5  # RD / RV where RV = 461.5 J/kg/K
    
    numerator = 1.0 + (L * r_s) / (RD * T)
    denominator = 1.0 + (L**2 * r_s) / (CPD * 461.5 * T**2)
    gamma_moist = gamma_dry * (numerator / denominator)

    # Environmental lapse rate (approximation)
    gamma_env = gamma_moist

    return {
        'dry': gamma_dry,
        'moist': gamma_moist,
        'environmental': gamma_env,
    }


def mixing_ratio_to_vapor_density(
    r: NumberOrArray,
    P: NumberOrArray,
    T: NumberOrArray,
    RD: float = 287.0,
    EPSILON: float = 0.622,
) -> NumberOrArray:
    """
    Convert mixing ratio to vapor density.

    Parameters
    ----------
    r : float or np.ndarray
        Mixing ratio (kg water vapor / kg dry air).
    P : float or np.ndarray
        Total pressure in Pa.
    T : float or np.ndarray
        Temperature in Kelvin.
    RD : float, optional
        Gas constant for dry air (J/kg/K). Default is 287.0.
    EPSILON : float, optional
        Ratio of gas constants (R_d / R_v). Default is 0.622.

    Returns
    -------
    float or np.ndarray
        Vapor density (kg/m^3).

    Notes
    -----
    The vapor density is calculated from the ideal gas law applied to water vapor.
    First, the partial pressure of water vapor is computed from the mixing ratio,
    then the vapor density is obtained from rho_v = e / (R_v * T).
    """
    # Calculate vapor pressure from mixing ratio
    e = vapor_pressure(P, r, EPSILON)
    
    # Gas constant for water vapor
    RV = RD / EPSILON
    
    # Vapor density from ideal gas law
    rho_v = e / (RV * T)
    
    return rho_v


def vapor_pressure(
    P: NumberOrArray,
    r: NumberOrArray,
    EPSILON: float = 0.622,
) -> NumberOrArray:
    """
    Calculate vapor pressure from total pressure and mixing ratio.

    Parameters
    ----------
    P : float or np.ndarray
        Total pressure in Pa.
    r : float or np.ndarray
        Mixing ratio (kg water vapor / kg dry air).
    EPSILON : float, optional
        Ratio of gas constants (R_d / R_v). Default is 0.622.

    Returns
    -------
    float or np.ndarray
        Vapor pressure (Pa).

    Notes
    -----
    The relationship between vapor pressure (e), total pressure (P), and
    mixing ratio (r) is:
        r = epsilon * e / (P - e)
    
    Solving for e:
        e = P * r / (epsilon + r)
    """
    e = P * r / (EPSILON + r)
    return e


def total_air_density(
    P: NumberOrArray,
    T: NumberOrArray,
    r: NumberOrArray,
    RD: float = 287.0,
    EPSILON: float = 0.622,
) -> NumberOrArray:
    """
    Calculate total air density using Wallace and Hobbs eq. 3.15.

    Parameters
    ----------
    P : float or np.ndarray
        Total pressure in Pa.
    T : float or np.ndarray
        Temperature in Kelvin.
    r : float or np.ndarray
        Mixing ratio (kg water vapor / kg dry air).
    RD : float, optional
        Gas constant for dry air (J/kg/K). Default is 287.0.
    EPSILON : float, optional
        Ratio of gas constants (R_d / R_v). Default is 0.622.

    Returns
    -------
    float or np.ndarray
        Total air density (kg/m^3).

    Notes
    -----
    Following Wallace and Hobbs (2006), Equation 3.15:
        rho = P / (R_d * T * (1 + r/epsilon) / (1 + r))
    
    Or equivalently:
        rho = P * (1 + r) / (R_d * T * (1 + r/epsilon))

    This accounts for the contribution of both dry air and water vapor to
    the total density.
    """
    numerator = P * (1.0 + r)
    denominator = RD * T * (1.0 + r / EPSILON)
    rho = numerator / denominator
    return rho


def potential_temperature(
    T: NumberOrArray,
    P: NumberOrArray,
    P0: float = 101325,
    RD: float = 287.0,
    CPD: float = 1005.7,
) -> NumberOrArray:
    """
    Calculate potential temperature.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    P : float or np.ndarray
        Pressure in Pa.
    P0 : float, optional
        Reference pressure (Pa). Default is 101325 (standard sea level pressure).
    RD : float, optional
        Gas constant for dry air (J/kg/K). Default is 287.0.
    CPD : float, optional
        Specific heat of dry air at constant pressure (J/kg/K). Default is 1005.7.

    Returns
    -------
    float or np.ndarray
        Potential temperature in Kelvin.

    Notes
    -----
    Potential temperature is the temperature a parcel of air would have if
    brought adiabatically from its current pressure to the reference pressure P0.
    
    The formula is:
        theta = T * (P0 / P)^(R_d / c_p_d)

    Potential temperature is conserved during dry adiabatic processes and is
    useful for identifying air masses and studying atmospheric stability.
    """
    kappa = RD / CPD
    theta = T * (P0 / P) ** kappa
    return theta


def equivalent_potential_temperature(
    T: NumberOrArray,
    P: NumberOrArray,
    r_s: NumberOrArray,
    L: float = 2.501e6,
    CPD: float = 1005.7,
) -> NumberOrArray:
    """
    Calculate equivalent potential temperature.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    P : float or np.ndarray
        Pressure in Pa.
    r_s : float or np.ndarray
        Saturation mixing ratio (kg/kg).
    L : float, optional
        Latent heat of vaporization (J/kg). Default is 2.501e6.
    CPD : float, optional
        Specific heat of dry air at constant pressure (J/kg/K). Default is 1005.7.

    Returns
    -------
    float or np.ndarray
        Equivalent potential temperature in Kelvin.

    Notes
    -----
    Equivalent potential temperature is the potential temperature a parcel
    would have if all its water vapor were condensed and the latent heat
    released was used to heat the parcel.

    The approximate formula used here is:
        theta_e = theta * exp(L * r_s / (c_p_d * T))

    where theta is the potential temperature. This approximation assumes
    that the parcel is saturated and neglects small terms involving the
    specific heat of water vapor.

    Equivalent potential temperature is conserved during both dry and
    moist adiabatic processes, making it a useful tracer for air masses.
    """
    # First compute potential temperature
    theta = potential_temperature(T, P)
    
    # Then compute equivalent potential temperature
    theta_e = theta * np.exp(L * r_s / (CPD * T))
    
    return theta_e
