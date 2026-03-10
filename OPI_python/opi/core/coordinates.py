"""
Coordinate transformations for the OPI model.

This module provides functions for converting between different coordinate
systems and calculating meteorological parameters related to coordinates.
"""

import numpy as np


def lonlat2xy(
    lon: float | np.ndarray,
    lat: float | np.ndarray,
    lon0: float,
    lat0: float,
    radius_earth: float = 6371e3
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert longitude/latitude to Cartesian coordinates (x, y) in meters.

    Uses the equirectangular projection (also known as equidistant cylindrical
    projection), which approximates the Earth as a sphere and projects coordinates
    onto a plane. This projection is suitable for small-scale regions where
    distortion is minimal.

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude(s) in degrees.
    lat : float or np.ndarray
        Latitude(s) in degrees.
    lon0 : float
        Reference longitude (origin) in degrees.
    lat0 : float
        Reference latitude (origin) in degrees.
    radius_earth : float, optional
        Earth's radius in meters. Default is 6371e3 (6371 km).

    Returns
    -------
    x : float or np.ndarray
        East-west Cartesian coordinate(s) in meters (positive east).
    y : float or np.ndarray
        North-south Cartesian coordinate(s) in meters (positive north).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Equirectangular_projection
    """
    # Convert degrees to radians
    deg2rad = np.pi / 180.0
    lat0_rad = lat0 * deg2rad

    # Meters per degree of latitude (constant)
    m_per_degree_lat = radius_earth * deg2rad

    # Meters per degree of longitude (varies with latitude)
    m_per_degree_lon = radius_earth * np.cos(lat0_rad) * deg2rad

    # Calculate Cartesian coordinates
    x = (lon - lon0) * m_per_degree_lon
    y = (lat - lat0) * m_per_degree_lat

    return x, y


def xy2lonlat(
    x: float | np.ndarray,
    y: float | np.ndarray,
    lon0: float,
    lat0: float,
    radius_earth: float = 6371e3
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert Cartesian coordinates (x, y) to longitude/latitude.

    This is the inverse transformation of `lonlat2xy`, converting Cartesian
    coordinates back to geographic coordinates using the equirectangular
    projection.

    Parameters
    ----------
    x : float or np.ndarray
        East-west Cartesian coordinate(s) in meters (positive east).
    y : float or np.ndarray
        North-south Cartesian coordinate(s) in meters (positive north).
    lon0 : float
        Reference longitude (origin) in degrees.
    lat0 : float
        Reference latitude (origin) in degrees.
    radius_earth : float, optional
        Earth's radius in meters. Default is 6371e3 (6371 km).

    Returns
    -------
    lon : float or np.ndarray
        Longitude(s) in degrees.
    lat : float or np.ndarray
        Latitude(s) in degrees.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Equirectangular_projection
    """
    # Convert degrees to radians
    deg2rad = np.pi / 180.0
    lat0_rad = lat0 * deg2rad

    # Meters per degree of latitude (constant)
    m_per_degree_lat = radius_earth * deg2rad

    # Meters per degree of longitude (varies with latitude)
    m_per_degree_lon = radius_earth * np.cos(lat0_rad) * deg2rad

    # Calculate geographic coordinates
    lon = lon0 + x / m_per_degree_lon
    lat = lat0 + y / m_per_degree_lat

    return lon, lat


def calculate_coriolis_parameter(
    lat: float | np.ndarray,
    omega: float = 7.2921e-5
) -> float | np.ndarray:
    """
    Calculate the Coriolis parameter f.

    The Coriolis parameter represents the effect of Earth's rotation on moving
    objects in a rotating reference frame. It is essential in geophysical fluid
    dynamics, particularly for modeling atmospheric and oceanic flows.

    Parameters
    ----------
    lat : float or np.ndarray
        Latitude(s) in degrees. Positive for northern hemisphere,
        negative for southern hemisphere.
    omega : float, optional
        Earth's angular rotation rate in rad/s. Default is 7.2921e-5 rad/s,
        which corresponds to one full rotation per sidereal day.

    Returns
    -------
    f : float or np.ndarray
        Coriolis parameter(s) in s^-1. Positive in the northern hemisphere,
        negative in the southern hemisphere, and zero at the equator.

    References
    ----------
    .. [1] Holton, J. R. (2004). An Introduction to Dynamic Meteorology.
           Elsevier Academic Press.
    .. [2] https://en.wikipedia.org/wiki/Coriolis_frequency
    """
    # Convert latitude from degrees to radians
    lat_rad = np.deg2rad(lat)

    # Calculate Coriolis parameter: f = 2 * omega * sin(lat)
    f = 2 * omega * np.sin(lat_rad)

    return f


def wind_components(
    speed: float | np.ndarray,
    azimuth: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert wind speed and azimuth to u and v components.

    The meteorological convention defines u as the eastward component and
    v as the northward component of the wind vector. The azimuth is measured
    clockwise from North (0° = North, 90° = East, 180° = South, 270° = West).

    Parameters
    ----------
    speed : float or np.ndarray
        Wind speed(s) in any consistent unit (e.g., m/s, km/h).
    azimuth : float or np.ndarray
        Wind direction azimuth(s) in degrees, measured clockwise from North.
        0° indicates wind coming from the North (blowing toward South),
        90° indicates wind coming from the East (blowing toward West), etc.

    Returns
    -------
    u : float or np.ndarray
        Eastward wind component(s) (positive east). Same units as speed.
    v : float or np.ndarray
        Northward wind component(s) (positive north). Same units as speed.

    Notes
    -----
    The conversion uses the meteorological convention where:
    - u = speed * sin(azimuth)
    - v = speed * cos(azimuth)

    Note that this convention assumes the azimuth indicates the direction
    FROM which the wind is coming (meteorological convention). If your
    azimuth indicates the direction TOWARD which the wind is blowing,
    you may need to add/subtract 180° before calling this function.

    References
    ----------
    .. [1] Holton, J. R. (2004). An Introduction to Dynamic Meteorology.
           Elsevier Academic Press.
    .. [2] https://en.wikipedia.org/wiki/Wind_direction
    """
    # Convert azimuth from degrees to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Calculate wind components
    u = speed * np.sin(azimuth_rad)  # East component
    v = speed * np.cos(azimuth_rad)  # North component

    return u, v
