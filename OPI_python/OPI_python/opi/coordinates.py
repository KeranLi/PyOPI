"""
Coordinate transformation functions
"""

import numpy as np
from .constants import M_PER_DEGREE


def lonlat2xy(lon, lat, lon0, lat0):
    """
    Convert longitude/latitude coordinates to x/y coordinates relative to a reference point.
    
    Parameters:
    -----------
    lon : float or array-like
        Longitude(s) in degrees
    lat : float or array-like
        Latitude(s) in degrees
    lon0 : float
        Reference longitude in degrees
    lat0 : float
        Reference latitude in degrees
    
    Returns:
    --------
    x : float or ndarray
        X coordinate(s) in meters
    y : float or ndarray
        Y coordinate(s) in meters
    """
    x = (lon - lon0) * M_PER_DEGREE * np.cos(np.radians(lat0))
    y = (lat - lat0) * M_PER_DEGREE
    
    return x, y


def xy2lonlat(x, y, lon0, lat0):
    """
    Convert x/y coordinates to longitude/latitude coordinates relative to a reference point.
    
    Parameters:
    -----------
    x : float or array-like
        X coordinate(s) in meters
    y : float or array-like
        Y coordinate(s) in meters
    lon0 : float
        Reference longitude in degrees
    lat0 : float
        Reference latitude in degrees
    
    Returns:
    --------
    lon : float or ndarray
        Longitude(s) in degrees
    lat : float or ndarray
        Latitude(s) in degrees
    """
    lon = lon0 + x / (M_PER_DEGREE * np.cos(np.radians(lat0)))
    lat = lat0 + y / M_PER_DEGREE
    
    return lon, lat