"""
Climate Records Processing

Processes paleoclimate records for OPI calculations.
Matches MATLAB's climateRecords.m

Uses MEBM (Moist Energy Balance Model) output and benthic foram climate records
to estimate past temperature and isotope values.
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional


def load_mebm_data(mat_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MEBM (Moist Energy Balance Model) data from MAT file.
    
    Parameters
    ----------
    mat_file_path : str
        Path to MEBM .mat file
    
    Returns
    -------
    dA_MEBM : ndarray
        Change in radiative heat loss intercept (W/m^2)
    lat_MEBM : ndarray
        Latitude (degrees)
    T_MEBM : ndarray
        Surface air temperature (K) for each (dA, lat) combination
    """
    from scipy.io import loadmat
    
    data = loadmat(mat_file_path, squeeze_me=True)
    
    dA_MEBM = data['dA_MEBM']
    lat_MEBM = data['lat_MEBM']
    T_MEBM = data['T_MEBM'] + 273.15  # Convert to Kelvin
    
    return dA_MEBM, lat_MEBM, T_MEBM


def load_benthic_foram_data(mat_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load benthic foram climate record from MAT file.
    
    Parameters
    ----------
    mat_file_path : str
        Path to benthic foram .mat file
    
    Returns
    -------
    age_record : ndarray
        Age (Ma)
    T_dw_record : ndarray
        Deep-water temperature (K)
    d18O_sw_record : ndarray
        d18O of seawater (per mil, VSMOW)
    """
    from scipy.io import loadmat
    
    data = loadmat(mat_file_path, squeeze_me=True)
    
    age_record = data['ageRecord']
    T_dw_record = data['TdwRecord']
    d18O_sw_record = data['d18OswRecord']
    
    return age_record, T_dw_record, d18O_sw_record


def calculate_climate_records(
    mebm_file: str,
    benthic_file: str,
    lat: float,
    T0_pres: float,
    d2H0_pres: float,
    d_d2H0_d_d18O0: float,
    d_d2H0_d_T0: float,
    span_age: float,
    smooth_method: str = 'loess'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time series for T0 and d2H0 at specified latitude.
    
    Matches MATLAB's climateRecords function.
    
    Parameters
    ----------
    mebm_file : str
        Path to MEBM data .mat file
    benthic_file : str
        Path to benthic foram climate record .mat file
    lat : float
        Latitude for study area (degrees)
    T0_pres : float
        Present sea-level air temperature (K)
    d2H0_pres : float
        Present d2H0 at specified latitude (fraction)
    d_d2H0_d_d18O0 : float
        Slope of local meteoric water line (dimensionless)
    d_d2H0_d_T0 : float
        Temperature scaling for d2H0 (fraction per degree C)
    span_age : float
        Age range (Ma) for smoothing records
    smooth_method : str
        Smoothing method ('loess', 'moving', or 'none')
    
    Returns
    -------
    T0_record : ndarray
        Time series for T0 (K) at specified latitude
    d2H0_record : ndarray
        Time series for d2H0 (fraction) at specified latitude
    age_record : ndarray
        Age (Ma) for record values
    T0_record_smooth : ndarray
        Smoothed T0 time series
    d2H0_record_smooth : ndarray
        Smoothed d2H0 time series
    """
    TC2K = 273.15
    
    # Load data
    dA_MEBM, lat_MEBM, T_MEBM = load_mebm_data(mebm_file)
    age_record, T_dw_record, d18O_sw_record = load_benthic_foram_data(benthic_file)
    
    # Estimate latitude of deep-water production
    # Find where MEBM temperature matches present deep-water temperature
    T_dw_mebs_pres = interp1d(lat_MEBM, T_MEBM[:, dA_MEBM == 0].flatten(), 
                              fill_value='extrapolate')(lat_MEBM)
    lat_dw = interp1d(T_dw_mebs_pres, lat_MEBM, fill_value='extrapolate')(T_dw_record[0])
    
    # Calculate variation of dA with age
    T_dw_mebs = interp1d(lat_MEBM, T_MEBM, axis=0, fill_value='extrapolate')(lat_dw)
    
    # Create interpolator for dA vs T_dw
    # Sort in ascending order for interpolation
    sort_idx = np.argsort(T_dw_mebs)
    T_dw_sorted = T_dw_mebs[sort_idx]
    dA_sorted = dA_MEBM[sort_idx]
    
    # Linear extrapolation for values outside range
    dA_record = interp1d(T_dw_sorted, dA_sorted, kind='linear', 
                         fill_value='extrapolate')(T_dw_record)
    
    # Estimate T0_record at specified latitude
    # Interpolate MEBM to study latitude
    T_lat_mebs = interp1d(lat_MEBM, T_MEBM, axis=0)(abs(lat))
    
    # Create interpolator for T vs dA at study latitude
    T_mebs_pres = interp1d(dA_MEBM, T_lat_mebs, fill_value='extrapolate')(0)
    T_mebs_past = interp1d(dA_MEBM, T_lat_mebs, fill_value='extrapolate')(dA_record)
    
    # Calculate T0 record
    T0_record = T0_pres + (T_mebs_past - T_mebs_pres)
    
    # Estimate d2H0_record
    d18O_sw_pres = d18O_sw_record[0]
    d2H0_record = (d2H0_pres 
                   + d_d2H0_d_d18O0 * (d18O_sw_record - d18O_sw_pres) * 1e-3
                   + d_d2H0_d_T0 * (T0_record - T0_pres))
    
    # Calculate smoothed versions
    n_record = len(age_record)
    n_span = int(np.ceil(n_record * span_age / (np.max(age_record) - np.min(age_record))))
    
    if smooth_method == 'loess':
        # Use LOWESS-like smoothing (locally weighted regression)
        T0_record_smooth = _loess_smooth(age_record, T0_record, n_span)
        d2H0_record_smooth = _loess_smooth(age_record, d2H0_record, n_span)
    elif smooth_method == 'moving':
        # Simple moving average
        T0_record_smooth = uniform_filter1d(T0_record, size=n_span, mode='nearest')
        d2H0_record_smooth = uniform_filter1d(d2H0_record, size=n_span, mode='nearest')
    else:
        T0_record_smooth = T0_record.copy()
        d2H0_record_smooth = d2H0_record.copy()
    
    # Set present values exactly
    present_idx = age_record == 0
    T0_record_smooth[present_idx] = T0_pres
    d2H0_record_smooth[present_idx] = d2H0_pres
    
    return T0_record, d2H0_record, age_record, T0_record_smooth, d2H0_record_smooth


def _loess_smooth(x: np.ndarray, y: np.ndarray, span: int) -> np.ndarray:
    """
    Simple LOESS-like smoothing.
    
    Parameters
    ----------
    x : ndarray
        Independent variable
    y : ndarray
        Dependent variable
    span : int
        Window size
    
    Returns
    -------
    y_smooth : ndarray
        Smoothed values
    """
    n = len(y)
    y_smooth = np.zeros_like(y)
    
    half_span = span // 2
    
    for i in range(n):
        # Define window
        start = max(0, i - half_span)
        end = min(n, i + half_span + 1)
        
        # Get window data
        x_window = x[start:end]
        y_window = y[start:end]
        
        # Calculate weights (tricube)
        d = np.abs(x_window - x[i])
        d_max = np.max(d) if len(d) > 0 else 1
        if d_max > 0:
            weights = (1 - (d / d_max) ** 3) ** 3
        else:
            weights = np.ones_like(d)
        
        # Weighted linear regression
        if len(x_window) > 1:
            # Add bias term
            X = np.column_stack([np.ones_like(x_window), x_window])
            W = np.diag(weights)
            
            # Weighted least squares: (X'WX)^(-1) X'Wy
            try:
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y_window, rcond=None)[0]
                y_smooth[i] = beta[0] + beta[1] * x[i]
            except:
                y_smooth[i] = np.average(y_window, weights=weights)
        else:
            y_smooth[i] = y[i]
    
    return y_smooth


def estimate_paleoclimate_parameters(
    age: float,
    mebm_file: str,
    benthic_file: str,
    lat: float,
    T0_pres: float,
    d2H0_pres: float,
    d_d2H0_d_d18O0: float = 8.0,
    d_d2H0_d_T0: float = None
) -> Tuple[float, float]:
    """
    Estimate paleoclimate parameters at a specific age.
    
    Parameters
    ----------
    age : float
        Target age (Ma)
    mebm_file, benthic_file : str
        Data file paths
    lat : float
        Latitude (degrees)
    T0_pres : float
        Present temperature (K)
    d2H0_pres : float
        Present d2H0 (fraction)
    d_d2H0_d_d18O0 : float
        MWL slope (default 8.0)
    d_d2H0_d_T0 : float, optional
        Temperature scaling (default: estimated from lat)
    
    Returns
    -------
    T0 : float
        Temperature at age (K)
    d2H0 : float
        d2H0 at age (fraction)
    """
    # Default temperature scaling if not provided
    if d_d2H0_d_T0 is None:
        # Typical value: ~4-5 permil per degree at mid-latitudes
        d_d2H0_d_T0 = 4.5e-3  # fraction per degree C
    
    T0_record, d2H0_record, age_record, _, _ = calculate_climate_records(
        mebm_file, benthic_file, lat, T0_pres, d2H0_pres,
        d_d2H0_d_d18O0, d_d2H0_d_T0, span_age=1.0, smooth_method='moving'
    )
    
    # Interpolate to target age
    T0 = interp1d(age_record, T0_record, fill_value='extrapolate')(age)
    d2H0 = interp1d(age_record, d2H0_record, fill_value='extrapolate')(age)
    
    return float(T0), float(d2H0)
