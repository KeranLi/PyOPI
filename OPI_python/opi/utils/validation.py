"""
Validation utility functions for OPI

This module provides functions for validating input data, parameters,
and grid consistency for OPI calculations.
"""

import numpy as np


def validate_topography(topography):
    """
    Validate topography dictionary has required keys.
    
    Checks that the topography dictionary contains all necessary
    keys and that the data arrays have consistent dimensions.
    
    Parameters
    ----------
    topography : dict
        Topography dictionary containing:
        - 'x': 1D array of x-coordinates (m)
        - 'y': 1D array of y-coordinates (m)
        - 'h': 2D array of elevation values (m)
    
    Returns
    -------
    bool
        True if validation passes
    
    Raises
    ------
    ValueError
        If required keys are missing or data is invalid
    TypeError
        If data types are incorrect
    
    Examples
    --------
    >>> topo = {
    ...     'x': np.array([0, 1000, 2000]),
    ...     'y': np.array([0, 1000, 2000]),
    ...     'h': np.array([[100, 150, 200], [120, 180, 220], [140, 200, 240]])
    ... }
    >>> validate_topography(topo)
    True
    """
    required_keys = ['x', 'y', 'h']
    
    # Check if all required keys are present
    for key in required_keys:
        if key not in topography:
            raise ValueError(f"Topography dictionary missing required key: '{key}'")
    
    x = topography['x']
    y = topography['y']
    h = topography['h']
    
    # Check data types
    for key, arr in [('x', x), ('y', y), ('h', h)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Topography['{key}'] must be a numpy array, got {type(arr)}")
    
    # Check dimensions
    if x.ndim != 1:
        raise ValueError(f"Topography['x'] must be 1D, got shape {x.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"Topography['y'] must be 1D, got shape {y.shape}")
    
    if h.ndim != 2:
        raise ValueError(f"Topography['h'] must be 2D, got shape {h.shape}")
    
    # Check array shapes are consistent
    if h.shape != (len(y), len(x)):
        raise ValueError(
            f"Topography['h'] shape {h.shape} does not match expected shape "
            f"(len(y), len(x)) = ({len(y)}, {len(x)})"
        )
    
    # Check for NaN values
    if np.any(np.isnan(x)):
        raise ValueError("Topography['x'] contains NaN values")
    if np.any(np.isnan(y)):
        raise ValueError("Topography['y'] contains NaN values")
    if np.any(np.isnan(h)):
        raise ValueError("Topography['h'] contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(x)):
        raise ValueError("Topography['x'] contains infinite values")
    if np.any(np.isinf(y)):
        raise ValueError("Topography['y'] contains infinite values")
    if np.any(np.isinf(h)):
        raise ValueError("Topography['h'] contains infinite values")
    
    # Check coordinates are monotonic
    if len(x) > 1:
        dx = np.diff(x)
        if not (np.all(dx > 0) or np.all(dx < 0)):
            raise ValueError("Topography['x'] must be monotonically increasing or decreasing")
        if not np.allclose(dx, dx[0]):
            # Non-uniform grid - just warn, don't fail
            pass
    
    if len(y) > 1:
        dy = np.diff(y)
        if not (np.all(dy > 0) or np.all(dy < 0)):
            raise ValueError("Topography['y'] must be monotonically increasing or decreasing")
    
    return True


def validate_parameters(params, bounds):
    """
    Validate parameters are within specified bounds.
    
    Checks that all parameters in the params dictionary are within
    the valid ranges specified in the bounds dictionary.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values to validate
    bounds : dict
        Dictionary mapping parameter names to (min, max) tuples.
        Use None for min or max to indicate no bound.
    
    Returns
    -------
    dict
        Dictionary with validation results:
        - 'valid': bool, True if all parameters are valid
        - 'errors': list of error messages for invalid parameters
        - 'warnings': list of warning messages
    
    Examples
    --------
    >>> params = {'U': 10.0, 'T0': 288.0, 'tau_c': 1500.0}
    >>> bounds = {
    ...     'U': (0, 50),
    ...     'T0': (250, 320),
    ...     'tau_c': (100, None)
    ... }
    >>> result = validate_parameters(params, bounds)
    >>> print(result['valid'])
    True
    """
    errors = []
    warnings_list = []
    
    for param_name, value in params.items():
        if param_name not in bounds:
            warnings_list.append(f"Parameter '{param_name}' not found in bounds, skipping validation")
            continue
        
        bound = bounds[param_name]
        
        # Handle single bounds (e.g., minimum only)
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise ValueError(f"Invalid bounds for parameter '{param_name}': {bound}. Expected (min, max) tuple.")
        
        min_val, max_val = bound
        
        # Check for NaN
        if np.isnan(value):
            errors.append(f"Parameter '{param_name}' is NaN")
            continue
        
        # Check for infinite values
        if np.isinf(value):
            errors.append(f"Parameter '{param_name}' is infinite")
            continue
        
        # Check minimum bound
        if min_val is not None and value < min_val:
            errors.append(
                f"Parameter '{param_name}' = {value} is below minimum bound {min_val}"
            )
        
        # Check maximum bound
        if max_val is not None and value > max_val:
            errors.append(
                f"Parameter '{param_name}' = {value} is above maximum bound {max_val}"
            )
    
    # Check for required parameters
    for param_name in bounds.keys():
        if param_name not in params:
            errors.append(f"Required parameter '{param_name}' is missing")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings_list
    }


def check_grid_consistency(x, y, h_grid):
    """
    Verify grid dimensions match between coordinates and data.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (length nx)
    y : array_like
        1D array of y-coordinates (length ny)
    h_grid : array_like
        2D array of elevation values with shape (ny, nx)
    
    Returns
    -------
    dict
        Dictionary with consistency check results:
        - 'consistent': bool, True if dimensions match
        - 'expected_shape': tuple, expected shape (ny, nx)
        - 'actual_shape': tuple, actual shape of h_grid
        - 'message': str, description of the check result
    
    Examples
    --------
    >>> x = np.array([0, 1000, 2000])
    >>> y = np.array([0, 1000])
    >>> h = np.array([[100, 150, 200], [120, 180, 220]])
    >>> result = check_grid_consistency(x, y, h)
    >>> print(result['consistent'])
    True
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h_grid = np.asarray(h_grid)
    
    nx = len(x)
    ny = len(y)
    expected_shape = (ny, nx)
    actual_shape = h_grid.shape
    
    consistent = actual_shape == expected_shape
    
    if consistent:
        message = f"Grid dimensions are consistent: shape {actual_shape}"
    else:
        message = (
            f"Grid dimensions mismatch: expected shape {expected_shape} "
            f"(ny={ny}, nx={nx}) but got {actual_shape}"
        )
    
    return {
        'consistent': consistent,
        'expected_shape': expected_shape,
        'actual_shape': actual_shape,
        'message': message
    }


def validate_wind_parameters(U, azimuth):
    """
    Validate wind parameters.
    
    Parameters
    ----------
    U : float
        Wind speed (m/s)
    azimuth : float
        Wind direction (degrees from North)
    
    Returns
    -------
    dict
        Validation result with 'valid' and 'errors' keys
    
    Examples
    --------
    >>> result = validate_wind_parameters(10.0, 90.0)
    >>> print(result['valid'])
    True
    """
    errors = []
    
    if U < 0:
        errors.append(f"Wind speed U must be non-negative, got {U}")
    
    if U > 100:
        errors.append(f"Wind speed U={U} m/s seems unrealistically high")
    
    if not 0 <= azimuth < 360:
        errors.append(f"Azimuth must be in range [0, 360), got {azimuth}")
    
    if np.isnan(U) or np.isnan(azimuth):
        errors.append("Wind parameters cannot be NaN")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


def validate_isotope_parameters(d2h0, d18o0=None):
    """
    Validate isotope parameters.
    
    Parameters
    ----------
    d2h0 : float
        Initial delta^2H value (fraction or permil)
    d18o0 : float, optional
        Initial delta^18O value (fraction or permil)
    
    Returns
    -------
    dict
        Validation result with 'valid' and 'errors' keys
    
    Examples
    --------
    >>> result = validate_isotope_parameters(-0.005)
    >>> print(result['valid'])
    True
    """
    errors = []
    
    # Check if values are in reasonable range (permil or fraction)
    # d2H typically ranges from -500 to +100 permil
    # d18O typically ranges from -60 to +10 permil
    
    if np.isnan(d2h0):
        errors.append("d2h0 cannot be NaN")
    else:
        # Check if likely permil (-500 to 100) or fraction (-0.5 to 0.1)
        if abs(d2h0) > 1:  # Likely permil
            if not -500 <= d2h0 <= 100:
                errors.append(f"d2h0={d2h0} is outside typical permil range [-500, 100]")
        else:  # Likely fraction
            if not -0.5 <= d2h0 <= 0.1:
                errors.append(f"d2h0={d2h0} is outside typical fraction range [-0.5, 0.1]")
    
    if d18o0 is not None:
        if np.isnan(d18o0):
            errors.append("d18o0 cannot be NaN")
        else:
            if abs(d18o0) > 1:  # Likely permil
                if not -60 <= d18o0 <= 10:
                    errors.append(f"d18o0={d18o0} is outside typical permil range [-60, 10]")
            else:  # Likely fraction
                if not -0.06 <= d18o0 <= 0.01:
                    errors.append(f"d18o0={d18o0} is outside typical fraction range [-0.06, 0.01]")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


if __name__ == "__main__":
    # Test validation functions
    print("Testing validation utility functions...")
    
    # Test topography validation
    valid_topo = {
        'x': np.array([0, 1000, 2000]),
        'y': np.array([0, 1000, 2000]),
        'h': np.array([[100, 150, 200], [120, 180, 220], [140, 200, 240]])
    }
    
    try:
        result = validate_topography(valid_topo)
        print(f"Valid topography test: PASSED")
    except Exception as e:
        print(f"Valid topography test: FAILED - {e}")
    
    # Test parameter validation
    params = {'U': 10.0, 'T0': 288.0, 'tau_c': 1500.0}
    bounds = {'U': (0, 50), 'T0': (250, 320), 'tau_c': (100, None)}
    result = validate_parameters(params, bounds)
    print(f"Parameter validation: {'PASSED' if result['valid'] else 'FAILED'}")
    if not result['valid']:
        print(f"  Errors: {result['errors']}")
    
    # Test grid consistency
    x = np.array([0, 1000, 2000])
    y = np.array([0, 1000])
    h = np.array([[100, 150, 200], [120, 180, 220]])
    result = check_grid_consistency(x, y, h)
    print(f"Grid consistency test: {'PASSED' if result['consistent'] else 'FAILED'}")
    
    # Test wind parameter validation
    result = validate_wind_parameters(10.0, 90.0)
    print(f"Wind validation test: {'PASSED' if result['valid'] else 'FAILED'}")
    
    # Test isotope parameter validation
    result = validate_isotope_parameters(-0.005)
    print(f"Isotope validation test: {'PASSED' if result['valid'] else 'FAILED'}")
    
    print("\nAll validation tests completed!")
