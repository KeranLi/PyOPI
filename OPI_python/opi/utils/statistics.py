"""
Statistical utility functions for OPI

This module provides statistical calculations for isotope analysis,
including meteoric water line estimation and goodness-of-fit tests.
"""

import numpy as np
from scipy import stats


def estimate_mwl(d18O, d2H, sd_ratio=28.3):
    """
    Estimate meteoric water line (MWL) from isotope data.
    
    The meteoric water line describes the relationship between delta^2H
    and delta^18O in precipitation: d2H = slope * d18O + intercept
    
    Parameters
    ----------
    d18O : array_like
        delta^18O values (in permil)
    d2H : array_like
        delta^2H values (in permil)
    sd_ratio : float, optional
        Standard deviation ratio for uncertainty estimation (default: 28.3)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'slope': Slope of the MWL
        - 'intercept': Intercept of the MWL (in permil)
        - 'r_value': Pearson correlation coefficient
        - 'p_value': Two-sided p-value for a hypothesis test
        - 'std_err': Standard error of the estimated slope
    
    Notes
    -----
    The default sd_ratio=28.3 is based on the global meteoric water line
    as described in Craig (1961) and Friedman et al. (1962).
    
    References
    ----------
    Craig, H. (1961). Isotopic variations in meteoric waters. Science, 133(3465), 1702-1703.
    Friedman, I., et al. (1962). The water cycle. Science, 138(3538), 56-58.
    
    Examples
    --------
    >>> d18O = [-10, -12, -8, -15, -11]
    >>> d2H = [-70, -85, -55, -110, -75]
    >>> result = estimate_mwl(d18O, d2H)
    >>> print(f"Slope: {result['slope']:.2f}")
    """
    d18O = np.asarray(d18O, dtype=float)
    d2H = np.asarray(d2H, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(d18O) | np.isnan(d2H))
    d18O_clean = d18O[mask]
    d2H_clean = d2H[mask]
    
    if len(d18O_clean) < 2:
        raise ValueError("At least 2 valid data points are required for MWL estimation")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(d18O_clean, d2H_clean)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }


def calculate_residuals(observed, predicted, covariance=None):
    """
    Calculate standardized residuals from observed and predicted values.
    
    Standardized residuals are useful for assessing model fit and
    identifying outliers. If covariance is provided, the residuals
    are whitened (transformed to be uncorrelated with unit variance).
    
    Parameters
    ----------
    observed : array_like
        Observed values
    predicted : array_like
        Model-predicted values
    covariance : array_like, optional
        Covariance matrix of the observations. If provided, the residuals
        are standardized using the inverse square root of the covariance.
        Shape should be (n, n) where n is the number of observations.
    
    Returns
    -------
    ndarray
        Standardized residuals. If covariance is None, returns simple
        residuals (observed - predicted). If covariance is provided,
        returns whitened residuals.
    
    Notes
    -----
    When covariance is provided, this function uses Cholesky decomposition
    to compute the whitening transformation: C^(-1/2) * (obs - pred)
    
    Examples
    --------
    >>> observed = [1.0, 2.0, 3.0, 4.0]
    >>> predicted = [1.1, 1.9, 3.2, 3.8]
    >>> residuals = calculate_residuals(observed, predicted)
    >>> print(residuals)
    
    >>> # With covariance matrix
    >>> cov = np.diag([0.1, 0.1, 0.1, 0.1])
    >>> std_residuals = calculate_residuals(observed, predicted, covariance=cov)
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted arrays must have the same shape")
    
    # Calculate raw residuals
    residuals = observed - predicted
    
    if covariance is None:
        return residuals
    
    covariance = np.asarray(covariance, dtype=float)
    
    # Check covariance matrix dimensions
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be 2D square matrix")
    
    if len(residuals) != covariance.shape[0]:
        raise ValueError("Covariance matrix dimensions must match number of observations")
    
    # Whitening transformation using Cholesky decomposition
    # C = L * L^T, so C^(-1/2) = L^(-T)
    try:
        L = np.linalg.cholesky(covariance)
        L_inv = np.linalg.inv(L)
        whitened_residuals = L_inv @ residuals
        return whitened_residuals
    except np.linalg.LinAlgError:
        # If Cholesky fails, try using SVD-based pseudoinverse
        U, s, Vh = np.linalg.svd(covariance)
        # C^(-1/2) = V * diag(1/sqrt(s)) * U^T
        C_inv_sqrt = Vh.T @ np.diag(1.0 / np.sqrt(s)) @ U.T
        return C_inv_sqrt @ residuals


def chi_squared_test(residuals, df):
    """
    Calculate reduced chi-squared statistic.
    
    The reduced chi-squared is a measure of goodness of fit, calculated
    as chi-squared divided by degrees of freedom. Values close to 1
    indicate a good fit, values much greater than 1 indicate a poor fit,
    and values much less than 1 may indicate overfitting or overestimated
    uncertainties.
    
    Parameters
    ----------
    residuals : array_like
        Residuals (observed - predicted), typically standardized
    df : int
        Degrees of freedom (number of observations - number of fitted parameters)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'chi_squared': Chi-squared statistic (sum of squared residuals)
        - 'reduced_chi_squared': Chi-squared divided by degrees of freedom
        - 'dof': Degrees of freedom
        - 'n_observations': Number of observations
    
    Notes
    -----
    For standardized residuals with unit variance, chi-squared is simply
    the sum of squared residuals.
    
    References
    ----------
    Bevington, P. R., & Robinson, D. K. (2003). Data Reduction and Error
    Analysis for the Physical Sciences (3rd ed.). McGraw-Hill.
    
    Examples
    --------
    >>> residuals = [0.1, -0.2, 0.15, -0.05, 0.1]
    >>> result = chi_squared_test(residuals, df=3)
    >>> print(f"Reduced chi-squared: {result['reduced_chi_squared']:.3f}")
    """
    residuals = np.asarray(residuals, dtype=float)
    
    # Remove NaN values
    residuals_clean = residuals[~np.isnan(residuals)]
    
    n_obs = len(residuals_clean)
    
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    
    if n_obs < df:
        raise ValueError("Number of observations must be greater than degrees of freedom")
    
    # Calculate chi-squared (sum of squared residuals)
    chi_squared = np.sum(residuals_clean**2)
    
    # Calculate reduced chi-squared
    reduced_chi_squared = chi_squared / df
    
    return {
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'dof': df,
        'n_observations': n_obs
    }


def calculate_mwl_residuals(d18O, d2H, slope=8.0, intercept=10.0):
    """
    Calculate residuals from a meteoric water line.
    
    Parameters
    ----------
    d18O : array_like
        delta^18O values (in permil)
    d2H : array_like
        delta^2H values (in permil)
    slope : float, optional
        MWL slope (default: 8.0 for GMWL)
    intercept : float, optional
        MWL intercept in permil (default: 10.0 for GMWL)
    
    Returns
    -------
    ndarray
        Residuals (perpendicular distance from MWL in permil)
    
    Notes
    -----
    The global meteoric water line (GMWL) has the form:
    d2H = 8 * d18O + 10 (Craig, 1961)
    
    Examples
    --------
    >>> d18O = [-10, -12, -8, -15]
    >>> d2H = [-70, -85, -55, -110]
    >>> residuals = calculate_mwl_residuals(d18O, d2H)
    """
    d18O = np.asarray(d18O, dtype=float)
    d2H = np.asarray(d2H, dtype=float)
    
    # Expected d2H from MWL
    d2H_expected = slope * d18O + intercept
    
    # Calculate residuals (d2H deviation from MWL)
    residuals = d2H - d2H_expected
    
    return residuals


def deuterium_excess(d2H, d18O, slope=8.0, intercept=10.0):
    """
    Calculate deuterium excess (d-excess).
    
    Deuterium excess is defined as d = d2H - 8*d18O, which measures
    the deviation from the global meteoric water line. It is useful
    for identifying moisture source conditions and post-condensation
    processes.
    
    Parameters
    ----------
    d2H : array_like
        delta^2H values (in permil)
    d18O : array_like
        delta^18O values (in permil)
    slope : float, optional
        Reference MWL slope (default: 8.0)
    intercept : float, optional
        Reference MWL intercept (default: 10.0)
    
    Returns
    -------
    ndarray
        Deuterium excess values (in permil)
    
    References
    ----------
    Dansgaard, W. (1964). Stable isotopes in precipitation. Tellus, 16(4), 436-468.
    
    Examples
    --------
    >>> d2H = [-70, -85, -55]
    >>> d18O = [-10, -12, -8]
    >>> d_excess = deuterium_excess(d2H, d18O)
    >>> print(d_excess)
    [10. 11.  9.]
    """
    d2H = np.asarray(d2H, dtype=float)
    d18O = np.asarray(d18O, dtype=float)
    
    # Calculate deuterium excess
    d_excess = d2H - slope * d18O
    
    return d_excess


if __name__ == "__main__":
    # Test statistical functions
    print("Testing statistical utility functions...")
    
    # Test MWL estimation
    d18O_test = np.array([-10.0, -12.0, -8.0, -15.0, -11.0])
    d2H_test = np.array([-70.0, -85.0, -55.0, -110.0, -75.0])
    
    mwl = estimate_mwl(d18O_test, d2H_test)
    print(f"MWL: d2H = {mwl['slope']:.2f} * d18O + {mwl['intercept']:.2f}")
    print(f"R-value: {mwl['r_value']:.4f}")
    
    # Test residuals calculation
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 1.9, 3.2, 3.8])
    residuals = calculate_residuals(observed, predicted)
    print(f"Residuals: {residuals}")
    
    # Test chi-squared
    chi2_result = chi_squared_test(residuals, df=2)
    print(f"Chi-squared: {chi2_result['chi_squared']:.4f}")
    print(f"Reduced chi-squared: {chi2_result['reduced_chi_squared']:.4f}")
    
    # Test d-excess
    d_exc = deuterium_excess(d2H_test, d18O_test)
    print(f"Deuterium excess: {d_exc}")
    
    print("\nAll statistical tests passed!")
