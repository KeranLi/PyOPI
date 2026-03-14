"""
Isotope Analysis Module

Provides functions for analyzing stable isotope data, including
meteoric water line (MWL) estimation using total least squares.

Matches MATLAB's private/estimateMWL.m
"""

import numpy as np
from typing import Tuple


def estimate_meteoric_water_line(
    d18o: np.ndarray,
    d2h: np.ndarray,
    sd_res_ratio: float = 3.0
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Estimate meteoric water line (MWL) using total least squares.
    
    Matches MATLAB's estimateMWL.m function.
    
    The data are trimmed to remove samples with dExcess < 5 per mil,
    which works well to identify precipitation samples that have been
    altered by evaporation. The total least squares method is used
    because the covariance for the isotope data has principal directions
    that are aligned with the MWL.
    
    Parameters
    ----------
    d18o : ndarray
        delta^18O values (fraction or per mil)
    d2h : ndarray
        delta^2H values (fraction or per mil)
    sd_res_ratio : float
        Estimated standard-deviation ratio for isotopic variation at
        a location due to seasonal variation, where the ratio equals
        maximum sd / minimum sd (default: 3.0)
    
    Returns
    -------
    b_hat : ndarray
        Coefficients for best fit [intercept, slope], where d2H = b_hat[0] + b_hat[1]*d18O
    sd_min : float
        Minimum standard deviation (perpendicular to MWL)
    sd_max : float
        Maximum standard deviation (along MWL)
    cov : ndarray
        Estimated 2x2 covariance matrix for isotopes due to seasonal variation
    i_fit : ndarray
        Logical array indicating samples used for fit (dExcess > 5)
    
    References
    ----------
    .. [1] MATLAB OPI 3.7 - private/estimateMWL.m (Mark Brandon, Yale University)
    """
    # Auto-detect units and convert to fraction if needed
    d18o = np.asarray(d18o)
    d2h = np.asarray(d2h)
    
    # Check if values are in permil (|value| > 0.1) or fraction
    if np.any(np.abs(d18o) > 0.1):
        # Convert from permil to fraction
        d18o = d18o * 1e-3
        d2h = d2h * 1e-3
    
    # Identify samples with dExcess < 5 per mil
    # dExcess = d2H - 8*d18O (in permil)
    d_excess = (d2h - 8 * d18o) * 1000  # Convert to permil for threshold
    i_fit = d_excess > 5.0
    n_fit = np.sum(i_fit)
    
    if n_fit < 2:
        raise ValueError(f"Insufficient data points after filtering: {n_fit} < 2")
    
    # Estimate a straight-line fit using total least squares
    d2h_mean = np.mean(d2h[i_fit])
    d18o_mean = np.mean(d18o[i_fit])
    
    # Build centered data matrix
    A = np.column_stack([
        d18o[i_fit] - d18o_mean,
        d2h[i_fit] - d2h_mean
    ])
    
    # Eigen decomposition of covariance matrix
    # A'*A/(nFit - 1) gives covariance matrix
    cov_matrix = A.T @ A / (n_fit - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues in ascending order
    idx_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]
    
    # Standard deviations (sqrt of eigenvalues)
    # Clip to handle numerical precision issues (negative eigenvalues should be 0)
    sd_min = np.sqrt(max(eigenvalues[0], 0))
    sd_max = np.sqrt(max(eigenvalues[1], 0))
    
    # Slope of MWL = V(2,2)/V(1,2) (components of major eigenvector)
    # The major eigenvector (corresponding to largest eigenvalue) gives the line direction
    slope = eigenvectors[1, 1] / eigenvectors[0, 1]
    
    # Recast as: d2H = d2HMean - slope*d18OMean + slope*d18O
    intercept = d2h_mean - slope * d18o_mean
    
    b_hat = np.array([intercept, slope])
    
    # Create estimated covariance matrix for isotope residuals
    # Organize so that V(d2H) and V(d18O) are 1st and 2nd on the diagonal,
    # which is required for the chiR2 calculation in calc_OneWind
    # and calc_TwoWinds functions.
    D_diag = np.array([eigenvalues[0], sd_res_ratio**2 * eigenvalues[0]])
    cov = eigenvectors @ np.diag(D_diag) @ eigenvectors.T
    
    # Reorder: [V(d2H), cov(d2H,d18O); cov(d18O,d2H), V(d18O)]
    cov = np.array([
        [cov[1, 1], cov[1, 0]],
        [cov[0, 1], cov[0, 0]]
    ])
    
    # Ensure diagonal elements are non-negative (handle numerical precision issues)
    cov[0, 0] = max(cov[0, 0], 0)
    cov[1, 1] = max(cov[1, 1], 0)
    
    return b_hat, sd_min, sd_max, cov, i_fit


def deuterium_excess(d2h: np.ndarray, d18o: np.ndarray) -> np.ndarray:
    """
    Calculate deuterium excess.
    
    Parameters
    ----------
    d2h : ndarray
        delta^2H values (fraction or permil)
    d18o : ndarray
        delta^18O values (fraction or permil)
    
    Returns
    -------
    d_excess : ndarray
        Deuterium excess values in permil
    """
    d2h = np.asarray(d2h)
    d18o = np.asarray(d18o)
    
    # Auto-detect if values are in permil or fraction
    if np.any(np.abs(d2h) > 0.1):  # Likely permil
        return d2h - 8 * d18o
    else:  # Fraction
        return (d2h - 8 * d18o) * 1000


def meteoric_water_line(d18o: np.ndarray, slope: float = 8.0, intercept: float = 10.0) -> np.ndarray:
    """
    Calculate expected d2H from d18O using meteoric water line.
    
    Parameters
    ----------
    d18o : ndarray
        delta^18O values
    slope : float
        MWL slope (default: 8.0)
    intercept : float
        MWL intercept in permil (default: 10.0)
    
    Returns
    -------
    d2h_expected : ndarray
        Expected d2H values
    """
    d18o = np.asarray(d18o)
    
    # Auto-detect units
    if np.any(np.abs(d18o) > 0.1):  # Permil
        return slope * d18o + intercept
    else:  # Fraction
        return (slope * d18o * 1000 + intercept) / 1000
