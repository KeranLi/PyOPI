"""
Custom Colormaps for OPI

Provides specialized colormaps including Haxby (oceanographic) and
cmapscale for data-driven color scaling.
"""

import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import Tuple, Optional, List


def haxby(n_colors: int = 256) -> ListedColormap:
    """
    Haxby colormap for oceanographic/bathymetry data.
    
    Based on W.F. Haxby's Gravity field of World's oceans, 1985.
    Commonly used for geoid, gravity, and bathymetry maps.
    
    Parameters
    ----------
    n_colors : int
        Number of colors in colormap
    
    Returns
    -------
    cmap : ListedColormap
        Haxby colormap
    """
    # Haxby color palette (11 base colors)
    colors = np.array([
        [37, 57, 175],      # Deep blue
        [40, 127, 251],     # Medium blue
        [50, 190, 255],     # Light blue
        [106, 235, 255],    # Cyan
        [138, 236, 174],    # Green-cyan
        [205, 255, 162],    # Light green
        [240, 236, 121],    # Yellow
        [255, 189, 87],     # Orange
        [255, 161, 68],     # Dark orange
        [255, 186, 133],    # Light brown
        [255, 255, 255],    # White
    ]) / 255.0
    
    # Interpolate to desired number of colors
    n_base = len(colors)
    r0 = np.linspace(0, 1, n_base)
    r1 = np.linspace(0, 1, n_colors)
    
    red = np.interp(r1, r0, colors[:, 0])
    green = np.interp(r1, r0, colors[:, 1])
    blue = np.interp(r1, r0, colors[:, 2])
    
    cmap_data = np.column_stack([red, green, blue])
    
    return ListedColormap(cmap_data, name='haxby')


def cmapscale(
    z: np.ndarray,
    cmap0: np.ndarray,
    factor: float = 1.0,
    z0: Optional[float] = None,
    n_ticks: Optional[int] = None,
    n_round: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """
    Rescale colormap to better match cumulative distribution of data.
    
    Matches MATLAB's cmapscale function. Provides data-driven color scaling
    for improved visualization of data distributions.
    
    Parameters
    ----------
    z : ndarray
        Data values to map
    cmap0 : ndarray
        Original colormap (n_colors x 3 RGB array)
    factor : float
        Contrast adjustment (0 to 1). 
        0 = linear mapping (no change)
        1 = uniform mapping (histogram equalization)
    z0 : float, optional
        Center value for diverging colormaps.
        If provided, centers colormap at this value.
    n_ticks : int, optional
        Number of colorbar ticks to generate
    n_round : int
        Decimal position for rounding tick labels
    
    Returns
    -------
    cmap1 : ndarray
        Rescaled colormap
    ticks : ndarray or None
        Tick positions for colorbar (if n_ticks provided)
    tick_labels : list or None
        Tick labels for colorbar (if n_ticks provided)
    """
    if not (0 <= factor <= 1):
        raise ValueError("factor must be in range [0, 1]")
    
    if cmap0.shape[1] != 3:
        raise ValueError("cmap0 must have 3 columns (RGB)")
    
    # Convert contrast factor to stretching parameter
    s = np.tan(np.pi * factor / 2)
    s = min(s, 1e4)  # Avoid numerical issues
    
    # Remove NaNs and flatten
    z_valid = z[~np.isnan(z)].flatten()
    
    z_min = np.min(z_valid)
    z_max = np.max(z_valid)
    
    # Check for uniform values
    if (z_max - z_min) < 10 * np.finfo(float).eps:
        print("cmapscale: z values are equal. Returning input colormap.")
        if n_ticks is not None:
            ticks = np.linspace(z_min, z_max, n_ticks)
            tick_labels = [f"{t:.{n_round}f}" for t in ticks]
            return cmap0, ticks, tick_labels
        return cmap0, None, None
    
    # Handle centering
    if z0 is not None:
        if z0 <= z_min:
            # Use upper half of colormap
            n_colors = cmap0.shape[0]
            indices = np.linspace((n_colors - 1) / 2, n_colors - 1, n_colors)
            cmap0 = np.array([np.interp(i, range(n_colors), cmap0[:, c]) 
                            for c in range(3)]).T
            z0 = None
        elif z0 >= z_max:
            # Use lower half of colormap
            n_colors = cmap0.shape[0]
            indices = np.linspace(0, (n_colors - 1) / 2, n_colors)
            cmap0 = np.array([np.interp(i, range(n_colors), cmap0[:, c])
                            for c in range(3)]).T
            z0 = None
    
    # Create probability scales
    # Linear probability
    p_linear = (z_valid - z_min) / (z_max - z_min)
    
    # Get unique values (within tolerance)
    tolerance = 1e-7
    p_unique, inverse = np.unique(np.round(p_linear / tolerance) * tolerance, 
                                   return_inverse=True)
    m = len(p_unique)
    
    # Create uniform probability scale
    p_uniform = np.arange(m) / (m - 1)
    
    # Map back to original z values
    z_unique = np.array([np.median(z_valid[inverse == i]) for i in range(m)])
    
    # Create stretched probability scale
    if z0 is None:
        # No centering
        p_stretch = (p_unique + s**2 * p_uniform) / (1 + s**2)
    else:
        # Center at z0
        # Find index closest to z0
        i0 = np.argmin(np.abs(z_unique - z0))
        p_linear_0 = p_unique[i0]
        p_uniform_0 = p_uniform[i0]
        
        p_stretch = np.zeros(m)
        # Lower half: 0 to 0.5
        p_stretch[:i0+1] = 0.5 * (p_unique[:i0+1] / p_linear_0 + 
                                   s**2 * p_uniform[:i0+1] / p_uniform_0) / (1 + s**2)
        # Upper half: 0.5 to 1
        if i0 < m - 1:
            p_stretch[i0:] = 0.5 + 0.5 * ((p_unique[i0:] - p_linear_0) / (1 - p_linear_0) +
                                           s**2 * (p_uniform[i0:] - p_uniform_0) / (1 - p_uniform_0)) / (1 + s**2)
    
    # Renormalize
    p_unique = p_unique / p_unique[-1]
    p_stretch = p_stretch / p_stretch[-1]
    
    # Interpolate new colormap
    n_map = cmap0.shape[0]
    p_linear_map = np.linspace(0, 1, n_map)
    p_stretch_map = np.interp(p_linear_map, p_unique, p_stretch)
    
    # Map stretch back to linear for colormap indexing
    cmap1 = np.array([np.interp(p_stretch_map, p_linear_map, cmap0[:, c])
                      for c in range(3)]).T
    
    # Calculate ticks
    ticks = None
    tick_labels = None
    if n_ticks is not None and n_ticks > 0:
        # Linear scale positions on colorbar
        p_s_ticks = np.linspace(0, 1, n_ticks)
        ticks = np.interp(p_s_ticks, [0, 1], [z_min, z_max])
        
        # Find corresponding z values in stretched distribution
        p_l_ticks = np.interp(p_s_ticks, p_stretch, p_unique)
        tick_values = np.interp(p_l_ticks, [0, 1], [z_min, z_max])
        
        # Round
        tick_values = np.round(tick_values, n_round)
        tick_values[tick_values == 0] = 0  # Fix negative zero
        
        format_str = f"{{:.{n_round}f}}"
        tick_labels = [format_str.format(v) for v in tick_values]
    
    return cmap1, ticks, tick_labels


def cool_warm(n_colors: int = 256) -> ListedColormap:
    """
    Cool-warm diverging colormap.
    
    Good for data centered at zero (e.g., anomalies).
    
    Parameters
    ----------
    n_colors : int
        Number of colors
    
    Returns
    -------
    cmap : ListedColormap
        Cool-warm colormap
    """
    # Use matplotlib's coolwarm as base
    from matplotlib import cm
    return cm.get_cmap('coolwarm', n_colors)


def create_centered_colormap(cmap_name: str = 'RdBu_r', n_colors: int = 256,
                             center: float = 0.0, vmin: float = -1.0,
                             vmax: float = 1.0) -> ListedColormap:
    """
    Create a colormap centered at a specific value.
    
    Parameters
    ----------
    cmap_name : str
        Base colormap name
    n_colors : int
        Number of colors
    center : float
        Center value
    vmin, vmax : float
        Data range
    
    Returns
    -------
    cmap : ListedColormap
        Centered colormap
    """
    from matplotlib import cm
    
    base_cmap = cm.get_cmap(cmap_name, n_colors)
    
    # Calculate relative position of center
    if vmax == vmin:
        center_frac = 0.5
    else:
        center_frac = (center - vmin) / (vmax - vmin)
    
    # Create asymmetric sampling
    if center_frac <= 0:
        indices = np.linspace(0.5, 1, n_colors)
    elif center_frac >= 1:
        indices = np.linspace(0, 0.5, n_colors)
    else:
        # Split colormap
        n_lower = int(n_colors * center_frac)
        n_upper = n_colors - n_lower
        
        lower = np.linspace(0, 0.5, n_lower) if n_lower > 0 else []
        upper = np.linspace(0.5, 1, n_upper) if n_upper > 0 else []
        indices = np.concatenate([lower, upper])
    
    colors = base_cmap(indices)
    return ListedColormap(colors, name=f'{cmap_name}_centered')


# Convenience dictionary of available colormaps
OPI_COLORMAPS = {
    'haxby': haxby,
    'coolwarm': cool_warm,
}
