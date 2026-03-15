"""
Cross-section visualization for OPI model

Generates cross-section plots (Fig 05 and Fig 06 in MATLAB)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot_cross_section(s_path, z_rho_c, rho_c, h_l_path, z_248_path, z_268_path,
                       s_l_section, z_l_section, s_prec_lines, h_prec_lines,
                       p_section, d2h_section, d2h0_section,
                       s_limits, h_l_max, azimuth,
                       title='Cross Section', save_path=None):
    """
    Create cross-section plot with cloud water density, precipitation rate, 
    and isotope profiles.
    
    Parameters
    ----------
    s_path : array
        Section path coordinates (m)
    z_rho_c : array
        Vertical grid for cloud water density (m)
    rho_c : 2D array
        Cloud water density (kg/m^3)
    h_l_path : array
        Topography height along section (m)
    z_248_path, z_268_path : array
        Heights of 248K and 268K isotherms (WBF zone) (m)
    s_l_section, z_l_section : array
        Streamline positions along section
    s_prec_lines, h_prec_lines : array
        Precipitation line positions
    p_section : array
        Precipitation rate along section (mm/hr)
    d2h_section : array
        d2H along section (fraction)
    d2h0_section : array
        Base d2H along section (fraction)
    s_limits : tuple
        (s_min, s_max) section limits (m)
    h_l_max : float
        Maximum section height (m)
    azimuth : float
        Wind direction (degrees)
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Create three subplots with shared x-axis
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Ensure z_rho_c is 1D (if passed as 2D, take first column assuming uniform z levels)
    if z_rho_c.ndim > 1:
        z_rho_c = z_rho_c[:, 0]
    
    # Convert to km
    s_path_km = s_path * 1e-3
    z_rho_c_km = z_rho_c * 1e-3
    h_l_path_km = h_l_path * 1e-3
    z_248_km = z_248_path * 1e-3 if z_248_path is not None else None
    z_268_km = z_268_path * 1e-3 if z_268_path is not None else None
    s_limits_km = (s_limits[0] * 1e-3, s_limits[1] * 1e-3)
    h_l_max_km = h_l_max * 1e-3
    
    # Top panel: Cloud water density
    # Create colormap: light blue for negative, grayscale for positive
    if rho_c is not None and rho_c.size > 0:
        # Create custom colormap: light blue for negative (first row), grayscale for positive
        # MATLAB uses: colormap([0.686, 1, 1; repmat(linspace(0.8, 0, 100)', 1, 3)]);
        n_gray = 100
        gray_colors = np.column_stack([np.linspace(0.8, 0, n_gray)] * 3)
        blue_color = np.array([[0.686, 1, 1]])  # Light blue for negative values
        custom_cmap = np.vstack([blue_color, gray_colors])
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(custom_cmap)
        
        # Create 2D meshgrid for pcolor
        S, Z = np.meshgrid(s_path, z_rho_c)
        
        # Create mask for section area (above topography and within limits)
        is_section = ((S >= s_limits[0]) & (S <= s_limits[1]) & 
                      (Z >= 0) & (Z <= h_l_max))
        
        # For each column, mask values below topography
        for j in range(len(s_path)):
            mask_below_topo = Z[:, j] < h_l_path[j]
            is_section[:, j] = is_section[:, j] & ~mask_below_topo
        
        # Find max positive value for color limits
        rho_c_masked = np.where(is_section, rho_c, np.nan)
        c_max = np.nanmax(rho_c_masked) if np.any(is_section) else np.nanmax(rho_c)
        
        # Plot cloud water density
        # Use masked array to handle negative values (will use first color = light blue)
        rho_c_plot = np.where(rho_c > 0, rho_c, -0.1 * c_max)  # Map negatives to near-zero
        
        im = ax1.pcolormesh(s_path_km, z_rho_c_km, rho_c_plot, 
                           shading='auto', cmap=cmap, vmin=0, vmax=c_max)
        ax1.set_ylim([0, h_l_max_km])
        
        # Plot WBF zone (freezing transition)
        if z_248_path is not None and z_268_path is not None:
            ax1.plot(s_path_km, z_248_km, 'b-', linewidth=1.5, label='248K')
            ax1.plot(s_path_km, z_268_km, 'b-', linewidth=1.5, label='268K')
    
    # Plot topography
    topo_x = np.concatenate([[s_path_km[0]], s_path_km, [s_path_km[-1]]])
    topo_y = np.concatenate([[0], h_l_path_km, [0]])
    ax1.fill(topo_x, topo_y, color=[0.82, 0.70, 0.55], edgecolor='black', linewidth=0.5)
    
    # Plot streamlines
    if s_l_section is not None and z_l_section is not None:
        ax1.plot(s_l_section * 1e-3, z_l_section * 1e-3, 'k-', linewidth=0.5)
    
    # Plot precipitation lines
    if s_prec_lines is not None and h_prec_lines is not None:
        ax1.plot(s_prec_lines * 1e-3, h_prec_lines * 1e-3, 'k:', linewidth=1)
    
    ax1.set_ylabel('Elevation (km)', fontsize=10)
    ax1.set_title(title, fontsize=12)
    ax1.set_xlim(s_limits_km)
    ax1.tick_params(labelbottom=False)
    
    # Middle panel: Precipitation rate
    if p_section is not None and p_section.size > 0:
        ax2.plot(s_path_km, p_section * 3.6e3, 'k-', linewidth=2)
        ax2.set_ylabel('Precipitation\nRate (mm/hr)', fontsize=10)
        ax2.set_ylim([0, np.max(p_section) * 1.05 * 3.6e3])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelbottom=False)
        ax2.set_xlim(s_limits_km)
    
    # Bottom panel: Isotope values
    if d2h_section is not None and d2h_section.size > 0:
        ax3.plot(s_path_km, d2h0_section * 1e3, '--k', linewidth=1, label='d2H0')
        ax3.plot(s_path_km, d2h_section * 1e3, '-k', linewidth=2, label='d2H')
        ax3.set_ylabel('d2H (permil)', fontsize=10)
        ax3.set_xlabel('Section Distance (km)', fontsize=12)
        ax3.set_xlim(s_limits_km)
        ax3.grid(True, alpha=0.3)
        
        # Set y-limits with some padding
        d2h_min = np.min(d2h_section) - 10e-3
        d2h_max = np.max(np.concatenate([d2h_section, [10e-3]]))
        ax3.set_ylim([d2h_min * 1e3, d2h_max * 1e3])
    
    # Add wind direction arrow annotation
    if ax2 is not None:
        ax2.annotate('', xy=(0.95, 0.5), xytext=(0.85, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax2.text(0.9, 0.6, f'{azimuth:.0f}°', transform=ax2.transAxes,
                fontsize=10, ha='center')
    
    # Use subplots_adjust instead of tight_layout for gridspec figures
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax3)


def extract_cross_section(data_grid, lon, lat, section_lon0, section_lat0, 
                          azimuth, width=50000, resolution=100):
    """
    Extract a cross-section from a 2D grid.
    
    Parameters
    ----------
    data_grid : 2D array
        Data to extract
    lon, lat : 1D arrays
        Coordinate vectors
    section_lon0, section_lat0 : float
        Section origin
    azimuth : float
        Section direction (perpendicular to wind)
    width : float
        Section width (m)
    resolution : int
        Number of points along section
    
    Returns
    -------
    dict with section data
    """
    # Section direction (perpendicular to wind, i.e., azimuth + 90)
    section_az = azimuth + 90
    section_az_rad = np.deg2rad(section_az)
    
    # Section coordinates
    s = np.linspace(-width/2, width/2, resolution)
    
    # Convert to lon/lat
    # Approximation: 1 degree = 111 km
    lon_section = section_lon0 + s * np.cos(section_az_rad) / 111000
    lat_section = section_lat0 + s * np.sin(section_az_rad) / 111000
    
    # Interpolate data to section
    from scipy.interpolate import RegularGridInterpolator
    interpolator = RegularGridInterpolator((lat, lon), data_grid, 
                                          method='linear', bounds_error=False)
    
    points = np.column_stack([lat_section, lon_section])
    data_section = interpolator(points)
    
    return {
        's': s,
        'lon': lon_section,
        'lat': lat_section,
        'data': data_section
    }
