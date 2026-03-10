"""
Summary Figure Creation

Create comprehensive multi-panel summary figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_summary_figure(topography, result, figsize=(16, 12), save_path=None):
    """
    Create a comprehensive summary figure with multiple panels.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid'
    result : object
        OPI result object
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib Figure
    """
    from .maps import plot_topography_contour, plot_precipitation_contour
    
    x, y = topography['x'], topography['y']
    h_grid = topography['h_grid']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Topography and Precipitation
    ax1 = fig.add_subplot(gs[0, 0])
    plot_topography_contour(x, y, h_grid, ax=ax1, title='Topography (m)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_precipitation_contour(x, y, result.p_grid, ax=ax2, 
                                title='Precipitation (kg/m²/s)')
    
    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(result, 'f_m_grid') and result.f_m_grid is not None:
        im = ax3.contourf(x/1000, y/1000, result.f_m_grid, levels=20, cmap='YlGnBu')
        ax3.set_title('Moisture Ratio')
        plt.colorbar(im, ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'No moisture\ndata available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Moisture Ratio')
    
    # Row 2: Isotopes
    ax4 = fig.add_subplot(gs[1, 0])
    d2H_permil = result.d2H_grid * 1000
    vmin, vmax = np.percentile(d2H_permil, [1, 99])
    im = ax4.contourf(x/1000, y/1000, d2H_permil, levels=20, 
                      cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax4.set_title('δ²H (‰)')
    plt.colorbar(im, ax=ax4, label='‰')
    
    ax5 = fig.add_subplot(gs[1, 1])
    d18O_permil = result.d18O_grid * 1000
    vmin, vmax = np.percentile(d18O_permil, [1, 99])
    im = ax5.contourf(x/1000, y/1000, d18O_permil, levels=20,
                      cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax5.set_title('δ¹⁸O (‰)')
    plt.colorbar(im, ax=ax5, label='‰')
    
    ax6 = fig.add_subplot(gs[1, 2])
    d_excess = (result.d2H_grid - 8 * result.d18O_grid) * 1000
    vmin, vmax = np.percentile(d_excess, [1, 99])
    im = ax6.contourf(x/1000, y/1000, d_excess, levels=20,
                      cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax6.set_title('d-excess (‰)')
    plt.colorbar(im, ax=ax6, label='‰')
    
    # Row 3: Cross-sections
    mid_j = len(x) // 2
    ax7 = fig.add_subplot(gs[2, :])
    
    ax7.fill_between(y/1000, 0, h_grid[:, mid_j]/1000, 
                     alpha=0.5, color='brown', label='Topography')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(y/1000, result.p_grid[:, mid_j]*1000, 
                  'b-', label='Precipitation', linewidth=2)
    ax7_twin.set_ylabel('Precipitation (×10⁻³ kg/m²/s)', color='b')
    ax7_twin.tick_params(axis='y', labelcolor='b')
    
    ax7_twin2 = ax7.twinx()
    ax7_twin2.spines['right'].set_position(('outward', 60))
    ax7_twin2.plot(y/1000, result.d2H_grid[:, mid_j]*1000, 
                   'r--', label='δ²H', linewidth=2)
    ax7_twin2.set_ylabel('δ²H (‰)', color='r')
    ax7_twin2.tick_params(axis='y', labelcolor='r')
    
    ax7.set_xlabel('Distance (km)')
    ax7.set_ylabel('Elevation (km)')
    ax7.set_title('Cross-section (East-West through center)')
    ax7.set_ylim(bottom=0)
    ax7.legend(loc='upper left')
    
    # Add overall statistics
    stats_text = (
        f"Model Parameters:\n"
        f"  Mean Precipitation: {result.p_grid.mean():.4f} kg/m²/s\n"
        f"  Max Precipitation: {result.p_grid.max():.4f} kg/m²/s\n"
        f"  δ²H Range: {result.d2H_grid.min()*1000:.1f}‰ to {result.d2H_grid.max()*1000:.1f}‰\n"
        f"  δ¹⁸O Range: {result.d18O_grid.min()*1000:.1f}‰ to {result.d18O_grid.max()*1000:.1f}‰"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             verticalalignment='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('OPI Simulation Results', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
