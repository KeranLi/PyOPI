"""
Figure Export Utilities

Provides figure saving and printing functionality.
Matches MATLAB's printFigure.m
"""

import os
import inspect
from typing import Optional
import matplotlib.pyplot as plt


def print_figure(filepath: Optional[str] = None, dpi: int = 300,
                format: str = 'pdf', facecolor: str = 'white',
                edgecolor: str = 'none') -> str:
    """
    Save current figure in publication-ready format.
    
    Matches MATLAB's printFigure function.
    
    Parameters
    ----------
    filepath : str, optional
        Output file path. If None, uses calling function name and figure number.
    dpi : int
        Resolution in dots per inch
    format : str
        Output format ('pdf', 'png', 'svg', 'eps', etc.)
    facecolor : str
        Figure face color
    edgecolor : str
        Figure edge color
    
    Returns
    -------
    output_path : str
        Path to saved figure
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> print_figure('my_figure.pdf')
    
    >>> # Auto-generate filename from calling function
    >>> def my_analysis():
    ...     plt.figure()
    ...     plt.plot(data)
    ...     print_figure()  # Saves as my_analysis_Fig01.pdf
    """
    fig = plt.gcf()
    
    # Set figure properties
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_edgecolor(edgecolor)
    
    # Generate filename if not provided
    if filepath is None:
        # Get calling function name
        stack = inspect.stack()
        if len(stack) > 1:
            caller_name = stack[1].function
        else:
            caller_name = 'figure'
        
        fig_num = fig.number
        filepath = f"{caller_name}_Fig{fig_num:02d}.{format}"
    
    # Ensure directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    plt.savefig(filepath, format=format, dpi=dpi,
                facecolor=facecolor, edgecolor=edgecolor,
                bbox_inches='tight', pad_inches=0.1)
    
    return os.path.abspath(filepath)


def save_figure_set(figures: list, basename: str, output_dir: str = '.',
                   formats: list = ['pdf', 'png'], dpi: int = 300) -> list:
    """
    Save multiple figures with consistent naming.
    
    Parameters
    ----------
    figures : list
        List of figure objects or numbers
    basename : str
        Base name for output files
    output_dir : str
        Output directory
    formats : list
        List of formats to save
    dpi : int
        Resolution
    
    Returns
    -------
    saved_files : list
        List of saved file paths
    """
    saved = []
    
    ensure_dir(output_dir)
    
    for i, fig in enumerate(figures):
        if isinstance(fig, int):
            fig = plt.figure(fig)
        
        for fmt in formats:
            filename = f"{basename}_Fig{i+1:02d}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, format=fmt, dpi=dpi,
                       bbox_inches='tight')
            saved.append(filepath)
    
    return saved


def ensure_dir(path: str) -> str:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path
