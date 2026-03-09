"""
Function to determine catchment nodes for sample points
"""

import numpy as np


def catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid):
    """
    Determine catchment nodes for sample locations.
    
    Parameters:
    -----------
    sample_x : array-like
        X coordinates of sample points
    sample_y : array-like
        Y coordinates of sample points
    sample_lc : str or array-like
        Sample type ('L' for local, 'C' for catchment)
    x : array-like
        X coordinates of the grid
    y : array-like
        Y coordinates of the grid
    h_grid : 2D array
        Elevation grid
    
    Returns:
    --------
    ij_catch : list of tuples
        List of (row, col) indices for catchment nodes
    ptr_catch : list of int
        Pointers to the start of each sample's catchment nodes in ij_catch
    """
    n_samples = len(sample_x)
    if n_samples == 0:
        return [], []
    
    # Convert sample coordinates to grid indices
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    
    x_indices = np.round((sample_x - x[0]) / dx).astype(int)
    y_indices = np.round((sample_y - y[0]) / dy).astype(int)
    
    # Ensure indices are within bounds
    x_indices = np.clip(x_indices, 0, len(x) - 1)
    y_indices = np.clip(y_indices, 0, len(y) - 1)
    
    # Create catchment nodes based on sample type
    ij_catch = []
    ptr_catch = [0]
    
    for i in range(n_samples):
        # Determine catchment type
        is_local = (isinstance(sample_lc, str) and sample_lc == 'L') or \
                   (hasattr(sample_lc, '__getitem__') and sample_lc[i] == 'L')
        
        if is_local:
            # Local catchment: only the sample grid cell
            row_idx = y_indices[i]
            col_idx = x_indices[i]
            
            if 0 <= row_idx < h_grid.shape[0] and 0 <= col_idx < h_grid.shape[1]:
                ij_catch.append((row_idx, col_idx))
        else:
            # Catchment: a small area around the sample point
            # For now, use a 3x3 area around the sample point
            row_idx = y_indices[i]
            col_idx = x_indices[i]
            
            for r_offset in [-1, 0, 1]:
                for c_offset in [-1, 0, 1]:
                    r = row_idx + r_offset
                    c = col_idx + c_offset
                    
                    if 0 <= r < h_grid.shape[0] and 0 <= c < h_grid.shape[1]:
                        ij_catch.append((r, c))
        
        # Update pointer for next sample
        ptr_catch.append(len(ij_catch))
    
    return ij_catch, ptr_catch