"""
<<<<<<< HEAD:OPI_python/opi/catchment_nodes.py
Function to determine catchment nodes for sample points
"""

import numpy as np
=======
Catchment Nodes Calculation

Find nodes in h_grid that are upstream of each sample location.
Uses D8 flow routing algorithm to identify upslope catchment area.

Reference: MATLAB catchmentNodes.m by Mark Brandon, Yale University
"""

import numpy as np
from scipy import ndimage
>>>>>>> dev:opi/catchment_nodes.py


def catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid):
    """
<<<<<<< HEAD:OPI_python/opi/catchment_nodes.py
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
=======
    Find nodes in h_grid that are upstream of each sample location.
    
    Parameters
    ----------
    sample_x, sample_y : array-like
        Sample coordinates in grid units
    sample_lc : str or array-like
        Sample type ('L' for local, 'C' for catchment)
    x, y : array-like
        Grid vectors for x and y coordinates
    h_grid : 2D array
        Elevation grid (ny x nx)
    
    Returns
    -------
    ij_catch : list of tuples
        List of (row, col) indices for catchment nodes (0-indexed)
    ptr_catch : list of int
        Pointers to start of each sample's catchment nodes in ij_catch
        (1-indexed to match MATLAB convention)
    """
    ny, nx = h_grid.shape
    n_samples = len(sample_x)
    
    if n_samples == 0:
        return [], [0]
    
    # Ensure sample_lc is array-like
    if isinstance(sample_lc, str):
        sample_lc = np.array([sample_lc] * n_samples)
    else:
        sample_lc = np.asarray(sample_lc)
    
    # Initialize D8 neighbor offsets (8 directions)
    # Order: E, NE, N, NW, W, SW, S, SE
    i_d8 = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    j_d8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    
    # Fill sinks in elevation grid (using morphological reconstruction)
    # This approximates MATLAB's imfill(hSGrid, 'holes')
    h_grid_filled = ndimage.grey_closing(h_grid, size=(3, 3))
    
    # Initialize output arrays
    ij_catch = []
    ptr_catch = np.zeros(n_samples, dtype=int)
    ptr_catch[0] = 0  # MATLAB uses 1-indexed pointers
    
    # Compute grid spacing
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    
    # Iterate through samples
    for k in range(n_samples):
        # Convert sample coordinates to grid indices
        # Using interpolation to find nearest grid cell
        j0 = int(np.round((sample_x[k] - x[0]) / dx))
        i0 = int(np.round((sample_y[k] - y[0]) / dy))
        
        # Clamp to valid range
        j0 = max(0, min(j0, nx - 1))
        i0 = max(0, min(i0, ny - 1))
        
        if sample_lc[k] == 'L':
            # Local water sample - no catchment needed
            # Append one node for sample to full list
            if k < n_samples - 1:
                ptr_catch[k + 1] = ptr_catch[k] + 1
            ij_catch.append((i0, j0))
            continue
        
        # Calculate upslope nodes for catchment water sample (sample_lc == 'C')
        # Initialize logical array to track identified catchment nodes
        is_catch_sample = np.zeros((ny, nx), dtype=bool)
        is_catch_sample[i0, j0] = True
        
        # List to store catchment node indices for this sample
        ij_catch_sample = [(i0, j0)]
        
        k_c = 0  # Number of demonstrated upslope grid nodes
        n_c = 1  # Number of grid nodes to be tested
        
        # Iterate through potential catchment nodes
        while True:
            i_curr, j_curr = ij_catch_sample[k_c]
            
            # Indices for D8 neighbors
            i_neighbors = i_curr + i_d8
            j_neighbors = j_curr + j_d8
            
            # Limit search to range of h_grid
            is_inside = (i_neighbors >= 0) & (i_neighbors < ny) & \
                       (j_neighbors >= 0) & (j_neighbors < nx)
            
            # Check each valid neighbor
            for idx in range(8):
                if not is_inside[idx]:
                    continue
                    
                i_n = i_neighbors[idx]
                j_n = j_neighbors[idx]
                
                # Check if neighbor is upslope and not already in catchment
                if not is_catch_sample[i_n, j_n] and \
                   h_grid_filled[i_n, j_n] >= h_grid_filled[i_curr, j_curr]:
                    ij_catch_sample.append((i_n, j_n))
                    is_catch_sample[i_n, j_n] = True
                    n_c += 1
            
            # Check for termination
            # Termination occurs when there are no more upslope grid points
            if k_c == n_c - 1:
                break
            
            # Prepare for next loop
            k_c += 1
        
        # Append results to full list of catchment nodes
        if k < n_samples - 1:
            ptr_catch[k + 1] = ptr_catch[k] + n_c
        
        ij_catch.extend(ij_catch_sample)
    
    return ij_catch, ptr_catch.tolist()


def catchment_indices(sample_idx, ij_catch, ptr_catch):
    """
    Extract catchment indices for a specific sample.
    
    Parameters
    ----------
    sample_idx : int
        Sample index (0-indexed)
    ij_catch : list of tuples
        List of (row, col) indices for catchment nodes
    ptr_catch : list of int
        Pointers to start of each sample's catchment nodes
    
    Returns
    -------
    list of tuples
        List of (row, col) indices for the specified sample's catchment
    """
    start_idx = ptr_catch[sample_idx]
    
    if sample_idx < len(ptr_catch) - 1:
        end_idx = ptr_catch[sample_idx + 1]
    else:
        end_idx = len(ij_catch)
    
    return ij_catch[start_idx:end_idx]


if __name__ == "__main__":
    print("Testing catchment_nodes module...")
    
    # Create a simple test grid
    x = np.linspace(0, 10000, 21)
    y = np.linspace(0, 10000, 21)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple mountain
    h_grid = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / (2 * 2000**2))
    
    # Test samples
    sample_x = np.array([5000, 3000, 7000])
    sample_y = np.array([5000, 3000, 7000])
    sample_lc = np.array(['L', 'C', 'C'])  # Local, Catchment, Catchment
    
    # Run catchment_nodes
    ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
    
    print(f"Number of samples: {len(sample_x)}")
    print(f"Total catchment nodes: {len(ij_catch)}")
    print(f"Pointer array: {ptr_catch}")
    
    for i in range(len(sample_x)):
        indices = catchment_indices(i, ij_catch, ptr_catch)
        print(f"Sample {i} ({sample_lc[i]}): {len(indices)} nodes")
    
    print("\nTest completed successfully!")
>>>>>>> dev:opi/catchment_nodes.py
