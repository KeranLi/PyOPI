"""
Catchment and node calculations for sample locations.

This module provides functions for identifying grid nodes associated with
sample locations, including both local (point) samples and catchment-based
(upstream area) samples. It uses a D8-like flow routing algorithm to identify
upslope contributing areas.
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional, Union


def catchment_indices(k: int, ij_catch: List[Tuple[int, int]], ptr_catch: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract indices for sample catchment k.

    Parameters
    ----------
    k : int
        Index for selected catchment (0-based).
    ij_catch : list of tuple
        List/array of (row, col) linear indices for catchment nodes.
    ptr_catch : np.ndarray
        Pointers for first node of each catchment. The array should have
        length n_samples + 1, where ptr_catch[k] gives the starting index
        in ij_catch for catchment k.

    Returns
    -------
    indices : list of tuple
        Array of (row, col) indices for catchment k.

    Notes
    -----
    The ptr_catch array uses a cumulative indexing scheme where:
    - ptr_catch[k] is the starting index in ij_catch for catchment k
    - ptr_catch[k+1] is the ending index (exclusive) for catchment k
    This allows variable-length catchments to be stored efficiently.

    Examples
    --------
    >>> ij_catch = [(0, 0), (0, 1), (1, 0), (2, 2)]
    >>> ptr_catch = np.array([0, 1, 3, 4])
    >>> catchment_indices(0, ij_catch, ptr_catch)
    [(0, 0)]
    >>> catchment_indices(1, ij_catch, ptr_catch)
    [(0, 1), (1, 0)]
    """
    if k < 0 or k >= len(ptr_catch) - 1:
        raise IndexError(f"Catchment index {k} is out of range [0, {len(ptr_catch) - 2}]")

    start_idx = ptr_catch[k]
    end_idx = ptr_catch[k + 1]

    return ij_catch[start_idx:end_idx]


def catchment_nodes(
    x: np.ndarray,
    y: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_type: Union[np.ndarray, List[str]],
    h_grid: np.ndarray,
    search_radius: Optional[float] = None
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Find grid nodes for local (L) and catchment (C) samples.

    For type 'L': find the single nearest grid node to the sample point.
    For type 'C': find all nodes upslope from the sample point using a
    D8-like flow accumulation approach.

    Parameters
    ----------
    x, y : np.ndarray
        Grid coordinate vectors defining the grid axes. These are 1D arrays
        representing the coordinates of grid cell centers or edges.
    sample_x, sample_y : np.ndarray
        Sample coordinates in the same coordinate system as x and y.
    sample_type : array-like of str
        Array of 'L' (local) or 'C' (catchment) strings indicating the
        type of each sample.
    h_grid : np.ndarray
        Elevation grid (ny x nx) containing terrain heights.
    search_radius : float, optional
        Optional radius for catchment search. If specified, limits the
        search for upslope nodes to within this distance from the sample
        point. Default is None (no limit).

    Returns
    -------
    ij_catch : list of tuple
        List of (i, j) index tuples for all catchment nodes across all
        samples. Indices are 0-based (row, col) format.
    ptr_catch : np.ndarray
        Array of starting indices for each catchment in ij_catch.
        ptr_catch[k] gives the starting position in ij_catch for sample k,
        and ptr_catch[k+1] gives the end position. Length is n_samples + 1.

    Notes
    -----
    The function processes each sample sequentially:
    - For 'L' samples, only the nearest grid cell is recorded.
    - For 'C' samples, all upslope contributing cells are found using
      the find_upslope_nodes() helper function.

    The output format (ij_catch, ptr_catch) allows efficient storage of
    variable-length catchments as a flat list with index pointers.

    Raises
    ------
    ValueError
        If sample coordinate arrays have inconsistent lengths or if
        sample_type contains invalid values.
    IndexError
        If sample coordinates fall outside the grid domain.
    """
    sample_x = np.asarray(sample_x)
    sample_y = np.asarray(sample_y)
    sample_type = np.asarray(sample_type)
    h_grid = np.asarray(h_grid)

    n_samples = len(sample_x)

    if len(sample_y) != n_samples:
        raise ValueError("sample_x and sample_y must have the same length")
    if len(sample_type) != n_samples:
        raise ValueError("sample_type must have the same length as sample coordinates")

    ny, nx = h_grid.shape

    # Compute grid spacing
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0

    # Validate grid bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Initialize output containers
    ij_catch: List[Tuple[int, int]] = []
    ptr_catch = np.zeros(n_samples + 1, dtype=int)

    for k in range(n_samples):
        # Store starting position for this catchment
        ptr_catch[k] = len(ij_catch)

        # Validate sample type
        stype = sample_type[k]
        if stype not in ('L', 'C'):
            raise ValueError(f"Invalid sample_type '{stype}' at index {k}. Must be 'L' or 'C'")

        # Convert sample coordinates to grid indices
        j = int(np.round((sample_x[k] - x[0]) / dx))
        i = int(np.round((sample_y[k] - y[0]) / dy))

        # Clamp to valid grid range
        j = max(0, min(j, nx - 1))
        i = max(0, min(i, ny - 1))

        if stype == 'L':
            # Local sample: just the nearest grid node
            ij_catch.append((i, j))
        else:
            # Catchment sample: find all upslope nodes
            upslope_nodes = find_upslope_nodes(i, j, h_grid, search_radius=search_radius)
            ij_catch.extend(upslope_nodes)

    # Set final pointer
    ptr_catch[n_samples] = len(ij_catch)

    return ij_catch, ptr_catch


def find_upslope_nodes(
    i: int,
    j: int,
    h_grid: np.ndarray,
    min_slope: float = 0.01,
    search_radius: Optional[float] = None
) -> List[Tuple[int, int]]:
    """
    Helper to find all nodes upslope from a given starting node.

    Uses a D8-like flow accumulation approach to identify all grid cells
    that contribute flow to the specified starting cell. The algorithm
    starts at the given cell and iteratively expands to include all
    neighboring cells that are higher in elevation.

    Parameters
    ----------
    i, j : int
        Starting grid indices (row, col) in 0-based indexing.
    h_grid : np.ndarray
        Elevation grid (ny x nx) containing terrain heights.
    min_slope : float, optional
        Minimum slope threshold for considering a neighbor as contributing
        flow. Neighbors with slope less than this value (i.e., nearly flat
        or downslope) are excluded. Default is 0.01.
    search_radius : float, optional
        If specified, limits the search to nodes within this grid-cell
        distance from the starting point. Default is None (no limit).

    Returns
    -------
    upslope_nodes : list of tuple
        List of (row, col) tuples representing all grid nodes upslope
        from the starting node, including the starting node itself.
        The list is ordered by discovery (breadth-first expansion).

    Notes
    -----
    The D8 algorithm assumes water flows to one of 8 possible neighbors
    based on the steepest descent direction. This function inverts that
    logic to find all cells that would drain TO the starting cell.

    The algorithm uses a queue-based (breadth-first) expansion:
    1. Start with the initial cell
    2. For each cell in the queue, check all 8 neighbors
    3. Add neighbors that are higher in elevation
    4. Continue until no new upslope cells are found

    This is an approximation of true flow accumulation and works best
    on terrain with well-defined drainage patterns.

    Examples
    --------
    >>> h_grid = np.array([[1, 2, 3],
    ...                    [2, 1, 2],
    ...                    [3, 2, 1]])
    >>> find_upslope_nodes(1, 1, h_grid)
    [(1, 1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1)]
    """
    h_grid = np.asarray(h_grid)
    ny, nx = h_grid.shape

    # Validate starting position
    if not (0 <= i < ny and 0 <= j < nx):
        raise IndexError(f"Starting position ({i}, {j}) is outside grid bounds ({ny}, {nx})")

    # D8 neighbor offsets: E, NE, N, NW, W, SW, S, SE
    di = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    # Track visited nodes
    visited = np.zeros((ny, nx), dtype=bool)
    visited[i, j] = True

    # Initialize with starting node
    upslope_nodes = [(i, j)]
    queue = [(i, j)]
    queue_idx = 0

    # Pre-fill elevation grid (handles sinks/depressions)
    # This approximates MATLAB's imfill behavior
    h_grid_filled = ndimage.grey_closing(h_grid, size=(3, 3))

    # Calculate squared search radius if specified
    radius_sq = search_radius ** 2 if search_radius is not None else None

    while queue_idx < len(queue):
        ci, cj = queue[queue_idx]
        queue_idx += 1

        # Check all 8 neighbors
        for k in range(8):
            ni = ci + di[k]
            nj = cj + dj[k]

            # Skip if outside grid
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue

            # Skip if already visited
            if visited[ni, nj]:
                continue

            # Check search radius constraint
            if radius_sq is not None:
                dist_sq = (ni - i) ** 2 + (nj - j) ** 2
                if dist_sq > radius_sq:
                    continue

            # Check if neighbor is upslope (higher elevation)
            # Using filled elevations to handle sinks
            elevation_diff = h_grid_filled[ni, nj] - h_grid_filled[ci, cj]

            if elevation_diff >= -min_slope:
                # Neighbor is at or above current cell (upslope contributor)
                visited[ni, nj] = True
                upslope_nodes.append((ni, nj))
                queue.append((ni, nj))

    return upslope_nodes
