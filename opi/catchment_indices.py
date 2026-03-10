"""
Function to extract catchment indices for a specific sample
"""

def catchment_indices(sample_idx, ij_catch, ptr_catch):
    """
    Extract indices for a specific sample's catchment.
    
    Parameters:
    -----------
    sample_idx : int
        Index of the sample (0-based)
    ij_catch : list of tuples
        List of (row, col) indices for catchment nodes
    ptr_catch : list of int
        Pointers to the start of each sample's catchment nodes in ij_catch
    
    Returns:
    --------
    indices : list of tuples
        List of (row, col) indices for the specified sample's catchment
    """
    if sample_idx < 0 or sample_idx >= len(ptr_catch) - 1:
        raise IndexError("sample_idx is out of range")
    
    start_idx = ptr_catch[sample_idx]
    end_idx = ptr_catch[sample_idx + 1]
    
    return ij_catch[start_idx:end_idx]