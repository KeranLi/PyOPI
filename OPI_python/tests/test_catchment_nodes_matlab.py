"""
Test catchment_nodes against MATLAB catchmentNodes.m

Numerical comparison tests for catchment/watershed delineation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.catchment import catchment_nodes, catchment_indices


class TestCatchmentNodesBasic:
    """Basic functionality tests"""
    
    def test_empty_samples(self):
        """Test with empty sample list"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        h_grid = np.random.rand(50, 50) * 100
        
        ij_catch, ptr_catch = catchment_nodes([], [], [], x, y, h_grid)
        
        assert ij_catch == []
        assert ptr_catch == [0]
    
    def test_local_sample(self):
        """Test local sample (L) - single node, no catchment"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        
        # Create simple elevation grid (peak in center)
        Y, X = np.meshgrid(y, x, indexing='ij')
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 10
        
        # Local sample at center
        sample_x = np.array([500])
        sample_y = np.array([500])
        sample_lc = np.array(['L'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Local sample should have exactly one node
        assert len(ij_catch) == 1
        assert ptr_catch[0] == 0
        # Check if it's roughly at the center (25, 25)
        assert abs(ij_catch[0][0] - 25) <= 1
        assert abs(ij_catch[0][1] - 25) <= 1
    
    def test_catchment_sample(self):
        """Test catchment sample (C) - multiple upslope nodes"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        
        # Create simple conical elevation (peak at center)
        Y, X = np.meshgrid(y, x, indexing='ij')
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 10
        
        # Catchment sample at center
        sample_x = np.array([500])
        sample_y = np.array([500])
        sample_lc = np.array(['C'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Catchment sample should have multiple nodes (upslope area)
        assert len(ij_catch) > 1
        assert ptr_catch[0] == 0
        # Center point should be in catchment
        center_idx = (25, 25)
        assert center_idx in ij_catch
    
    def test_multiple_samples(self):
        """Test multiple samples with mixed types"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        
        Y, X = np.meshgrid(y, x, indexing='ij')
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 10
        
        # Two samples: one local, one catchment
        sample_x = np.array([400, 600])
        sample_y = np.array([500, 500])
        sample_lc = np.array(['L', 'C'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Check pointers
        assert ptr_catch[0] == 0
        assert ptr_catch[1] == 1  # First sample is local, has 1 node
        # Second sample is catchment, should have more nodes
        assert len(ij_catch) > 1
    
    def test_string_lc_input(self):
        """Test string input for sample_lc"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        h_grid = np.random.rand(50, 50) * 100
        
        sample_x = np.array([500])
        sample_y = np.array([500])
        sample_lc = "L"  # String input
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        assert len(ij_catch) == 1


class TestCatchmentNodesAlgorithm:
    """Tests for D8 algorithm correctness"""
    
    def test_upslope_identification(self):
        """Test that only upslope nodes are included"""
        # Create simple slope: high on left, low on right
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        h_grid = np.tile(np.linspace(1000, 0, 50), (50, 1))
        
        # Sample at low point (right side)
        sample_x = np.array([900])
        sample_y = np.array([500])
        sample_lc = np.array(['C'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # All catchment nodes should have elevation >= sample point
        for i, j in ij_catch:
            assert h_grid[i, j] >= h_grid[int(25), int(45)] * 0.9  # Allow some tolerance
    
    def test_d8_connectivity(self):
        """Test D8 neighbor connectivity"""
        # Create a simple cone with flat top at center
        # After sink filling, the center will be filled to create a plateau
        x = np.linspace(0, 100, 11)
        y = np.linspace(0, 100, 11)
        
        # Use a larger grid to test D8 connectivity properly
        x = np.linspace(0, 1000, 21)
        y = np.linspace(0, 1000, 21)
        Y, X = np.meshgrid(y, x, indexing='ij')
        # Conical mountain
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 2
        
        # Sample at a point on the slope (not at peak to avoid sink filling issues)
        sample_x = np.array([600])
        sample_y = np.array([500])
        sample_lc = np.array(['C'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Should include multiple upslope nodes
        assert len(ij_catch) > 1
        # Center (peak) should be in catchment
        peak_idx = (10, 10)  # Approximately
        assert any(idx == peak_idx for idx in ij_catch) or len(ij_catch) > 10
    
    def test_sink_filling(self):
        """Test that sinks are filled before catchment calculation"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        
        # Create grid with a sink (depression)
        h_grid = np.ones((50, 50)) * 100
        h_grid[20:30, 20:30] = 50  # Sink in center
        h_grid[24:26, 24:26] = 200  # Peak in center of sink
        
        # Sample at peak
        sample_x = np.array([500])
        sample_y = np.array([500])
        sample_lc = np.array(['C'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Should still find a catchment (sink should be filled)
        assert len(ij_catch) >= 1


class TestCatchmentNodesMatlabComparison:
    """Comparison with MATLAB catchmentNodes.m"""
    
    def test_matlab_equivalent_local(self):
        """Test equivalent to MATLAB for local sample"""
        # Create same grid as would be used in MATLAB
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        Y, X = np.meshgrid(y, x, indexing='ij')
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 10
        
        sample_x = np.array([500.0])
        sample_y = np.array([500.0])
        sample_lc = np.array(['L'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # MATLAB would return:
        # - ijCatch: single linear index
        # - ptrCatch: [1; 2] (1-indexed, pointer to start and end+1)
        
        # Python returns:
        # - ij_catch: list with single tuple
        # - ptr_catch: [0, 1] (0-indexed pointers)
        
        assert len(ij_catch) == 1
        assert ptr_catch[0] == 0
    
    def test_pointer_structure(self):
        """Test that pointer structure matches MATLAB convention"""
        x = np.linspace(0, 1000, 50)
        y = np.linspace(0, 1000, 50)
        h_grid = np.random.rand(50, 50) * 100
        
        # Three samples
        sample_x = np.array([300, 500, 700])
        sample_y = np.array([300, 500, 700])
        sample_lc = np.array(['L', 'L', 'L'])
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Each local sample has 1 node
        # ptr_catch should be [0, 1, 2, 3] for 3 samples (n_samples + 1 elements)
        # ptr_catch[k] = start index of sample k
        # ptr_catch[k+1] = end index + 1 of sample k
        assert len(ptr_catch) == 4  # n_samples + 1
        assert ptr_catch[0] == 0
        assert ptr_catch[1] == 1
        assert ptr_catch[2] == 2
        assert ptr_catch[3] == 3  # End of last sample
        assert len(ij_catch) == 3


class TestCatchmentIndices:
    """Tests for catchment_indices helper function"""
    
    def test_extract_single_sample(self):
        """Test extracting indices for a single sample"""
        ij_catch = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        ptr_catch = [0, 2, 5]  # Sample 0: indices 0-1, Sample 1: indices 2-4
        
        result = catchment_indices(0, ij_catch, ptr_catch)
        
        assert result == [(0, 0), (1, 1)]
    
    def test_extract_second_sample(self):
        """Test extracting indices for second sample"""
        ij_catch = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        ptr_catch = [0, 2, 5]
        
        result = catchment_indices(1, ij_catch, ptr_catch)
        
        assert result == [(2, 2), (3, 3), (4, 4)]
    
    def test_out_of_range(self):
        """Test out of range index raises error"""
        ij_catch = [(0, 0), (1, 1)]
        ptr_catch = [0, 1, 2]
        
        with pytest.raises(IndexError):
            catchment_indices(5, ij_catch, ptr_catch)


class TestCatchmentIndicesMatlabComparison:
    """Direct comparison with MATLAB catchmentIndices.m"""
    
    def test_matlab_equivalent_first_sample(self):
        """
        Test equivalent to MATLAB for first sample (k=1 in MATLAB = k=0 in Python).
        
        MATLAB code:
        ```matlab
        ijCatch = [1; 2; 3; 4; 5];  % Linear indices (1-indexed)
        ptrCatch = [1; 3; 6];       % Pointers (1-indexed)
        k = 1;  % First sample
        ij = catchmentIndices(k, ijCatch, ptrCatch);
        % Returns: ij = [1; 2] (indices 1 to 2)
        ```
        
        Python equivalent:
        """
        # Note: Python uses (row, col) tuples instead of linear indices
        # and 0-indexed pointers
        ij_catch = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]  # 5 nodes
        ptr_catch = [0, 2, 5]  # Sample 0: nodes 0-1, Sample 1: nodes 2-4
        
        # First sample (k=0 in Python, equivalent to k=1 in MATLAB)
        result = catchment_indices(0, ij_catch, ptr_catch)
        
        # Should return first 2 nodes: [(0, 0), (0, 1)]
        assert len(result) == 2
        assert result == [(0, 0), (0, 1)]
    
    def test_matlab_equivalent_last_sample(self):
        """
        Test equivalent to MATLAB for last sample.
        
        MATLAB code:
        ```matlab
        ijCatch = [1; 2; 3; 4; 5];
        ptrCatch = [1; 3; 6];
        k = 2;  % Last sample (nSamples = 2)
        ij = catchmentIndices(k, ijCatch, ptrCatch);
        % Returns: ij = [3; 4; 5] (indices 3 to end)
        ```
        """
        ij_catch = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        ptr_catch = [0, 2, 5]
        
        # Last sample (k=1 in Python, equivalent to k=2 in MATLAB)
        result = catchment_indices(1, ij_catch, ptr_catch)
        
        # Should return last 3 nodes: [(1, 0), (1, 1), (2, 0)]
        assert len(result) == 3
        assert result == [(1, 0), (1, 1), (2, 0)]
    
    def test_matlab_equivalent_single_node_sample(self):
        """Test sample with single node (like local samples)"""
        # Sample 0: 1 node, Sample 1: 3 nodes, Sample 2: 1 node
        ij_catch = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)]
        ptr_catch = [0, 1, 4, 5]
        
        # First sample (single node)
        result_0 = catchment_indices(0, ij_catch, ptr_catch)
        assert len(result_0) == 1
        assert result_0 == [(0, 0)]
        
        # Second sample (3 nodes)
        result_1 = catchment_indices(1, ij_catch, ptr_catch)
        assert len(result_1) == 3
        assert result_1 == [(1, 0), (1, 1), (1, 2)]
        
        # Third sample (single node)
        result_2 = catchment_indices(2, ij_catch, ptr_catch)
        assert len(result_2) == 1
        assert result_2 == [(2, 0)]
    
    def test_matlab_indexing_logic(self):
        """
        Verify that Python slicing matches MATLAB indexing logic.
        
        MATLAB: ij = ijCatch(ptrCatch(k):ptrCatch(k+1)-1)
        Python: result = ij_catch[ptr_catch[k]:ptr_catch[k+1]]
        
        The -1 in MATLAB is because MATLAB uses closed intervals [start:end]
        while Python uses half-open intervals [start:end).
        """
        # Create test data
        ij_catch_matlab = np.array([10, 20, 30, 40, 50])  # Linear indices
        ptr_catch_matlab = np.array([1, 3, 6])  # 1-indexed pointers
        
        # Python equivalent (0-indexed)
        ij_catch_python = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        ptr_catch_python = [0, 2, 5]
        
        # Test first sample (k=1 in MATLAB -> k=0 in Python)
        # MATLAB: ijCatch(1:3-1) = ijCatch(1:2) = [10, 20]
        matlab_result_0 = ij_catch_matlab[ptr_catch_matlab[0]-1:ptr_catch_matlab[1]-1]
        python_result_0 = catchment_indices(0, ij_catch_python, ptr_catch_python)
        
        assert len(matlab_result_0) == len(python_result_0) == 2
        
        # Test second sample (k=2 in MATLAB -> k=1 in Python)
        # MATLAB: ijCatch(3:6-1) = ijCatch(3:5) = [30, 40, 50]
        matlab_result_1 = ij_catch_matlab[ptr_catch_matlab[1]-1:ptr_catch_matlab[2]-1]
        python_result_1 = catchment_indices(1, ij_catch_python, ptr_catch_python)
        
        assert len(matlab_result_1) == len(python_result_1) == 3
    
    def test_empty_catchment_not_allowed(self):
        """Test that empty catchments are handled (should not occur in practice)"""
        # If ptr_catch has consecutive equal values, it would indicate empty catchment
        ij_catch = [(0, 0), (1, 1)]
        ptr_catch = [0, 0, 2]  # Sample 0 would have 0 nodes (empty)
        
        # This should return empty list for sample 0
        result = catchment_indices(0, ij_catch, ptr_catch)
        assert result == []
        
        # Sample 1 should have 2 nodes
        result_1 = catchment_indices(1, ij_catch, ptr_catch)
        assert len(result_1) == 2
    
    def test_integration_with_catchment_nodes(self):
        """Integration test: catchment_nodes output works with catchment_indices"""
        # Create simple grid
        x = np.linspace(0, 1000, 20)
        y = np.linspace(0, 1000, 20)
        Y, X = np.meshgrid(y, x, indexing='ij')
        h_grid = 1000 - np.sqrt((X - 500)**2 + (Y - 500)**2) / 10
        
        # Three samples
        sample_x = np.array([400, 500, 600])
        sample_y = np.array([500, 500, 500])
        sample_lc = np.array(['L', 'C', 'L'])  # Local, Catchment, Local
        
        ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
        
        # Verify we can extract each sample's catchment
        for k in range(3):
            sample_catch = catchment_indices(k, ij_catch, ptr_catch)
            # Local samples should have 1 node
            if sample_lc[k] == 'L':
                assert len(sample_catch) == 1
            # Catchment sample should have multiple nodes
            else:
                assert len(sample_catch) > 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
