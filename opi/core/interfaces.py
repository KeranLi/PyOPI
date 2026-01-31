"""
Interface definitions (protocols) for OPI components.

These define the contracts that implementations must satisfy.
"""

from typing import Protocol, runtime_checkable, Any
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class GridCalculator(Protocol):
    """Protocol for grid-based calculations."""
    
    def calculate(self, grid_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform calculation on grid data.
        
        Args:
            grid_data: Input grid array
            
        Returns:
            Calculated grid array
        """
        ...


@runtime_checkable
class IsotopeCalculator(Protocol):
    """Protocol for isotope calculations."""
    
    def calculate_fractionation(self, temperature: float) -> float:
        """
        Calculate fractionation factor at given temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Fractionation factor
        """
        ...
    
    def calculate_distillation(self, initial_value: float, 
                               remaining_fraction: float) -> float:
        """
        Calculate isotope value after Rayleigh distillation.
        
        Args:
            initial_value: Initial isotope ratio
            remaining_fraction: Fraction of vapor remaining
            
        Returns:
            Isotope ratio after distillation
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for numerical solvers."""
    
    def solve(self, *args, **kwargs) -> Any:
        """
        Execute solver.
        
        Returns:
            Solver-specific result object
        """
        ...
    
    @property
    def converged(self) -> bool:
        """Whether the solver converged."""
        ...


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for data loaders."""
    
    def load(self, path: Path) -> Any:
        """
        Load data from file.
        
        Args:
            path: Path to data file
            
        Returns:
            Loaded data object
        """
        ...
    
    def supports(self, path: Path) -> bool:
        """
        Check if this loader supports the given file.
        
        Args:
            path: Path to check
            
        Returns:
            True if supported, False otherwise
        """
        ...


@runtime_checkable
class ResultExporter(Protocol):
    """Protocol for result exporters."""
    
    def export(self, result: Any, path: Path) -> None:
        """
        Export result to file.
        
        Args:
            result: Result object to export
            path: Output file path
        """
        ...
    
    @property
    def file_extension(self) -> str:
        """File extension for this format (e.g., '.nc', '.npz')."""
        ...
