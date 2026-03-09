"""
New-style One Wind Calculator demonstrating the refactored architecture.

This is a proof-of-concept showing how the OPI calculation would look
with the new object-oriented, type-safe design.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from ..core.calculator import OPICalculator
from ..core.exceptions import ValidationError
from ..models.config import OPIConfig
from ..models.parameters import OneWindParameters
from ..models.results import OneWindResults, PrecipitationGrid, IsotopeGrid
from ..models.domain import Grid
from ..solvers.fourier import FourierSolver
from ..solvers.precipitation import PrecipitationCalculator
from ..solvers.isotope import IsotopeCalculator


class OneWindCalculatorNew(OPICalculator[OneWindParameters, OneWindResults]):
    """
    Single wind field calculator (refactored version).
    
    This demonstrates the new architecture with:
    - Type-safe parameters and results
    - Composable solvers
    - Clear separation of concerns
    
    Example:
        >>> from opi.models import OPIConfig, OneWindParameters, WindField
        >>> config = OPIConfig()
        >>> calculator = OneWindCalculatorNew(config)
        >>> 
        >>> params = OneWindParameters(
        ...     wind=WindField(speed=10.0, azimuth=90.0),
        ...     atmosphere=AtmosphereState(...),
        ...     ...
        ... )
        >>> 
        >>> results = calculator.calculate(params)
        >>> print(f"Mean precipitation: {results.precipitation.mean}")
    """
    
    def __init__(self, config: Optional[OPIConfig] = None):
        """
        Initialize calculator.
        
        Args:
            config: OPI configuration
        """
        super().__init__(config)
        
        # Initialize component solvers
        self._fourier_solver = FourierSolver(config.solver if config else None)
        self._precip_calc = PrecipitationCalculator()
        self._isotope_calc = IsotopeCalculator()
        
        self.log_info("OneWindCalculatorNew initialized")
    
    def _validate_input(self, params: OneWindParameters):
        """
        Validate input parameters.
        
        Args:
            params: Input parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Check wind speed
        if params.wind.speed <= 0:
            raise ValidationError(
                "Wind speed must be positive",
                details={'speed': params.wind.speed}
            )
        
        # Check temperature
        if params.atmosphere.sea_level_temperature < 200:  # K
            raise ValidationError(
                "Temperature seems unreasonably low",
                details={'temperature': params.atmosphere.sea_level_temperature}
            )
        
        # Check topography is set
        if self.topography is None:
            raise ValidationError(
                "Topography must be set before calculation",
                details={'topography': None}
            )
        
        self.log_debug("Input validation passed")
    
    def _preprocess(self, params: OneWindParameters):
        """
        Pre-processing: prepare atmosphere state.
        
        Args:
            params: Input parameters
        """
        # Update buoyancy frequency based on actual topography
        h_max = self.topography.max_elevation
        nm = params.wind.speed * 0.25 / h_max  # M = 0.25 typical
        
        # Create updated atmosphere state (would use dataclasses.replace in real impl)
        self.log_debug(f"Using buoyancy frequency: {nm:.6f} rad/s")
    
    def _calculate(self, params: OneWindParameters) -> OneWindResults:
        """
        Core calculation.
        
        Args:
            params: Input parameters
            
        Returns:
            Calculation results
        """
        self.log_info("Starting Fourier solution...")
        fourier_sol = self._fourier_solver.solve(
            self.topography,
            params.wind,
            params.atmosphere
        )
        
        self.log_info("Calculating precipitation...")
        precip = self._precip_calc.calculate(
            fourier_sol,
            params.atmosphere,
            params.microphysics
        )
        
        self.log_info("Calculating isotopes...")
        d2h_grid = self._isotope_calc.calculate(
            precip,
            params.isotopes,
            params.atmosphere
        )
        
        # Create results
        results = OneWindResults(
            precipitation=precip,
            d2h=d2h_grid,
            d18o=d2h_grid,  # Would calculate properly
            fourier_solution=fourier_sol
        )
        
        self.log_info("Calculation complete")
        return results
    
    def _postprocess(self, results: OneWindResults) -> OneWindResults:
        """
        Post-processing: add summary statistics.
        
        Args:
            results: Raw results
            
        Returns:
            Processed results
        """
        # Add any final processing
        self.log_debug(f"Final precipitation range: {results.precipitation.min:.6f} to {results.precipitation.max:.6f}")
        return results


# Example usage function
def example_usage():
    """Demonstrate usage of the new calculator."""
    print("=" * 60)
    print("OneWindCalculatorNew Example Usage")
    print("=" * 60)
    
    from ..models.domain import Topography, Grid
    from ..models.parameters import (
        WindField, AtmosphereState, IsotopeParameters, MicrophysicsParameters
    )
    
    # 1. Create configuration
    config = OPIConfig()
    print("\n1. Configuration created")
    
    # 2. Create calculator
    calculator = OneWindCalculatorNew(config)
    print("2. Calculator created")
    
    # 3. Create topography
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    elevation = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    grid = Grid(x=x, y=y)
    topography = Topography(elevation=elevation, grid=grid)
    calculator.topography = topography
    print("3. Topography set")
    
    # 4. Create parameters
    params = OneWindParameters(
        wind=WindField(speed=10.0, azimuth=90.0),
        atmosphere=AtmosphereState(
            sea_level_temperature=290.0,
            buoyancy_frequency=0.01
        ),
        isotopes=IsotopeParameters(
            base_d2h=-5e-3,
            base_d18o=-0.5e-3
        ),
        microphysics=MicrophysicsParameters(
            eddy_diffusion=0.0,
            condensation_time=1000.0,
            residual_precipitation=1.0
        )
    )
    print("4. Parameters created")
    
    # 5. Run calculation
    print("\n5. Running calculation...")
    try:
        results = calculator.calculate(params)
        print("   Calculation successful!")
        print(f"   Mean precipitation: {results.precipitation.mean:.6f}")
        print(f"   Mean d2H: {results.d2h.mean_per_mil:.2f} permil")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()
