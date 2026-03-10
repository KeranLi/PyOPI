"""
Result data classes for OPI calculations.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from .domain import Grid


@dataclass
class CalculationMetadata:
    """Metadata about a calculation run."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    version: str = "1.0.0"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def finalize(self):
        """Mark calculation as complete."""
        self.end_time = datetime.now()


@dataclass
class BaseGridResult:
    """Base class for gridded results."""
    data: NDArray[np.float64]
    grid: Grid
    
    def __post_init__(self):
        if self.data.shape != self.grid.shape:
            raise ValueError(
                f"Data shape {self.data.shape} doesn't match grid {self.grid.shape}"
            )
    
    @property
    def min(self) -> float:
        return float(np.min(self.data))
    
    @property
    def max(self) -> float:
        return float(np.max(self.data))
    
    @property
    def mean(self) -> float:
        return float(np.mean(self.data))
    
    @property
    def std(self) -> float:
        return float(np.std(self.data))
    
    def get_values_at_points(self, x: NDArray[np.float64], 
                             y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate to given points."""
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (self.grid.y, self.grid.x),
            self.data,
            bounds_error=False,
            fill_value=np.nan
        )
        return interp(np.column_stack([y, x]))


@dataclass
class PrecipitationGrid(BaseGridResult):
    """
    Precipitation rate grid.
    
    Units: kg/m^2/s (can be converted to mm/day)
    """
    
    @property
    def total_precipitation(self) -> float:
        """Total precipitation over domain (kg/s)."""
        return float(np.sum(self.data) * self.grid.dx * self.grid.dy)
    
    def to_mm_per_day(self) -> NDArray[np.float64]:
        """Convert to mm/day."""
        # 1 kg/m^2/s = 86400 mm/day (assuming water density 1000 kg/m^3)
        return self.data * 86400
    
    @property
    def max_precipitation_rate(self) -> float:
        """Maximum precipitation rate (kg/m^2/s)."""
        return self.max


@dataclass
class IsotopeGrid(BaseGridResult):
    """
    Isotope composition grid.
    
    Values are in fraction (e.g., -0.050 for -50 permil).
    Use per_mil property for permil values.
    """
    
    @property
    def per_mil(self) -> NDArray[np.float64]:
        """Values in permil."""
        return self.data * 1000
    
    @property
    def min_per_mil(self) -> float:
        return self.min * 1000
    
    @property
    def max_per_mil(self) -> float:
        return self.max * 1000
    
    @property
    def mean_per_mil(self) -> float:
        return self.mean * 1000
    
    def calculate_d_excess(self, d18o_grid: 'IsotopeGrid') -> NDArray[np.float64]:
        """Calculate deuterium excess using another grid for d18O."""
        return (self.data - 8 * d18o_grid.data) * 1000


@dataclass
class FourierSolution:
    """
    Results from Fourier solution of linearized Euler equations.
    
    This includes the transformed topography and vertical wavenumbers
    needed for precipitation calculations.
    """
    h_hat: NDArray[np.complex128]  # Fourier coefficients
    k_z: NDArray[np.complex128]    # Vertical wavenumbers
    k_s: NDArray[np.float64]       # Horizontal wavenumbers (s direction)
    k_t: NDArray[np.float64]       # Horizontal wavenumbers (t direction)
    wind_coords: Tuple[NDArray[np.float64], NDArray[np.float64]]  # (s, t) coordinates
    
    def __post_init__(self):
        # Ensure consistent shapes
        if self.h_hat.shape != self.k_z.shape:
            raise ValueError("h_hat and k_z must have same shape")


@dataclass
class AtmosphericProfile:
    """
    Vertical atmospheric profile at a point or averaged.
    
    Attributes:
        heights: Height levels (m)
        temperature: Temperature profile (K)
        pressure: Pressure profile (Pa)
        density: Air density profile (kg/m^3)
        vapor_density: Water vapor density profile (kg/m^3)
    """
    heights: NDArray[np.float64]
    temperature: NDArray[np.float64]
    pressure: Optional[NDArray[np.float64]] = None
    density: Optional[NDArray[np.float64]] = None
    vapor_density: Optional[NDArray[np.float64]] = None
    
    @property
    def environmental_lapse_rate(self) -> NDArray[np.float64]:
        """Environmental lapse rate (K/m)."""
        return -np.gradient(self.temperature, self.heights)
    
    def get_lapse_rate_at(self, height: float) -> float:
        """Get lapse rate at specific height."""
        from scipy.interpolate import interp1d
        lapse = self.environmental_lapse_rate
        f = interp1d(self.heights, lapse, bounds_error=False, fill_value='extrapolate')
        return float(f(height))


@dataclass
class FitStatistics:
    """Statistics from parameter fitting."""
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    r_squared: Optional[float] = None
    rmse: Optional[float] = None
    
    @property
    def goodness_of_fit(self) -> str:
        """Qualitative assessment of fit."""
        if self.reduced_chi_squared < 1.5:
            return "excellent"
        elif self.reduced_chi_squared < 3:
            return "good"
        elif self.reduced_chi_squared < 5:
            return "acceptable"
        else:
            return "poor"


@dataclass
class OneWindResults:
    """
    Complete results from single wind field calculation.
    
    This is the main output from OneWindCalculator.
    """
    # Gridded outputs
    precipitation: PrecipitationGrid
    d2h: IsotopeGrid
    d18o: IsotopeGrid
    moisture_ratio: Optional[BaseGridResult] = None
    relative_humidity: Optional[BaseGridResult] = None
    
    # Derived quantities
    atmospheric_profile: Optional[AtmosphericProfile] = None
    fourier_solution: Optional[FourierSolution] = None
    
    # Scalar quantities
    tau_f: Optional[float] = None  # Fall time
    h_s: Optional[float] = None    # Water vapor scale height
    
    # Metadata
    metadata: CalculationMetadata = field(default_factory=CalculationMetadata)
    
    def finalize(self):
        """Mark calculation as complete."""
        self.metadata.finalize()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'precipitation_mean': self.precipitation.mean,
            'precipitation_max': self.precipitation.max,
            'd2h_mean': self.d2h.mean_per_mil,
            'd2h_range': (self.d2h.min_per_mil, self.d2h.max_per_mil),
            'calculation_time': self.metadata.duration_seconds
        }


@dataclass
class TwoWindResults:
    """
    Results from two wind fields calculation.
    
    Includes both individual wind field results and combined results.
    """
    # Individual wind field results
    wind1_results: OneWindResults
    wind2_results: OneWindResults
    
    # Combined results (weighted by fraction)
    combined_precipitation: PrecipitationGrid
    combined_d2h: IsotopeGrid
    combined_d18o: IsotopeGrid
    
    # Mixing fraction
    fraction_wind2: float
    
    # Metadata
    metadata: CalculationMetadata = field(default_factory=CalculationMetadata)
    
    def finalize(self):
        self.metadata.finalize()
    
    @property
    def fraction_wind1(self) -> float:
        return 1.0 - self.fraction_wind2
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'wind1': self.wind1_results.get_summary(),
            'wind2': self.wind2_results.get_summary(),
            'combined_precip_mean': self.combined_precipitation.mean,
            'fraction_wind2': self.fraction_wind2,
            'calculation_time': self.metadata.duration_seconds
        }


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    success: bool
    optimal_parameters: NDArray[np.float64]
    final_misfit: float
    iterations: int
    function_evaluations: int
    message: str
    
    # Optional detailed results
    parameter_history: Optional[NDArray[np.float64]] = None
    misfit_history: Optional[NDArray[np.float64]] = None
    
    @property
    def converged(self) -> bool:
        return self.success
