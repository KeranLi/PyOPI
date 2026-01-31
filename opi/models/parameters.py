"""
Parameter data classes for OPI calculations.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass(frozen=True)
class WindField:
    """
    Wind field parameters.
    
    Attributes:
        speed: Wind speed (m/s)
        azimuth: Wind direction in degrees from North (0-360)
    """
    speed: float
    azimuth: float
    
    def __post_init__(self):
        if self.speed < 0:
            raise ValueError(f"Wind speed must be non-negative, got {self.speed}")
        if not 0 <= self.azimuth < 360:
            raise ValueError(f"Azimuth must be in [0, 360), got {self.azimuth}")
    
    @property
    def direction_vector(self) -> Tuple[float, float]:
        """Returns (u, v) wind components (east, north)."""
        azimuth_rad = np.radians(self.azimuth)
        u = self.speed * np.sin(azimuth_rad)
        v = self.speed * np.cos(azimuth_rad)
        return (u, v)
    
    @property
    def direction_radians(self) -> float:
        """Azimuth in radians."""
        return np.radians(self.azimuth)


@dataclass(frozen=True)
class AtmosphereState:
    """
    Atmospheric base state parameters.
    
    Attributes:
        sea_level_temperature: Temperature at sea level (K)
        buoyancy_frequency: Brunt-Väisälä frequency N_m (rad/s)
        density_scale_height: Scale height for density (m)
    """
    sea_level_temperature: float
    buoyancy_frequency: float
    density_scale_height: float = 8000.0
    
    def __post_init__(self):
        if self.sea_level_temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.sea_level_temperature}")
        if self.buoyancy_frequency <= 0:
            raise ValueError(f"Buoyancy frequency must be positive, got {self.buoyancy_frequency}")
    
    @property
    def sea_level_temperature_celsius(self) -> float:
        return self.sea_level_temperature - 273.15
    
    def get_temperature_at_height(self, height: float) -> float:
        """Get temperature at given height using environmental lapse rate."""
        # Approximate environmental lapse rate
        gamma_env = 0.0065  # K/m
        return self.sea_level_temperature - gamma_env * height


@dataclass(frozen=True)
class IsotopeParameters:
    """
    Isotope base parameters.
    
    Attributes:
        base_d2h: Base d2H value (fraction, not permil)
        base_d18o: Base d18O value (fraction, not permil)
        latitude_gradient_d2h: d2H gradient per degree latitude
        latitude_gradient_d18o: d18O gradient per degree latitude
    """
    base_d2h: float
    base_d18o: float = -0.5e-3  # Default approx value
    latitude_gradient_d2h: float = 0.0
    latitude_gradient_d18o: float = 0.0
    
    @property
    def base_d2h_per_mil(self) -> float:
        return self.base_d2h * 1000
    
    @property
    def base_d18o_per_mil(self) -> float:
        return self.base_d18o * 1000
    
    def get_value_at_latitude(self, lat: float, reference_lat: float = 45.0) -> Tuple[float, float]:
        """Get isotope values at given latitude."""
        d2h = self.base_d2h + self.latitude_gradient_d2h * (lat - reference_lat)
        d18o = self.base_d18o + self.latitude_gradient_d18o * (lat - reference_lat)
        return (d2h, d18o)


@dataclass(frozen=True)
class MicrophysicsParameters:
    """
    Microphysical parameters.
    
    Attributes:
        eddy_diffusion: Eddy diffusion coefficient (m^2/s)
        condensation_time: Condensation time scale (s)
        residual_precipitation: Fraction of precipitation remaining after evaporation
    """
    eddy_diffusion: float
    condensation_time: float
    residual_precipitation: float = 1.0
    
    def __post_init__(self):
        if self.eddy_diffusion < 0:
            raise ValueError(f"Eddy diffusion must be non-negative")
        if self.condensation_time <= 0:
            raise ValueError(f"Condensation time must be positive")
        if not 0 <= self.residual_precipitation <= 1:
            raise ValueError(f"Residual precipitation must be in [0, 1]")


@dataclass(frozen=True)
class OneWindParameters:
    """
    Complete parameters for single wind field calculation.
    
    This is the main input for OneWindCalculator.
    
    Attributes:
        wind: Wind field parameters
        atmosphere: Atmospheric state
        isotopes: Isotope parameters
        microphysics: Microphysical parameters
    """
    wind: WindField
    atmosphere: AtmosphereState
    isotopes: IsotopeParameters
    microphysics: MicrophysicsParameters
    
    @classmethod
    def from_array(cls, beta: NDArray[np.float64]) -> "OneWindParameters":
        """Create from 9-element parameter array.
        
        Array order: [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_dlat, f_p0]
        """
        if len(beta) != 9:
            raise ValueError(f"Expected 9 parameters, got {len(beta)}")
        
        wind = WindField(speed=beta[0], azimuth=beta[1])
        
        # Calculate buoyancy frequency from mountain-height number
        # M = N_m * h_max / U  =>  N_m = M * U / h_max
        # We use a typical h_max here, will be updated with actual topography
        h_max_typical = 2000.0
        nm = beta[3] * beta[0] / h_max_typical
        
        atmosphere = AtmosphereState(
            sea_level_temperature=beta[2],
            buoyancy_frequency=nm
        )
        
        isotopes = IsotopeParameters(
            base_d2h=beta[6],
            latitude_gradient_d2h=beta[7]
        )
        
        microphysics = MicrophysicsParameters(
            eddy_diffusion=beta[4],
            condensation_time=beta[5],
            residual_precipitation=beta[8]
        )
        
        return cls(wind, atmosphere, isotopes, microphysics)
    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to 9-element parameter array."""
        return np.array([
            self.wind.speed,
            self.wind.azimuth,
            self.atmosphere.sea_level_temperature,
            # M = N_m * h_max / U, but we don't have h_max here
            # This is approximate
            self.atmosphere.buoyancy_frequency * 2000.0 / self.wind.speed,
            self.microphysics.eddy_diffusion,
            self.microphysics.condensation_time,
            self.isotopes.base_d2h,
            self.isotopes.latitude_gradient_d2h,
            self.microphysics.residual_precipitation
        ])


@dataclass(frozen=True)
class TwoWindParameters:
    """
    Parameters for two wind fields calculation.
    
    Attributes:
        wind1: First wind field
        wind2: Second wind field
        fraction_wind2: Fraction of second wind field (0-1)
    """
    wind1: OneWindParameters
    wind2: OneWindParameters
    fraction_wind2: float
    
    def __post_init__(self):
        if not 0 <= self.fraction_wind2 <= 1:
            raise ValueError(f"Fraction must be in [0, 1], got {self.fraction_wind2}")
    
    @classmethod
    def from_array(cls, beta: NDArray[np.float64]) -> "TwoWindParameters":
        """Create from 19-element parameter array.
        
        Array order: [9 params for wind1, 9 params for wind2, fraction]
        """
        if len(beta) != 19:
            raise ValueError(f"Expected 19 parameters, got {len(beta)}")
        
        wind1 = OneWindParameters.from_array(beta[0:9])
        wind2 = OneWindParameters.from_array(beta[9:18])
        fraction = beta[18]
        
        return cls(wind1, wind2, fraction)
    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to 19-element parameter array."""
        return np.concatenate([
            self.wind1.to_array(),
            self.wind2.to_array(),
            [self.fraction_wind2]
        ])


@dataclass
class Sample:
    """
    Individual sample with isotope measurements.
    
    Attributes:
        x: X coordinate (m)
        y: Y coordinate (m)
        d2h: Measured d2H value (fraction)
        d18o: Measured d18O value (fraction)
        sample_type: 'L' for local, 'C' for catchment
        elevation: Elevation at sample location (m)
    """
    x: float
    y: float
    d2h: float
    d18o: float
    sample_type: str = 'C'
    elevation: Optional[float] = None
    
    @property
    def d_excess(self) -> float:
        """Deuterium excess in permil."""
        return (self.d2h - 8 * self.d18o) * 1000
    
    @property
    def is_local(self) -> bool:
        return self.sample_type.upper() == 'L'
    
    @property
    def is_catchment(self) -> bool:
        return self.sample_type.upper() == 'C'


@dataclass
class SampleCollection:
    """Collection of samples."""
    samples: List[Sample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def add(self, sample: Sample):
        self.samples.append(sample)
    
    @property
    def d2h_array(self) -> NDArray[np.float64]:
        return np.array([s.d2h for s in self.samples])
    
    @property
    def d18o_array(self) -> NDArray[np.float64]:
        return np.array([s.d18o for s in self.samples])
    
    @property
    def positions(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns (x_array, y_array)."""
        x = np.array([s.x for s in self.samples])
        y = np.array([s.y for s in self.samples])
        return (x, y)
