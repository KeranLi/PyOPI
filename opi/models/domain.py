"""
Domain models for grid and coordinate representations.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterator
import numpy.typing as npt


@dataclass(frozen=True)
class Grid:
    """
    Rectangular grid definition.
    
    Attributes:
        x: Grid coordinates in x direction (meters)
        y: Grid coordinates in y direction (meters)
        nx: Number of grid points in x
        ny: Number of grid points in y
        dx: Grid spacing in x
        dy: Grid spacing in y
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    
    def __post_init__(self):
        # Validate inputs
        if len(self.x.shape) != 1:
            raise ValueError(f"x must be 1D, got shape {self.x.shape}")
        if len(self.y.shape) != 1:
            raise ValueError(f"y must be 1D, got shape {self.y.shape}")
    
    @property
    def nx(self) -> int:
        return len(self.x)
    
    @property
    def ny(self) -> int:
        return len(self.y)
    
    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0]) if self.nx > 1 else 0.0
    
    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0]) if self.ny > 1 else 0.0
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)
    
    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Returns (x_min, x_max, y_min, y_max)"""
        return (float(self.x[0]), float(self.x[-1]), 
                float(self.y[0]), float(self.y[-1]))
    
    def meshgrid(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create 2D meshgrid coordinates."""
        return np.meshgrid(self.x, self.y)
    
    def create_array(self, fill_value: float = 0.0) -> NDArray[np.float64]:
        """Create an empty array with grid shape."""
        return np.full(self.shape, fill_value, dtype=np.float64)
    
    def __iter__(self) -> Iterator[Tuple[int, int, float, float]]:
        """Iterate over grid points: (i, j, x, y)"""
        for i, y in enumerate(self.y):
            for j, x in enumerate(self.x):
                yield (i, j, x, y)


@dataclass
class Topography:
    """
    Topography (elevation) data on a grid.
    
    Attributes:
        elevation: 2D array of elevation values (meters)
        grid: Grid definition
        max_elevation: Maximum elevation
        min_elevation: Minimum elevation
    """
    elevation: NDArray[np.float64]
    grid: Grid
    
    def __post_init__(self):
        if self.elevation.shape != self.grid.shape:
            raise ValueError(
                f"Elevation shape {self.elevation.shape} "
                f"does not match grid shape {self.grid.shape}"
            )
    
    @property
    def max_elevation(self) -> float:
        return float(np.max(self.elevation))
    
    @property
    def min_elevation(self) -> float:
        return float(np.min(self.elevation))
    
    @property
    def mean_elevation(self) -> float:
        return float(np.mean(self.elevation))
    
    def get_gradient(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate topographic gradient (dz/dx, dz/dy)."""
        dz_dy, dz_dx = np.gradient(self.elevation, self.grid.dy, self.grid.dx)
        return dz_dx, dz_dy
    
    def interpolate(self, x: float, y: float) -> float:
        """Bilinear interpolation at point (x, y)."""
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (self.grid.y, self.grid.x),
            self.elevation,
            bounds_error=False,
            fill_value=0.0
        )
        return float(interp([[y, x]])[0])


@dataclass(frozen=True)
class CoordinateSystem:
    """
    Coordinate system transformation between geographic and projected coordinates.
    
    Attributes:
        lon_origin: Origin longitude (degrees)
        lat_origin: Origin latitude (degrees)
        meters_per_degree: Conversion factor
    """
    lon_origin: float = 0.0
    lat_origin: float = 0.0
    meters_per_degree: float = 111320.0  # At equator
    
    def to_projected(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert geographic to projected coordinates."""
        x = (lon - self.lon_origin) * self.meters_per_degree * np.cos(np.radians(lat))
        y = (lat - self.lat_origin) * self.meters_per_degree
        return (x, y)
    
    def to_geographic(self, x: float, y: float) -> Tuple[float, float]:
        """Convert projected to geographic coordinates."""
        lat = y / self.meters_per_degree + self.lat_origin
        lon = x / (self.meters_per_degree * np.cos(np.radians(lat))) + self.lon_origin
        return (lon, lat)
    
    def calculate_coriolis(self, lat: Optional[float] = None) -> float:
        """Calculate Coriolis parameter at given latitude."""
        omega = 7.2921e-5  # Earth's rotation rate (rad/s)
        if lat is None:
            lat = self.lat_origin
        return 2 * omega * np.sin(np.radians(lat))
