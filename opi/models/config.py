"""
Configuration classes for OPI.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class SolverConfig:
    """Configuration for physical solvers."""
    
    # FFT solver settings
    padding_factor: float = 2.0  # Zero padding factor
    wavenumber_tolerance: float = 1e-10
    handle_singularities: bool = True
    
    # Numerical stability
    min_buoyancy_frequency: float = 0.001  # rad/s
    max_vertical_wavenumber: float = 1e6   # rad/m
    
    # Isotherm calculation
    isotherm_temperatures: List[float] = field(default_factory=lambda: [223.0, 258.0])


@dataclass
class OptimizerConfig:
    """Configuration for parameter optimization."""
    
    # CRS3 parameters
    mu: int = 25  # Population size factor
    epsilon: float = 1e-6  # Convergence tolerance
    max_iterations: int = 10000
    
    # Parallel settings
    parallel: bool = False
    num_workers: int = 1
    
    # Random seed (None for random)
    random_seed: Optional[int] = None
    
    # Progress reporting
    verbose: bool = True
    print_interval: int = 100


@dataclass
class IOConfig:
    """Configuration for input/output."""
    
    # Input settings
    default_data_path: Path = Path("data")
    topography_filename: str = "topography.mat"
    sample_filename: Optional[str] = None
    
    # Output settings
    output_path: Path = Path("output")
    save_grids: bool = True
    save_format: str = "npz"  # "npz", "nc", "mat"
    
    # Grid preprocessing
    apply_tukey_window: bool = False
    tukey_ratio: float = 0.25


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: Path = Path("opi.log")
    log_to_console: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching."""
    
    enabled: bool = True
    cache_dir: Path = Path(".opi_cache")
    max_size_mb: int = 1000
    ttl_hours: int = 24  # Time to live


@dataclass
class ParallelConfig:
    """Configuration for parallel computing."""
    
    enabled: bool = False
    num_workers: int = -1  # -1 for auto (all cores)
    backend: str = "multiprocessing"  # "multiprocessing", "threading", "loky"


@dataclass
class OPIConfig:
    """
    Main configuration class for OPI.
    
    This is the central configuration object passed to calculators.
    
    Example:
        config = OPIConfig(
            solver=SolverConfig(padding_factor=2.0),
            optimizer=OptimizerConfig(max_iterations=5000),
            parallel=ParallelConfig(enabled=True, num_workers=4)
        )
        calculator = OneWindCalculator(config)
    """
    
    # Sub-configurations
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    io: IOConfig = field(default_factory=IOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # Global settings
    version: str = "1.0.0"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OPIConfig":
        """Create configuration from dictionary."""
        # Extract sub-configs
        solver_dict = config_dict.get('solver', {})
        optimizer_dict = config_dict.get('optimizer', {})
        io_dict = config_dict.get('io', {})
        logging_dict = config_dict.get('logging', {})
        cache_dict = config_dict.get('cache', {})
        parallel_dict = config_dict.get('parallel', {})
        
        return cls(
            solver=SolverConfig(**solver_dict),
            optimizer=OptimizerConfig(**optimizer_dict),
            io=IOConfig(**io_dict),
            logging=LoggingConfig(**logging_dict),
            cache=CacheConfig(**cache_dict),
            parallel=ParallelConfig(**parallel_dict),
            version=config_dict.get('version', "1.0.0")
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "OPIConfig":
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: Path) -> "OPIConfig":
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, path: Path):
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: Path):
        """Save configuration to JSON file."""
        import json
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Validate solver config
        if self.solver.padding_factor < 1.0:
            issues.append("Solver padding_factor must be >= 1.0")
        
        # Validate optimizer config
        if self.optimizer.mu < 1:
            issues.append("Optimizer mu must be >= 1")
        if self.optimizer.epsilon <= 0:
            issues.append("Optimizer epsilon must be > 0")
        
        # Validate IO config
        if self.io.save_format not in ["npz", "nc", "mat"]:
            issues.append(f"Unknown save format: {self.io.save_format}")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# Default configurations
DEFAULT_CONFIG = OPIConfig()

FAST_CONFIG = OPIConfig(
    solver=SolverConfig(padding_factor=1.5),
    optimizer=OptimizerConfig(max_iterations=1000, mu=10),
    cache=CacheConfig(enabled=False)
)

PRECISE_CONFIG = OPIConfig(
    solver=SolverConfig(padding_factor=3.0),
    optimizer=OptimizerConfig(max_iterations=50000, epsilon=1e-8),
    cache=CacheConfig(enabled=True)
)

PARALLEL_CONFIG = OPIConfig(
    parallel=ParallelConfig(enabled=True, num_workers=-1),
    optimizer=OptimizerConfig(parallel=True)
)
