"""
Physical constants and conversion factors for the OPI package
"""

# Physical constants
G = 9.81              # Standard gravity (m/s^2)
CPD = 1005.7          # Heat capacity dry air, constant pressure & 0 C (J/kg-K)
CPV = 1952            # Heat capacity water vapor, constant pressure & 0 C (J/kg-K)
RD = 287.0            # Specific gas constant dry air (J/kg-K)
L = 2.501e6           # Latent heat of vaporization (water -> vapor) (J/kg)
P0 = 101325           # Standard sea-level pressure (Pa)
EPSILON = 0.622       # Molecular mass of water relative to air (kg/kg)

# Earth characteristics
RADIUS_EARTH = 6371e3      # Mean radius of the Earth (m)
M_PER_DEGREE = 3.14159 * RADIUS_EARTH / 180  # Meters per arc degree for the Earth's surface

# Conversion factors
TC2K = 273.15         # Convert from Celsius to Kelvin

# Default parameters
HR = 540              # Average distance (m) for isotopic exchange (Friedman et al., 1962)
SD_RES_RATIO = 28.3   # Estimated standard-deviation ratio for isotopic variation