"""
Setup script for OPI package
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version
version = "2.0.0"

# Core dependencies
install_requires = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
]

# Optional dependencies
extras_require = {
    'io': [
        'xarray>=0.19.0',
        'netCDF4>=1.5.0',
        'rasterio>=1.2.0',
        'pandas>=1.3.0',
    ],
    'viz': [
        'cartopy>=0.20.0',
    ],
    'parallel': [
        'joblib>=1.0.0',
        'tqdm>=4.60.0',
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'jupyter>=1.0.0',
    ],
}

# All extras combined
extras_require['all'] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="opi-orographic-precipitation-isotopes",
    version=version,
    author="Mark Brandon, Yale University (MATLAB); AI Assistant (Python)",
    author_email="",
    description="Orographic Precipitation and Isotopes Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/.../opi-python",
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'opi-run=opi.__main__:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
