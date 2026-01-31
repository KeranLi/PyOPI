from setuptools import setup, find_packages

setup(
    name='opi-orographic-precipitation-isotopes',
    version='1.0.0',
    description='Python implementation of Orographic Precipitation and Isotopes models',
    author='Mark Brandon (adapted to Python)',
    author_email='mark.brandon@yale.edu',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
        'xarray>=0.19.0',
        'netcdf4>=1.5.0',
        'h5py>=3.0.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)