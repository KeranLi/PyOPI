Installation
============

This guide will help you install OPI and its dependencies.

Requirements
------------

OPI requires the following Python packages:

* Python >= 3.8
* NumPy >= 1.21.0
* SciPy >= 1.7.0
* Matplotlib >= 3.4.0
* Pandas >= 1.3.0
* Xarray >= 0.19.0
* NetCDF4 >= 1.5.0
* h5py >= 3.0.0

Optional dependencies:

* Jupyter (for running notebooks)
* pytest (for running tests)

Installing from Source
----------------------

The recommended way to install OPI is from the source code:

1. Clone or download the repository::

       git clone https://github.com/yourusername/OPI.git
       cd OPI/OPI_python

2. Install the required dependencies::

       pip install -r requirements.txt

3. Add the ``OPI_python`` directory to your Python path, or install in 
   development mode::

       pip install -e .

Or set the Python path in your scripts::

    import sys
    sys.path.insert(0, '/path/to/OPI/OPI_python')

Installing Dependencies Only
------------------------------

If you prefer to manage the package manually, install just the dependencies::

    pip install numpy scipy matplotlib pandas xarray netCDF4 h5py

Verifying the Installation
--------------------------

To verify that OPI is installed correctly, open a Python interpreter and run::

    import opi
    print(opi.__version__)

You should see the version number (e.g., ``2.0.0``) printed.

Running the Tests
-----------------

To verify the installation and ensure everything works correctly, run the 
test suite::

    cd OPI_python
    python -m pytest tests/

Or run a specific test::

    python -m pytest tests/test_models.py

Troubleshooting
---------------

ImportError: No module named 'opi'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This error indicates that Python cannot find the OPI package. Ensure that:

1. The ``OPI_python`` directory is in your Python path
2. You are running Python from the correct directory
3. The package files are present in the ``opi/`` directory

NetCDF4 Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues installing NetCDF4, you may need to install the 
NetCDF C library first:

* **Ubuntu/Debian**::

      sudo apt-get install libnetcdf-dev

* **macOS (with Homebrew)**::

      brew install netcdf

* **Windows**: Use the pre-built wheels::

      pip install netCDF4

For more help, see the `NetCDF4 documentation <https://unidata.github.io/netcdf4-python/>`_.

Getting Help
------------

If you encounter issues not covered here:

1. Check the :doc:`tutorials/index` for common usage patterns
2. Review the :doc:`api/modules` for detailed function documentation
3. Look at the example scripts in the ``examples/`` directory
4. Open an issue on the GitHub repository
