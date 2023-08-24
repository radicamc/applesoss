## Code for the APPLESOSS Empirical Profile Construction Module

The empirical profile construction, aka the **APPLESOSS** (A Producer of ProfiLEs for SOSS) module builds 2D spatial profiles for the first, second, and third diffraction orders for a NIRISS/SOSS
GR700XD/CLEAR observation. The profiles are entirely data driven and retain a high level of fidelity to the original observations.

These profiles are intended to be used as the specprofile reference file for the **ATOCA** 1D extration algorithm, or alternatively as a PSF weighting for optimal extractions.

The latest release of **APPLESOSS** can be downloaded from PyPI via:

    pip install applesoss

or the latest development version can be grabbed from GitHub:

    git clone https://github.com/radicamc/applesoss
    cd applesoss
    python setup.py install

The GitHub version also includes testing data and examples. 

If you wish to use simulations instead of purely empirical wings to construct the trace profiles, you will need 
the ```webbpsf``` package, which can be included via the following option during installation:

    pip install applesoss[webbpsf]

Alternatively, its installation instructions can be found [here](https://webbpsf.readthedocs.io/en/latest/installation.html#installing-or-updating-synphot).


If you make use of this code, please cite the [APPLESOSS paper](https://ui.adsabs.harvard.edu/abs/2022arXiv220705136R/abstract).
