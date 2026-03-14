# QCDark2
QCDark2 computes the RPA dielectric function $\epsilon(\omega, \mathbf{q})$ at finite momentum to use in dark matter-electron scattering calculations. DFT is performed with PySCF using Gaussian-type orbitals, then the RPA dielectric function is calculated analytically using the properties of Gaussian basis sets. Local field effects can be included in the RPA calculation. Standard DM calculations including scattering rates, electron recoil spectra, and sensitivity projections can be performed with the functions in `dark_matter_rates.py`.   
- Pre-computed dielectric functions for Si, Ge, GaAs, SiC, and diamond are in [dielectric_functions](dielectric_functions/)  
- We include a notebook showing examples of DM calculations: [DM_calculations](examples/DM_calculations.ipynb)  

RPA dielectric functions for new functions can also be computed from scratch. To do so, this repository can be downloaded and installed as a python package.  