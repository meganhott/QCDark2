# QCDark2

[Documentation](https://meganhott.github.io/QCDark2)

QCDark2 computes the RPA dielectric function $\epsilon(\omega, \mathbf{q})$ at finite momentum to use in dark matter-electron scattering calculations. DFT is performed with [PySCF](https://pyscf.org/) using Gaussian-type orbitals, then the RPA dielectric function is calculated analytically using the properties of Gaussian basis sets. Local field effects can be included in the RPA calculation. Standard DM calculations including scattering rates, electron recoil spectra, and sensitivity projections can be performed with the functions in `dark_matter_rates.py`.   
- Pre-computed dielectric functions for Si, Ge, GaAs, SiC, and diamond are in [dielectric_functions](dielectric_functions/)  
- We include a notebook showing examples of DM calculations: [DM_calculations](examples/DM_calculations.ipynb)  
- To calculate the dielectric function for a new material, the material parameters can be specified in an input file. See the [documentation](https://meganhott.github.io/QCDark2)

If you would like to download all results and example notebooks included in this repo as well as install the package, you can clone the repo and then install as a package with pip: 
```
git clone https://github.com/meganhott/QCDark2.git
cd QCDark2
pip install .
```

qcdark2 can then be imported as a package in a python shell (see example notebook [DM_calculations](examples/DM_calculations.ipynb))  

To run a new dielectric function calculation:
```
python3 -m qcdark2.dielectric_pyscf input_file.in
```
Example input files can be found in [materials](qcdark2/materials/).

Requirements:
 - python > 3.9.6  
 - pyscf > 2.9.0  
Optional:
- mpi4py (recommended for large computing clusters but not required)

More details about QCDark2 can be found in [2603.12326](https://arxiv.org/abs/2603.12326)

The previous iteration of this code can be found at [QCDark1](https://github.com/asingal14/QCDark)

What is different from QCDark1?
- Full RPA dielectric function calculated for ab initio screening
- Better low-momentum resolution and binning
- Local field effects can be included up to moderately-high momentum 
- Optimized code
