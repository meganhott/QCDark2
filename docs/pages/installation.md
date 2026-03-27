---
title: Installation
layout: default
nav_order: 3
---
# Installation

QCDark2 can be installed locally as a python package from the github repository:  
```
git clone https://github.com/meganhott/QCDark2.git
cd QCDark2
pip install .
```
Dark matter-electron scattering rates can be calculated with existing dielectric functions (Si, Ge, GaAs, SiC, and diamond are available on [github](https://github.com/meganhott/QCDark2/tree/main/dielectric_functions)) by importing the package in a python shell:
```
import qcdark2.dark_matter_rates as dm
```
See the [notebook](https://github.com/meganhott/QCDark2/blob/main/examples/DM_calculations.ipynb) for examples of DM scattering rate calculations.  

A new dielectric function calculation can be performed with material parameters specified in an input file (see [materials](https://github.com/meganhott/QCDark2/tree/main/qcdark2/materials)):
```
python3 -m qcdark2.dielectric_pyscf input_file.in
``` 

The following python version and dependencies are required:  
 - python > 3.9.6    
 - pyscf > 2.9.0   
 - numpy  
 - numba  
 - h5py   
 - scipy   
 - psutil   

The following packages are optional  
- basis_set_exchange (Includes some Gaussian basis sets that PySCF does not include natively.)  
- mpi4py (Highly recommended for large calculations on computing clusters so multiple nodes can be used.)  