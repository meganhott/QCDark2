---
title: Outputs
layout: default
nav_order: 5
---
# Outputs

## Dielectric Function
The dielectric function is saved as an hdf5 file in the specified output directory as `epsilon.hdf5`. The information in this file can be read directly with the `h5py` python package:
```
h5 = h5py.File('epsilon.hdf5', 'r')
h5.keys() # lists all datasets
h5.attrs.keys() # lists all attributes
```
The results can also be loaded in as a dielectric function object with the `dark_matter_rates.py` module:
```
import dielectric_pyscf.dark_matter_rates as dm
epsilon = dm.load_epsilon('epsilon.hdf5')
```
`epsilon` is a `dm.df` object with the following attributes copied over from the hdf5 file:

| dm.df | hdf5 | Description |
| ----- | ---- | ----------- |
| `epsilon.eps` | `h5['epsilon'][:]` | $\epsilon(q, \omega)$ |
| `epsilon.E` | `h5['E'][:]` | $\omega$ in eV | 
| `epsilon.q` | `h5['q'][:]` | $q$ in a.u. |
| `epsilon.M_cell` | `h5.attrs['M_cell']` | Mass of unit cell in eV |
| `epsilon.V_cell` | `h5.attrs['V_cell']` | Volume of unit cell in (bohr<sup>-1<\sup>)<sup>3<\sup> |
| `epsilon.dE` | `h5.attrs['dE']` | $\Delta \omega$ in eV|

The hdf5 file contains extra information about the input parameters which can be accessed as attributes like `M_cell` and `V_cell` above. `h5.attrs.keys()` will return all attributes. For more information on working with hdf5 files in python, see the `h5py` [docs](https://docs.h5py.org/en/stable/).  

The energy loss function (ELF) and dynamic structure factor (S) can be obtained through their respective methods in the dielectric function class:
```
epsilon.elf()
epsilon.S()
```

## Post-processing
The `plot_templates.py` module has many useful functions for quickly plotting the dielectric function, ELF, and dynamic structure factor. See the Jupyter notebook located in **<>** for examples of these plots.  

`dark_matter_rates.py` and Jupyter notebook
