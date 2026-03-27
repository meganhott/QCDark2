---
title: Home
layout: home
nav_order: 1
---
# Overview

[QCDark2](https://github.com/meganhott/QCDark2) computes the RPA dielectric function at finite momentum to use in dark matter-electron scattering calculations. DFT is performed with [PySCF](https://pyscf.org/) using Gaussian-type orbitals, then the RPA dielectric function is calculated analytically using the properties of Gaussian basis sets. Local field effects can be included in the RPA calculation. Standard DM calculations including scattering rates, electron recoil spectra, and sensitivity projections can be performed with the functions in `qcdark2.dark_matter_rates`.
# What can be calculated with QCDark2?
### Dielectric Function
- RPA dielectric function with local field effects (LFEs): $\epsilon^{\mathrm{LFE}}(\vec{q}, \omega)$ or without LFEs: $\epsilon^{\mathrm{noLFE}}(\vec{q}, \omega)$
- 3D dielectric function $\epsilon(\vec{q}, \omega)$
- Directionally-averaged 1D dielectric function $\epsilon(q, \omega)$

### Dark Matter Direct Detection
Currently the following are only implemented for spin-independent DM-electron interactions:
- Ionization yield rates $\Delta R_Q$
- Scattering rates $dR/d\omega$
- Projected sensitivity in $(m_{\chi}, \bar{\sigma}_e)$  

See the [notebook](github.com/meganhott/QCDark2/blob/main/examples/DM_calculations.ipynb) for examples of these calculations.

# Papers

C. Dreyer, R. Essig, M. Fernandez-Serra, M. Hott, A. Singal, *All-electron dark matter-electron scattering with random-phase approximation dielectric screening and local field effects* [2603.12326](https://arxiv.org/abs/2603.12326)