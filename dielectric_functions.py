#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function:
     RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
     version: 0.01

Implementation details:
     Current facilities available: 
          None.
     Implementation procedure:
          1. Switch from a 3-dimensional Etab formulation to 1-dimensional E-tab formulation
               1.1 Build all atomic orbitals such that one can construct each orbital from just the relevant primitive gaussians.
               1.2 Use primitive gaussians as indexed items. 
               1.3 

          . Switch from contracted gaussians to primitive gaussian-based calculations
               .1 Change cartesian_moments.py code to build 1-dimensional integrals over primitive gaussians.
               .2 Test direct input times versus interpolation time-scales. 
"""

import time, pdb, logging
import numpy as np
import pdb
import logging
import pyscf.lib
import routines as routines
import input_parameters as parmt

c = 299792458 #c in m/s
me = 0.51099895000*10**6 #eV/c^2
alpha = 1/137

##==== Conversion factors ==============================
har2ev = 27.211386245988 #convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28)) #convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter
a2bohr = 1.8897259886 #convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m

def main():
     cell = routines.build_cell_from_input()
     return

if __name__ == '__main__':
     main()