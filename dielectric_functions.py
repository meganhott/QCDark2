#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function:
     RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
     version: 0.01
     authors: github: @meganhott; github: @asingal14
     
Implementation steps to implement next:
     1. Construct overlap of primitive 1D gaussians, routines.py/calc_ovlp_1D_prim_gauss
     2. Implement a function to do the density functional theory calculation, and save results to store.
     ...
     -1.Implement store:
          Check if hdf5 file previously exists. 
          Y. Check if the parameters of calculation have changed.
               Y. Continue with all previously saved data.
               N.   1. Check if cell and k-grid parameters are the same.
                         Y. Check if DFT parameters are the same (Exchange-Correlation)
                              Y. continue with same DFT.
                              N. delete DFT related files and parameters if they exist.
                         N. delete hdf5 file and all data in store, and move to step (-1.N).
                    2. Delete all pre-existing epsilon results, and molecular orbital overlap files if they exist.
          N. Create new file.
               1. Add/update attributes and parameters.
               2. Make all necessary folders
          
"""

import numpy as np
import routines as routines
import input_parameters as parmt

def initialize_cell() -> tuple[routines.pbcgto.cell.Cell, dict]:
     cell = routines.build_cell_from_input()                                    # Build cell object
     primgauss = routines.gen_all_1D_prim_gauss(cell)                           # Get all primitive gaussian objects
     primindices = routines.gen_prim_gauss_indices(primgauss)                   # Get all main indices for primitive gaussian objects. 
     all_ao = routines.gen_all_atomic_orbitals(cell, primgauss)                 # Get all atomic orbitals
     G_vectors = routines.gen_G_vectors(cell)                                   # Get all relevant G vectors
     dark_objects = {
          'primitive_gaussians': primgauss,
          'all_ao': all_ao,
          'G_vectors': G_vectors,
          'primguass_i': primindices
     }
     return cell, dark_objects

def electronic_structure(cell: routines.pbcgto.cell.Cell, dark_objects: dict) -> None:
     kmf = routines.KS_electronic_structure(cell)
     routines.KS_non_self_consistent_field(kmf)
     dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
     return None

def main():
     cell, dark_objects = initialize_cell()
     electronic_structure(cell, dark_objects)
     return

if __name__ == '__main__':
     routines.check_requirements()                                              # Check requirements
     routines.patch()                                                           # Patch required for some versions, 
                                                                                # see function for details
     main()