#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function: 
    RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
    version: 0.01
    authors: github: @meganhott; github: @asingal14
"""
    
import routines as routines

def initialize_cell() -> tuple[routines.pbcgto.cell.Cell, dict]:
    cell = routines.build_cell_from_input()                                    # Build cell object
    primgauss = routines.gen_all_1D_prim_gauss(cell)                           # Get all primitive gaussian objects
    primindices = routines.gen_prim_gauss_indices(primgauss)                   # Get all main indices for primitive gaussian objects.
    all_ao = routines.gen_all_atomic_orbitals(cell, primgauss)                 # Get all atomic orbitals
    G_vectors = routines.gen_G_vectors(cell)                                   # Get all relevant G vectors
    R_vectors = routines.construct_R_vectors(cell)
    dark_objects = {
        'primitive_gaussians': primgauss,
        'all_ao': all_ao,
        'G_vectors': G_vectors,
        'primindices': primindices[0],
        'atom_locs': primindices[1],
        'R_vectors': R_vectors
    }
    return cell, dark_objects

def electronic_structure(cell: routines.pbcgto.cell.Cell, dark_objects: dict) -> dict:
    kmf = routines.KS_electronic_structure(cell)
    routines.KS_non_self_consistent_field(kmf)
    routines.convert_to_eV_and_scissor(cell)
    dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
    return dark_objects

def dielectric_RPA(dark_objects: dict) -> None:
    routines.primgauss_1D_overlaps(dark_objects)
    return dark_objects

def main():
    cell, dark_objects = initialize_cell()
    dark_objects = electronic_structure(cell, dark_objects)
    dark_objects = dielectric_RPA(dark_objects)
    return

if __name__ == '__main__':
    routines.check_requirements()                                              # Check requirements
    routines.patch()                                                           # Patch required for some versions, see function for details
    main()
