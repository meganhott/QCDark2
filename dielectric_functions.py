#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function: 
    RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
    version: 0.01
    authors: github: @meganhott; github: @asingal14
"""

import dark_objects_routines as do_routines
import dft_routines
import epsilon_routines as eps_routines
import utils
import input_parameters as parmt

def initialize_cell() -> tuple[do_routines.pbcgto.cell.Cell, dict]:
    cell = do_routines.build_cell_from_input()
    primgauss = do_routines.gen_all_1D_prim_gauss(cell)
    primindices = do_routines.gen_prim_gauss_indices(primgauss)
    aos = do_routines.gen_all_atomic_orbitals(cell, primgauss)
    blocks = do_routines.get_basis_blocks(aos)
    G_vectors = do_routines.gen_G_vectors(cell)
    R_vectors = do_routines.construct_R_vectors(cell)
    V_cell = do_routines.get_cell_volume(cell)
    
    dark_objects = {
        'primitive_gaussians': primgauss,
        'aos': aos,
        'G_vectors': G_vectors,
        'primindices': primindices[0],
        'atom_locs': primindices[1],
        'R_vectors': R_vectors,
        'blocks': blocks,
        'V_cell': V_cell
    }
    return cell, dark_objects

def electronic_structure(cell: do_routines.pbcgto.cell.Cell, dark_objects: dict) -> dict:
    kmf = dft_routines.KS_electronic_structure(cell)
    dft_routines.KS_non_self_consistent_field(kmf)
    dft_routines.get_band_indices()
    return dark_objects

def dielectric_RPA(cell: do_routines.pbcgto.cell.Cell, dark_objects: dict) -> dict:
    dft_routines.convert_to_eV_and_scissor(cell)
    dark_objects['unique_q'] = do_routines.get_1BZ_q_points(cell)
    dark_objects['R_cutoffs'] = do_routines.primgauss_1D_overlaps(dark_objects)
    dark_objects['R_cutoff_q_points'] = do_routines.store_R_ids(dark_objects)
    return dark_objects

def main():
    cell, dark_objects = initialize_cell()
    if parmt.new_dft:
        dark_objects = electronic_structure(cell, dark_objects)
    dark_objects = dielectric_RPA(cell, dark_objects)

    #Calculate and save interpolated 3D binned epsilon
    eps_routines.get_RPA_dielectric(dark_objects)

if __name__ == '__main__':
    utils.check_requirements() # Check requirements
    utils.patch() # Patch required for some versions, see function for details
    main()
