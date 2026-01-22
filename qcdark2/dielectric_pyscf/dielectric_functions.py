#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function: 
    RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
    version: 0.01
    authors: github: @meganhott; github: @asingal14
"""

import dielectric_pyscf.dark_objects_routines as do_routines
import dielectric_pyscf.dft_routines as dft_routines
import dielectric_pyscf.epsilon_routines as eps_routines
import dielectric_pyscf.utils as utils
import dielectric_pyscf.input_parameters as parmt

def initialize_cell() -> tuple[do_routines.pbcgto.cell.Cell, dict]:
    cell = do_routines.build_cell_from_input()
    primgauss = do_routines.gen_all_1D_prim_gauss(cell)
    primindices = do_routines.gen_prim_gauss_indices(primgauss)
    aos = do_routines.gen_all_atomic_orbitals(cell, primgauss)
    primgauss_arr, AO_arr, coeff_arr = do_routines.get_ao_blocks(aos)
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
        'primgauss_arr': primgauss_arr,
        'AO_arr': AO_arr,
        'coeff_arr': coeff_arr,
        'V_cell': V_cell,
        'cell': cell
    }
    return cell, dark_objects

def electronic_structure(cell: do_routines.pbcgto.cell.Cell, dft_params: dict):
    kmf = dft_routines.KS_electronic_structure(cell, dft_params, orth=parmt.orth)
    dft_routines.KS_non_self_consistent_field(kmf, dft_params)

def dielectric_RPA(cell: do_routines.pbcgto.cell.Cell, dark_objects: dict, dft_params: dict) -> dict:
    dft_routines.convert_to_eV_and_scissor(cell, dft_params)
    dft_routines.get_band_indices(dft_params)
    dark_objects['unique_q'] = do_routines.get_1BZ_q_points(cell)
    dark_objects['R_cutoffs'] = do_routines.primgauss_1D_overlaps(dark_objects)
    dark_objects['R_cutoff_q_points'] = do_routines.store_R_ids(dark_objects)
    dark_objects['unique_Ri'] = do_routines.load_unique_R()
    return dark_objects

def main_setup() -> dict:
    cell, dark_objects = initialize_cell()

    new_dft, dft_params = dft_routines.save_dft(cell)
    if new_dft:
        electronic_structure(cell, dft_params)
    dark_objects = dielectric_RPA(cell, dark_objects, dft_params)
    return dark_objects

def main_eps(dark_objects, rank, q_start, q_stop):
    bin_eps, bin_weights, bin_centers = eps_routines.get_RPA_dielectric(dark_objects, rank, q_start, q_stop)
    return bin_eps, bin_weights, bin_centers

if __name__ == '__main__':
    utils.check_requirements() # Check requirements
    utils.patch() # Patch required for some versions, see function for details
    dark_objects = main_setup()
    main_eps(dark_objects)

