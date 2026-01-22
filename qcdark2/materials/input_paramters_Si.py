#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

"""Naming-parameters: system_name and name of final file"""
system_name = '/gpfs/scratch/mhott/dielectric_pyscf_results/Si_cc-pvtpdz_pbe_8k_8q_lfe_139_223'
res_filename = system_name + '_eps.hdf5'
DFT_resources_path = '/gpfs/scratch/mhott/dielectric_pyscf_results'

mpi = True                         # If True, MPI parallelization will be implemented
q_start = 139
q_stop = 223

"""Build the periodic system, all units in angstrom"""
a = 5.43 #5.4877                   # Crystal lattice constant
lattice_vectors = [[a/2,      a/2,      0. ], 
                   [0.,       a/2,      a/2], 
                   [a/2,      0.,       a/2]]
atomloc =  ''' Si 0 0 0; Si {} {} {}'''.format(a/4, a/4, a/4)
mybasis = {'Si': 'cc-pv(t+d)z'}        # Recommended to keep all basis sets same, however can be made different. We follow pyscf notations, see pyscf documentation

orth = False                   # Perform orthogonalization on SCF calculation. Often required for diffuse basis functions or large basis sets
density_fitting = 'MDF' #'RSDF'

effective_core_potential = None    # All electron basis sets mandate the use of no ECPs, basis sets and ECPS must be chosen consistently
pseudo = None                      # Similar to ECPs, slight differences (uses CP2K based systems)

"""Computational parameters: these determine precision and whether optional parameters are applied"""
precision = 1e-12                   # DFT precision parameter, fed to pyscf only
precision_R = 1e-9                 # R cutoff precision, used for dielectric function precision
xcfunc =   'pbe'#'HYB_GGA_XC_HSE06'                     # Exchange-Correlation Functional
k_grid = [8,8,8]                # k-grid: resolution of grid points in reciprocal space
q_shift_dir = [1,1,1]              # Direction for q-shift (gets normalized automatically)
q_shift = 0.01                     # In units of alpha*(mass of electron), magnitude of q shift 
scissor_bandgap = 1.1              # If float in eV, the scissor correction is applied to meet the specified bandgap. If None, scissor correction is not applied
include_lfe = True                # If False, does not incorporate local field effects into the calculation of epsilon. If True, LFEs will be calculated by inverting eps_{GG'}

"""Parameters for dielectric function calculations"""
dq = 0.02                          # In units of alpha*(mass of electron), size of momentum bins 
q_max = 8                          # In units of alpha*(mass of electron), maximum momentum
q_min = 0                          # In units of alpha*(mass of electron), minimum momentum
N_theta = 9                        # Number of theta bins
N_phi = 16                         # Number of phi bins
dE = 0.1                           # In eV, size of energy bins
E_max = 50                         # In eV, maximum energy
numval = 'auto'                     # Number of valence bands to include in the calculation, use 'all' for all available valence bands and 'auto' to exclude irrelevant bands based on E_max
numcon = 'auto'                     # Number of conduction bands to include in the calculation, use 'all' for all available conduction bands and 'auto' to exclude irrelevant bands based on E_max

"""Logging and calculation parameters"""
store = system_name + '_resources' # Location to store intermediate calculations to reduce memory load
qcdark_outfile = system_name + '_eps.log'
pyscf_outfile = system_name + '_pyscf.log'
pyscf_outlev = 4
debug_logging = False