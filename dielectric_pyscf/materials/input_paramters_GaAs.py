#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

"""Naming-parameters: system_name and name of final file"""
system_name = '/gpfs/scratch/mhott/dielectric_pyscf_results/GaAs_cc-pvtz_pbe_10k'
res_filename = system_name + '_eps.hdf5'
DFT_resources_path = '/gpfs/scratch/mhott/dielectric_pyscf_results'

mpi = True                         # If True, MPI parallelization will be implemented

alt_binning = False                 #Temporary: set to True to use alternate binning technique, where interpolation and binning happen at end only. Do not use for large q since epsilon is kept in memory for all q+G vectors.

"""Build the periodic system, all units in angstrom"""
a = 5.653                           # Crystal lattice constant
lattice_vectors = [[a/2,      a/2,      0. ], 
                   [0.,       a/2,      a/2], 
                   [a/2,      0.,       a/2]]
atomloc =  ''' Ga 0 0 0; As {} {} {}'''.format(a/4, a/4, a/4)
mybasis = {'Ga': 'cc-pvtz', 'As': 'cc-pvtz'}        # Recommended to keep all basis sets same, however can be made different. We follow pyscf notations, see pyscf documentation

orth = False                   # Perform orthogonalization on SCF calculation. Often required for diffuse basis functions or large basis sets
density_fitting = 'MDF' #'RSDF'

effective_core_potential = None    # All electron basis sets mandate the use of no ECPs, basis sets and ECPS must be chosen consistently
pseudo = None                      # Similar to ECPs, slight differences (uses CP2K based systems)

"""Computational parameters: these determine precision and whether optional parameters are applied"""
precision = 1e-12                   # DFT precision parameter, fed to pyscf only
precision_R = 1e-9                 # R cutoff precision, used for dielectric function precision
xcfunc =   'pbe'#'HYB_GGA_XC_HSE06'                     # Exchange-Correlation Functional
k_grid = [10,10,10]                # k-grid: resolution of grid points in reciprocal space
q_shift_dir = [1,1,1]              # Direction for q-shift (gets normalized automatically)
q_shift = 0.01                     # In units of alpha*(mass of electron), magnitude of q shift 
scissor_bandgap = 1.424              # If float in eV, the scissor correction is applied to meet the specified bandgap. If None, scissor correction is not applied
include_lfe = False                # If False, does not incorporate local field effects into the calculation of epsilon. If True, LFEs will be calculated by inverting eps_{GG'}

"""Parameters for dielectric function calculations"""
dq = 0.02                          # In units of alpha*(mass of electron), size of momentum bins 
q_max = 1                          # In units of alpha*(mass of electron), maximum momentum
q_min = 0                          # In units of alpha*(mass of electron), minimum momentum
N_theta = 9                        # Number of theta bins
N_phi = 16                         # Number of phi bins
dE = 0.1                           # In eV, size of energy bins
E_max = 50                         # In eV, maximum energy
numval = 'auto'                     # Number of valence bands to include in the calculation, use 'all' for all available valence bands and 'auto' to exclude irrelevant bands based on E_max
numcon = 'auto'                     # Number of conduction bands to include in the calculation, use 'all' for all available conduction bands and 'auto' to exclude irrelevant bands based on E_max

"""Logging and calculation parameters"""
q_start = None                     # If None, calculation is performed for all q vectors. If set to an integer, the 
                                   # calculations are started at that q vector. Only use if previous calculation was interrupted.
q_stop = None

store = system_name + '_resources' # Location to store intermediate calculations to reduce memory load
qcdark_outfile = system_name + '_eps.log'
pyscf_outfile = system_name + '_pyscf.log'
pyscf_outlev = 4
