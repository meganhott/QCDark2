#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

import os

"""Naming-parameters: system_name & name of final file"""
system_name = 'SiC_cc-pvdz_pbe_8k'
res_filename = system_name + '_eps.hdf5'

alt_binning = False #Temporary: set to True to use alternate binning technique, where interpolation and binning happen at end only. Do not use for large q since epsilon is kept in memory for all q+G vectors.

"""Build the periodic system, all units in angstrom"""
lattice_vectors = [[2.1798,    2.1798,    0. ], 
                   [0.,       2.1798,    2.1798], 
                   [2.1798,    0.,       2.1798]]
atomloc =  ''' Si	0.	     0.	     0.    
		     C	1.0899	1.0899	1.0899'''
mybasis = {'Si': 'cc-pvdz', 'C': 'cc-pvdz'}                            # Recommended to keep all basis sets same, however can be made different
                                                       # We follow pyscf notations, see pyscf documentation
effective_core_potential = None                        # All electron basis sets mandate the use of no ECPs,
                                                       # basis sets and ECPS must be chosen consistently
pseudo = None                                          # Similar to ECPs, slight differences (uses CP2K based systems)

"""computational parameters;
     these determine precision, and whether optional parameters are applied"""
precision = 1e-12                                      # DFT precision parameter, fed to pyscf only
precision_R = 1e-9                                     # R cutoff precision
xcfunc = 'pbe'                                         # Exchange-Correlation Functional
k_grid = [8,8,8]                                       # k-grid: resolution of grid points in reciprocal space
q_shift_dir = [1,1,1]                                  # Direction for q-shift
q_shift = 0.01                                         # In units of alpha*(mass of electron) 
scissor_bandgap = 2.36                                  # If None, scissor correction is not applied, generally float in eV
include_lfe = False                                    # If False, does not incorporate local field effects into the calculation of 
                                                       # epsilon. If True, LFEs will be calculated by inverting eps_{GG'} 

"""parameters for dielectric function calculations,
     including q_max, bin widths, number of bands, etc"""
dq = 0.02                                              # In units of alpha*(mass of electron) 
q_max = 1                                              # In units of alpha*(mass of electron)
q_min = 0                                              # In units of alpha*(mass of electron)
N_theta = 9                                            # Number of theta bins
N_phi = 16                                             # Number of phi bins
dE = 0.1                                               # In eV
E_max = 50                                             # In eV
numval = 'all'                                         # number of valence bands to include in the calculation, 
                                                       # use 'all' for all available valence bands
numcon = 'all'                                         # number of conduction bands to include in the calculation,
                                                       # use 'all' for all available conduction bands

"""Logging and calculation parameters"""
save_temp_eps = True                                   # Save calculations after each q is completed
q_start = None                                         # If None, calculation is performed for all q vectors. If set to an integer, the 
                                                       # calculations are started at that q vector. Only use if previous calculation was interrupted.

store = system_name + '_resources'                     # Need this: allows us to keep heavy objects off memory,
                                                       # as well as allows us to resume if calculation stopped at any point.
qcdark_outfile = system_name + '_eps.log'
pyscf_outfile = system_name + '_pyscf.log'
pyscf_outlev = 4
