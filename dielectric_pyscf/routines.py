#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

import os
import time
import logging
import numpy as np
from functools import wraps, partial

import dielectric_pyscf.input_parameters as parmt

##==== Constants in the whole calculation ============
c = 299792458.                                          # c in m/s
me = 0.51099895000e6                                    # eV/c^2
alpha = 1/137

##==== Conversion factors ============================
har2ev = 27.211386245988                                # convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28))     # convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11)                        # convert Bohr radius to meter
a2bohr = 1.8897259886                                   # convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6)                           # hbarc in eV*m
amu2eV = 9.315e8                                        # eV/u

def time_wrapper(func=None, *, n_tabs=0):
    """
    Wrapper for printing execution time to logger.
    """
    if func is None:
        return partial(time_wrapper, n_tabs=n_tabs)
    
    @wraps(func)
    def wrap(*args, **kwargs):
        logger.info('\t'*n_tabs + f'Entering function {func.__name__}')
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        logger.info('\t'*n_tabs + f'Exiting function {func.__name__}. Time taken = {(end - start):.2f} s.\n')
        return val
    
    @wraps(func) # For MPI so we only log rank 0
    def wrap_nolog(*args, **kwargs):
        val = func(*args, **kwargs)
        return val

    # When using MPI, only log for rank 0
    if parmt.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank

        if rank == 0:
            return wrap
        else:
            return wrap_nolog
        
    else: # Not using MPI
        return wrap

def makedir(dirname: str, log = False):
    """
    Function that makes a directory if it does not exist.
    Inputs:
        dirname (str):
            Name of directory to be constructed
        log (bool):
            Switch to control logging. Construction of new directory to be logged or not?
    """
    try:
        os.mkdir(dirname)
        if log:
            logger.info(f'Made directory {dirname}')
    except FileExistsError:
        if os.path.isdir(dirname):
            if log:
                logger.info(f'Directory {dirname} already exists.')
        else:
            logger.info(f'File exists with name same as directory {dirname}. Cannot make directory, raising exception.')
            raise FileExistsError(f'Cannot proceed with making directory {dirname}, file exists with same name.')

def get_all_unique_nums_in_array(array: np.ndarray, round_to: int = None, log_name: str = None) -> np.ndarray:
    """
    Condense array to 1D and find all unique unique elements. 
    Simplifies np.unique(np.round(array, round_to))
    Inputs:
        array (np.ndarray)
        round_to (int):
            Rounding precision
    """
    if not round_to is None:
        array = np.round(array, round_to)
    unq = np.unique(array)
    if not log_name is None:
        logger.info(f'Number of unique elements found for {log_name} is {unq.size}.')
    return unq

@time_wrapper
def get_all_unique_vectors_in_array(array: np.ndarray, round_to: int = None) -> np.ndarray:
    """
    Finds all unique vectors in an (n, 3) array of vectors. 
    """
    if not round_to is None:
        array = np.round(array, round_to)
    unq = np.unique(array, axis=0)
    return unq

# Initialize log
makedir(parmt.store, log=False)
logger = logging
logger.basicConfig(filename=parmt.qcdark_outfile, filemode = 'w', level=logging.INFO, format='%(message)s')