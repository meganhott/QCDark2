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

import input_parameters as parmt

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

#Initialize log
logger = logging
logger.basicConfig(filename=parmt.qcdark_outfile, filemode = 'w', level=logging.INFO, format='%(message)s')

def time_wrapper(func=None, *, n_tabs=0):
    """
    Wrapper for printing execution time to logger.
    """
    if func is None:
        return partial(time_wrapper, n_tabs=n_tabs)
    
    @wraps(func)
    def wrap(*args, **kwargs):
        logger.info('\t'*n_tabs + 'Entering function {}'.format(func.__name__))
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        logger.info('\t'*n_tabs + 'Exiting function {}. Time taken = {:.2f} s.\n'.format(func.__name__, end - start))
        return val

    return wrap

def makedir(dirname: str, log = False) -> None:
    """
    Function that makes a directory if it does not exist.
    Inputs:
        dirname:    str, name of directory to be constructed
        log:        bool, default = False
                    switch to control logging. Construction of new directory to be logged or not?
    """
    try:
        os.mkdir(dirname)
        if log:
            logger.info('Made directory \'' + dirname + '\'.')
    except FileExistsError:
        if os.path.isdir(dirname):
            if log:
                logger.info('Directory \''+dirname+'\' already exists.')
        else:
            logger.info('File exists with name same as directory, \'' + dirname + '\'. Cannot make directory, raising exception.')
            raise FileExistsError('Cannot proceed with making directory \'' + dirname + '\', file exists with same name.')
    return None

def get_all_unique_nums_in_array(array: np.ndarray, round_to: int = None, log_name: str = None) -> np.ndarray:
    """
    Given any array, np.ndarray, condense it into 1D and find all unique unique elements.
    function to simplify np.unique(np.round(array, round_to)).
    Inputs:
        array:      np.ndarray
        tol:        round to digits
    """
    if not round_to is None:
        array = np.round(array, round_to)
    unq = np.unique(array)
    if not log_name is None:
        logger.info("Number of unique elements found for {} = {}.".format(log_name, unq.size))
    return unq

@time_wrapper
def get_all_unique_vectors_in_array(array: np.ndarray, round_to: int = None) -> np.ndarray:
    """
    Finds all unique vectors in an (n, 3) array of vectors 
    """
    if not round_to is None:
        array = np.round(array, round_to)
    unq = np.unique(array, axis=0)
    return unq

def load_unique_R():
    """
    Loads unique R in each dimension from saved 1D intgrals.
    """
    unique_Ri = []
    for dim in range(3):
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        unique_Ri.append(np.load(dir+'Ru.npy'))
    return unique_Ri
