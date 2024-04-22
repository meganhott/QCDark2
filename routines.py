#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

import os, time, itertools, logging, pdb, struct, sys, h5py, functools
import numpy as np
import pyscf.pbc.dft as pbcdft
import cartesian_moments as cartmoments
from multiprocessing import Pool
from functools import partial
import shutil
import dielectric_functions

##==== Constants in the whole calculation ============
c = 299792458 #c in m/s
me = 0.51099895000*10**6 #eV/c^2
alpha = 1/137

##==== Conversion factors ============================
har2ev = 27.211386245988 #convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28)) #convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter
a2bohr = 1.8897259886 #convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m
amu2eV = 9.315e8 # eV/u

logging.basicConfig(filename=parmt.logname, level=logging.INFO, format='%(message)s') 

def patch():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logging.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
        orig_send_bytes.__code__.co_filename == __file__
        and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logging.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logging.info(patchname + " applied")
