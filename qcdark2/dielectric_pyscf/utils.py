import sys
import functools
import warnings
import struct
import numpy as np
import scipy
import pyscf

import qcdark2.dielectric_pyscf.input_parameters as parmt
from qcdark2.dielectric_pyscf.routines import logger, time_wrapper
 
@time_wrapper
def check_requirements() -> None:
    """
    Function to check requirements for the implementation of the code.
    """
    if pyscf.__version__ < '2.2.0':
        raise Exception('pyscf version {} found. The program uses features implemented in pyscf version 2.2.0, and has been verified to work in pyscf version 2.6.0.'.format(pyscf.__version__))
    elif pyscf.__version__ < '2.6.0':
        warnings.warn('pyscf version {} found. While we do not anticipate compatibility issues, the program is tested for pyscf version >= 2.6.0.'.format(pyscf.__version__))
    if pyscf.__version__ <= '2.6.0' and np.__version__ >= '2.0.0':
        warnings.warn('pyscf version {} and numpy version {} are not fully compatible. There could be errors. We recommend updating pyscf to version 2.6.1 or above.'.format(pyscf.__version__, np.__version__))
    logger.info('\tpython version {},\n\tpyscf version {},\n\tnumpy version {},\n\tscipy version {}.'.format(sys.version, pyscf.__version__, np.__version__, scipy.__version__))
    return

@time_wrapper
def patch():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.
    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s.",
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
        logger.info(patchname + " already applied, skipping.")
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

    logger.info(patchname + " applied")

def check_inputs():
    """
    Ensures correct format for parameters in input_parameters.py before starting computation.

    Need to add all checks eventually!
    """
    if type(parmt.mpi) is not bool:
        raise ValueError('Parameter "mpi" in input_parameters.py must be either True if running with MPI or False if not using MPI.')
    
    if type(parmt.include_lfe) is not bool:
        raise ValueError('Parameter "include_lfe" in input_parameters.py must be either True if including local field effects in the dielectric function calculation or False if not.')

    if (parmt.lfe_q_cutoff is not None) or (type(parmt.lfe_q_cutoff) is not float):
        raise ValueError('Parameter lfe_q_cutoff in input_parameters.py must be either None or of type float.')