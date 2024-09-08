#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

import os, time, itertools, logging, pdb, struct, sys, h5py, functools, psutil
import warnings
import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc
import pyscf
import scipy
import scipy.interpolate
import cartesian_moments as cartmoments
import multiprocessing as mp
from functools import partial
import shutil
import input_parameters as parmt
import binning as bin
from numba import njit

##==== Constants in the whole calculation ============
c = 299792458.                                                                  # c in m/s
me = 0.51099895000e6                                                            # eV/c^2
alpha = 1/137

##==== Conversion factors ============================
har2ev = 27.211386245988                                                        # convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28))                             # convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11)                                                # convert Bohr radius to meter
a2bohr = 1.8897259886                                                           # convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6)                                                   # hbarc in eV*m
amu2eV = 9.315e8                                                                # eV/u

logging.basicConfig(filename=parmt.qcdark_outfile, filemode = 'w', level=logging.INFO, format='%(message)s') 

#temporary
try:
    import configure_pyscf
except:
    pass

def time_wrapper(func):
    """
    Wrapper for printing execution time to logger.
    """
    def wrap(*args, **kwargs):
        logging.info('Entering function {}'.format(func.__name__))
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        logging.info('Exiting function {}. Time taken = {:.2f} s.\n'.format(func.__name__, end - start))
        return val

    return wrap

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
    logging.info('\tpython version {},\n\tpyscf version {},\n\tnumpy version {},\n\tscipy version {}.'.format(sys.version, pyscf.__version__, np.__version__, scipy.__version__))
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
        logging.info(
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
        logging.info(patchname + " already applied, skipping.")
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

def check_inputs():
    """
    Ensures correct format for parameters in input_parameters.py before starting computation.

    Need to add all checks eventually!
    """
    if (parmt.lfe_q_cutoff is not None) or (type(parmt.lfe_q_cutoff) is not float):
        raise ValueError('Parameter lfe_q_cutoff in input_parameters.py must be either None or of type float.')

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
            logging.info('Made directory \'' + dirname + '\'.')
    except FileExistsError:
        if os.path.isdir(dirname):
            if log:
                logging.info('Directory \''+dirname+'\' already exists.')
        else:
            logging.info('File exists with name same as directory, \'' + dirname + '\'. Cannot make directory, raising exception.')
            raise FileExistsError('Cannot proceed with making directory \'' + dirname + '\', file exists with same name.')
    return None

@time_wrapper
def build_cell_from_input() -> pbcgto.cell.Cell:
    """
    Function that builds cell from inputs.
    Note that the function requires no inputs but uses attributes defined in input_parameters.py.
    Returns:
        cell:   pyscf.pbc.gto.cell.Cell object
    """
    cell = pbcgto.M(
        a = np.asarray(parmt.lattice_vectors),
        atom = parmt.atomloc,
        basis = parmt.mybasis,
        cart = True,
        verbose = parmt.pyscf_outlev,
        output = parmt.pyscf_outfile,
        ecp = parmt.effective_core_potential,
        precision = parmt.precision
        )
    
    max_memory = psutil.virtual_memory().available/(2**20)*.9
    rcut = pbcgto.cell.estimate_rcut(cell, precision=cell.precision)
    cell.rcut = rcut
    cell.max_memory = max_memory
    cell.space_group_symmetry = True
    cell.build()

    logging.info("Built cell object. Note that we only use cartesian gaussians.\nWe have set maximum memory available to pyscf = 90% of available memory of system = {:.2f} MB".format(max_memory))
    logging.info("Parameters fed, in atomic units:")

    logging.info("\tLattice vectors:")
    for a in cell.lattice_vectors():
        logging.info("\t\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1], a[2]))

    logging.info("\tReciprocal Vectors:")
    for a in cell.reciprocal_vectors():
        logging.info("\t\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1], a[2]))

    logging.info("\tAtom locations:")
    for a in cell._atom:
        logging.info("\t\t{}:\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1][0], a[1][1], a[1][2]))
    
    logging.info("\tBasis set: ")
    for a in cell.basis:
        logging.info("\t\t{}: {}".format(a, cell.basis[a]))
    
    logging.info("\tEffective core potential:")
    if len(cell.ecp):
        for a in cell.ecp:
            logging.info("\t\t{}: {}".format(a, cell.ecp[a]))
    else:
        logging.info("\t\tNone, all-electron calculation.")
    
    logging.info("\tSelected precision: {}".format(cell.precision))
    logging.info("Further information is in {}.".format(cell.output))

    makedir(parmt.store, log = False)

    return cell

@time_wrapper
def gen_all_1D_prim_gauss(cell: pbcgto.cell.Cell) -> np.ndarray:
    """
    Generates all primitive gaussians and their 1D components, and places them into an output np.ndarray.
    Skeleton of the dielectric function procedure, links overlaps calculated in <undefined function> with
    atomic orbital calculated 
    Input:
        cell:       pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Output:
        primgauss:  np.ndarray of shape (N, 7)
                    Contains all primitive gaussians including each possible angular momentum configuration
                    in any given direction. It does not discriminate between x, y, z directions and is general

                    primgauss[index]:   [atom_index, maximum_l, i, exponent, A_x, A_y, A_z]
                    primgauss.dtype =   float
                        atom_index:     I.D. of atom, can be obtained via cell._atom (should be int, but stored as float)
                        maximum_l:      shell type, (s => 0, p => 1, d => 2, ...) (should be int, but stored as float)
                        i:              angular momentum in given dir
                                        exponent from 0 to l, inclusive (should be int, stored as float)
                        exponent:       Basis set element
                        A_x, A_y, A_z:  location of the atom from atom_index
                        norm:           cube-root of the normalization.
    """
    def normalize(exp, L) -> None:
        """
        Routine to normalize the basis functions. Note that s and p orbitals have different normalizations
        for cartesian and spherical shells, while higher orbitals carry same normalization scheme no matter what.
        Function modifies attribute:
            self.norm
        """
        if L < 2:
            return np.sqrt(np.power(2,2*(L)+1.5)*np.power(exp,L+1.5)/cartmoments.fact2(2*L-1)/cartmoments.fact2(-1)/cartmoments.fact2(-1)/np.power(np.pi,1.5))
        else:
            return np.sqrt(np.power(2, L + 2.5)*np.power(exp, L + 1.5)/cartmoments.gamma(1.5 + L))
        
    primgauss = np.zeros((0, 8))
    for atom_id, atom in enumerate(cell._atom):
        species, loc = atom[0], atom[1]
        atom_basis = cell._basis[species]
        for basis_element in atom_basis:
            l = basis_element[0]
            elements = basis_element[1:]
            for element in elements:
                norm = normalize(element[0], l)
                for i in range(l+1):
                    primgauss = np.append(primgauss, [[atom_id, l, i, element[0], loc[0], loc[1], loc[2], norm]], axis = 0)
    logging.info("All 1D primitive gaussians found for the cell.\n\tNumber of unique primitive gaussians = {}.\n\tThis includes all possible angular momentum in one direction.".format(primgauss.shape[0]))
    return primgauss

@time_wrapper
def gen_prim_gauss_indices(primgauss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function returns index in array containing atom_id, l, exponent, and initial index 
    (corresponding to i = 0) in primgauss object constructed from function gen_all_1D_primgauss.
    This function generates the objects to be passed into the <undefined function> for 
    generating overlaps. It is also called from within the controlling function.
    Input:
        primgauss:  np.ndarray object of shape (N, 7)
    Output:
        indx_arr:   np.ndarray object of shape (M, 4)
                    each row contains the following information: 
                        [atom_id, <total angular momentum on l>, exponent on primitive gaussian, <index of first point in primgauss>]
        location:   np.ndarray object of shape (num_atoms, 3), locations of atoms corresponding to atom_id
                    location[atom_id] = location of atom with atom_id
    """
    
    where = np.where(primgauss[:,2] == 0)[0]
    where = where.reshape(len(where), 1)
    indx_arr = np.append(primgauss[where[:,0]][:, [0, 1, 3]], where, axis = 1)

    unique_atoms = sorted(set(primgauss[:,0].astype(int)))
    locs = []
    for atm_id in unique_atoms:
        locs.append(primgauss[np.where(primgauss[:,0].astype(int) == atm_id)][0, 4:])

    logging.info("Constructed unique exponent for each atom. Total number of exponents at different locations: {}".format(indx_arr.shape[0]))
    return indx_arr, np.array(locs)

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
        logging.info("Number of unique elements found for {} = {}.".format(log_name, unq.size))
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

@time_wrapper
def construct_R_vectors(cell: pbcgto.cell.Cell) -> np.ndarray:
    """
    Function to construct all R vectors {R_i} relevant to the cell. We use pyscf build in methods and sort the
    R vectors from there.
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Returns:
        Rvecs:  np.ndarray of shape (N, 3), R vectors relevant to the calculation for convergence to
                cell.precision parameter.
        unR:    np.ndarray of shape (M, ), M << N, unique scalars in Rvecs. 
    """
    Rvecs = cell.get_lattice_Ls()
    Rvecs = Rvecs[np.argsort(np.linalg.norm(Rvecs, axis=1))] #sort by norm
    np.save(parmt.store + '/R_vectors.npy', Rvecs)
    logging.info('{} R vectors generated for the cell given precision = {}, and saved to {}.'.format(Rvecs.shape[0], cell.precision, parmt.store + '/R_vectors.npy'))
    return Rvecs

def make_kpts(cell: pbcgto.cell.Cell, with_gamma: bool = True) -> pyscf.pbc.lib.kpts.KPoints:
    """
    Function to get the grid in reciprocal unit cell given k_grid density in input_parameters.py.
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Returns:
        k_grid: np.ndarray of shape (N, 3), k vectors generated from the cell.
    """
    k_grid = parmt.ik_grid
    kpts = cell.make_kpts(k_grid, wrap_around=True, with_gamma_point=with_gamma, space_group_symmetry=True)
    np.save(parmt.store + '/k-pts_i', kpts.kpts)
    logging.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_i.npy', k_grid[0], k_grid[1], k_grid[2]))
    return kpts

@time_wrapper
def gen_all_atomic_orbitals(cell: pbcgto.cell.Cell, primgauss: np.ndarray) -> list[cartmoments.AO]:
    """
    Given a cell, generate all atomic orbitals in the cell. Note that we, as usual assume Cartestian Gaussians.
    Inputs:
        cell:       pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
        primgauss:  np.ndarray of shape (N,7), returned from function gen_all_1D_prim_gauss.
    Returns:
        all_ao:     list of all atomic orbitals
    """
    i_cart, all_ao = 0, []
    cart_labs = cell.cart_labels()
    for i in range(cell.nbas):
        atm_indx=cell._bas[i][0]
        exps = cell.bas_exp(i)
        coeffs = cell.bas_ctr_coeff(i)
        ncgto = coeffs.shape[1]
        for j in range(ncgto):
            for _ in range(cell.bas_len_cart(i)):
                ijk = (cart_labs[i_cart].count('x'), cart_labs[i_cart].count('y'), cart_labs[i_cart].count('z'))
                all_ao.append(cartmoments.AO(atom_index = atm_indx, exps = exps, coeffs = coeffs[:,j], ijk = ijk, primgauss = primgauss))
                i_cart += 1
    logging.info("Generated all cartesian contracted gaussians, number of shells = {}.".format(len(all_ao)))
    return all_ao

@time_wrapper
def get_basis_blocks(aos: list[cartmoments.AO]) -> dict:
    """
    Get a dictionary object of all blocks of AO in terms of their prim_indices.
    Inputs:
        aos:    list[cartesian_moments.AO]
    Returns:
        blocks: dict object containing other dictionaries
                blocks[tuple][index of AO object]: np.ndarray containing coef*norm
    """
    blocks = {}
    for i, ao in enumerate(aos):
        pg = tuple(tuple(pi) for pi in ao.prim_indices)
        if pg in blocks:
            blocks[pg][i] = ao.coef
        else:
            blocks[pg] = {i: ao.coef}
    logging.info('Generated all blocks in 1D primitive gaussians, number of blocks = {}.'.format(len(blocks)))
    return blocks

@time_wrapper
def KS_electronic_structure(cell: pbcgto.cell.Cell) -> pbcdft.krks_ksymm.KsymAdaptedKRKS:
    """
    Function to do density functional theory. Solves the RKS if only one kpoint in kgrid, otherwise 
    solves RKS at each k-point and constructs density matrix from integrating over 1BZ. 
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
        kpts:   np.ndarray consisting of k-points. 
                If None, kgrid is generated using parameters input in parmt.k_grid
    Returns:
        None
    Saves:
        molecular energies, molecular coefficients and molecular occupation numbers.
    """
    dft_path = parmt.store + '/DFT/'
    makedir(dft_path)
    logging.info('Initial state calculation:')
    kpts = make_kpts(cell, True)
    kmf = pbcdft.KRKS(cell, kpts).density_fit()
    kmf.xc = parmt.xcfunc
    kmf.kernel()
    if kmf.converged:
        np.save(dft_path + 'mo_en_i', kpts.transform_mo_energy(kmf.mo_energy))
        np.save(dft_path + 'mo_coeff_i', kpts.transform_mo_coeff(kmf.mo_coeff))
        np.save(dft_path + 'mo_occ_i', kpts.transform_mo_occ(kmf.mo_occ))
    else:
        raise ValueError('DFT not converged. Might need to orthogonalize basis before continuing (Not Implemented).')
    logging.info('Electronic structure converged, KS energy is {:.2f} Hartrees.\n\tDFT data is stored to {}.'.format(kmf.e_tot, dft_path))
    return kmf

@time_wrapper
def KS_non_self_consistent_field(kmf: pbcdft.krks_ksymm.KsymAdaptedKRKS) -> None:
    """
    Non self consistent field calculation for final states.
    Inputs:
        kmf:    pyscf.pbc.dft.krks_ksymm.KsymAdaptedKRKS object
    Return:
        None
    """
    logging.info("Final State Calculation:")
    dft_path = parmt.store + '/DFT/'
    #kpts = make_kpts(kmf.cell, False)
    k_grid = parmt.fk_grid
    kpts = kmf.cell.make_kpts(k_grid, space_group_symmetry=True, wrap_around = True, scaled_center = kmf.cell.get_scaled_kpts(np.ones(3)[None, :]*parmt.dq*.5/(3**.5)))
    np.save(parmt.store + '/k-pts_f', kpts.kpts)
    logging.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_f.npy', k_grid[0], k_grid[1], k_grid[2]))
    ek , ck = kmf.get_bands(kpts.kpts_ibz)
    ek = kpts.transform_mo_energy(ek)
    ck = kpts.transform_mo_coeff(ck)
    np.save(dft_path + 'mo_en_f', ek)
    np.save(dft_path + 'mo_coeff_f', ck)
    logging.info('Non self consistent field equations solved for final state k-points. Data is stored to {}.'.format(dft_path))
    return None

@time_wrapper
def gen_G_vectors(cell: pbcgto.cell.Cell) -> np.ndarray:
    """
    Generate G vectors given a cell.
    Inputs:
        cell:       pyscf.pbc.gto.cell.Cell object
    Returns:
        G_vectors:  np.ndarray object of shape (N, 3)
    """
    reciprocal_vectors = cell.reciprocal_vectors()
    norm = np.min(np.linalg.norm(reciprocal_vectors, axis = 1))
    n = int(2*(parmt.q_max + np.max(np.linalg.norm(reciprocal_vectors, axis = 1))*(3**.5))/norm)
    N_range = list(range(-n,n+1)) 
    triplets = list(itertools.product(N_range, repeat=3))
    mygrid = np.asarray(triplets)
    lattice = np.sum(mygrid[:,:,np.newaxis]*reciprocal_vectors[np.newaxis,:,:], axis = 1).astype('float')
    lattice = lattice[np.linalg.norm(lattice, axis=1)<=parmt.q_max + np.max(np.linalg.norm(reciprocal_vectors, axis = 1))*(3**.5)]
    sortindx = np.argsort(np.linalg.norm(lattice, axis=1))
    logging.info('Generated {} G vectors, with maximum q = {:.2f} atomic units.'.format(lattice.shape[0], parmt.q_max))
    lattice = lattice[sortindx]
    np.save(parmt.store + '/G_vectors', lattice)
    return lattice

@time_wrapper
def convert_to_eV_and_scissor(cell: pbcgto.cell.Cell) -> None:
    """
    Function converts energies from Ryd to eV - prints bandgap, if scissor - scissor corrects bandgap.
    and scissor corrects to given bandgap. Otherwise
    Inputs:
        cell:                               pyscf.pbc.gto.cell.Cell object
    Reads:
        parmt.scissor_bandgap:              float, scissor energies in eV
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    Writes:
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    """
    occ_orb = cell.tot_electrons()//2
    en_i = np.load(parmt.store + '/DFT/mo_en_i.npy')*alpha*alpha*me
    en_f = np.load(parmt.store + '/DFT/mo_en_f.npy')*alpha*alpha*me
    homo = max(en_i[:,:occ_orb].max(), en_f[:,:occ_orb].max())
    en_i, en_f = en_i - homo, en_f - homo
    homo = 0
    lumo = min(en_i[:,occ_orb:].min(), en_f[:,occ_orb:].min())
    logging.info("All energies converted to eV. Calculated Bandgap = {:.2f} eV.".format(lumo - homo))
    if parmt.scissor_bandgap is not None:
        if type(parmt.scissor_bandgap) != float:
            raise ValueError("Parameter scissor_bandgap in input_parameters.py must be either None or of type float.")
        correction = parmt.scissor_bandgap - lumo
        en_i[:,occ_orb:], en_f[:,occ_orb:] = en_i[:,occ_orb:] + correction, en_f[:,occ_orb:] + correction
        logging.info("Scissor Correction applied, new bandgap is {:.2f} eV.".format(parmt.scissor_bandgap))
    np.save(parmt.store + '/DFT/mo_en_i.npy', en_i)
    np.save(parmt.store + '/DFT/mo_en_f.npy', en_f)
    logging.info("Electronic structure energies updated in files.")
    return

def project_vectors_to_1BZ(G: np.ndarray, D: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Take in a list of vectors, with q.ndim = 3, and return their projection to 1BZ.
    Inputs:
        G:      np.ndarray, reciprocal vectors of the cell
        D:      np.ndarray, inverse of G (equivalent to cell.lattice_vectors().T/2/np.pi)
        q:      np.ndarray of shape (N, M, 3)
    Returns:
        q_1BZ:  np.ndarray of shape (N, M, 3)
    """
    if q.ndim != 3 or q.shape[2] != 3:
        raise Exception("Input to project_vectors_to_1BZ must be 3-dimensional, with q.shape[2] == 3.")
    q = np.transpose(np.tensordot(D, q, axes=(0,-1)), axes = (1, 2, 0)) + 0.5
    q = q%1 - 0.5
    return np.transpose(np.tensordot(G, q, axes=(0,-1)), axes = (1, 2, 0))

@time_wrapper
def get_1BZ_q_points(cell: pbcgto.cell.Cell) -> dict:
    """
    Function to read both initial and final state k-points, and make all q = k2 - k1.
    We then project all q to 1BZ and keep all the unique q-points. All unique q are stored 
    as a numpy array file, and the function returns a dictionary object described below. 
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object
    Returns:
        q_1BZ:  dict, keys are unique q vectors, and each key contains
                        the indices of [k2, k1] pair which lead to the unique q vector. 
    """
    G = cell.reciprocal_vectors()
    D = np.linalg.inv(G)
    k1 = np.load(parmt.store + '/k-pts_i.npy')
    k2 = np.load(parmt.store + '/k-pts_f.npy')
    allq = project_vectors_to_1BZ(G, D, k2[None,:,:] - k1[:,None,:])
    dic = {}
    for i, qa in enumerate(allq):
        for j, q in enumerate(qa):
            tup = tuple(np.round(q, 10))
            if tup in dic:
                dic[tup].append([i, j])
            else:
                dic[tup] = [[i, j]]
    qu = np.array(list(dic.keys()))
    np.save(parmt.store + '/unique_q', qu)
    logging.info("{} unique q-vectors found in 1BZ. Storing all unique q-vectors in {}/unique_q.npy.".format(qu.shape[0], parmt.store))
    return dic

@time_wrapper
def primgauss_1D_overlaps(dark_objects: dict) -> list[np.ndarray]:
    """
    Store all 1D primitive gaussians in files. 
    parmt.store
        primgauss_1d_integrals
            dim_i
                q+G.npy
    Each stored array has shape (N_Ru_i, N_primgauss, N_primgauss)

    Inputs:
        dark_objects:   dict: equivalent to a class object, except not self-referential.
    """
    def store_primgauss_1D(dim: int, qG: np.ndarray, results: np.ndarray, Ru: np.ndarray, norms: np.ndarray) -> np.ndarray:
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        makedir(dir)
        np.save(dir + 'Ru', Ru)
        results = np.transpose(results, axes = (3, 0, 1, 2))
        val = []
        for q, res in zip(qG, results):
            np.save(dir + '{:.5f}'.format(q), res)
            loc = np.where(np.abs(np.einsum('Rab,a,b->Rab', res, norms, norms, optimize = False)).max(axis = (1,2)) > parmt.precision)[0]
            val.append([q, min(np.abs(Ru[loc.min()]), np.abs(Ru[loc.max()]))])
        return np.array(val)
    def get_R_cutoffs(vals: np.ndarray) -> np.ndarray:
        uns = get_all_unique_nums_in_array(vals[:,1])
        v = []
        for u in uns:
            v.append([min(np.abs(vals[vals[:,1] == u, 0])), u*(1+parmt.precision)])
        v.reverse()
        return np.array(v)
    norms = dark_objects['primitive_gaussians'][:,-1]**(1./3.)
    primindices = dark_objects['primindices']
    atom_locs = dark_objects['atom_locs']
    q, G = np.load(parmt.store + '/unique_q.npy'), dark_objects['G_vectors']
    Rv = dark_objects['R_vectors']
    logging.info("Generating overlaps of 1D primitve gaussians.")
    makedir(parmt.store + '/primgauss_1d_integrals/')
    vals = []
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        for d in range(3):
            qu, Gu = get_all_unique_nums_in_array(q[:,d], round_to=10), get_all_unique_nums_in_array(G[:,d], round_to=10)
            qG = (qu[:, None] + Gu[None, :]).reshape((-1))
            qG = get_all_unique_nums_in_array(qG, round_to=10)
            qG = qG[np.abs(qG) <= parmt.q_max]
            Ru = get_all_unique_nums_in_array(Rv[:,d], round_to=10)
            logging.info('\tDimension = {}:\n\t\tNumber of unique q = {};\n\t\tNumber of unique R = {}.'.format(d, qG.size, Ru.size))
            res = p.map(partial(cartmoments.primgauss_1D_overlaps_uR, primindices = primindices, q = qG, atom_locs = atom_locs[:,d]), Ru)
            res = np.array(res)
            vals.append(store_primgauss_1D(d, qG, res, Ru, norms))
    vals = np.append(vals[0], np.append(vals[1], vals[2], axis = 0), axis = 0)
    logging.info("Generated overlaps of 1D primitive gaussians.")
    return get_R_cutoffs(vals)

def get_band_indices():
    """
    Outputs:
        num_all_occ:        int: total number of occupied (valence) bands obtained from DFT calculation
        num_occ_bands:      int: number of occupied (valence) bands to include in dielectric function calculation
        num_unocc_bands:    int: number of unoccupied (conduction) bands to include in dielectric function calculation
        f_i:                int: occupancy of initial states: f_{ik} factor assuming temperature = 0K
    """
    dft_path = parmt.store + '/DFT/'
    mo_occ_i = np.load(dft_path + 'mo_occ_i.npy')

    if not (mo_occ_i == mo_occ_i[0]).all():
        raise NotImplementedError('Occupancy of bands was found to vary with k. Partially filled bands may cause issues determining occupied and unoccupied states, especially since k_f is different from k_i. Check mo_occ_i.npy.')
    
    if mo_occ_i[0][0] != 2:
        raise NotImplementedError('Occupancy of filled bands is not 2. Spin-dependent DFT has not been implemented. Check mo_occ_i.npy if you expect filled bands to have an occupancy of 2.')
    
    num_bands = mo_occ_i.shape[1]
    num_all_val = sum(mo_occ_i[0] != 0) #total number of occupied bands from dft calculation

    if parmt.numval == 'all':
        num_val = num_all_val
    elif parmt.numval > num_all_val:
        raise Exception(f'The specified number of valence bands to include ({parmt.numval}) is larger than the number of valence bands obtained from the DFT calculation ({num_all_val}). Check input parameter "numval" and DFT output file mo_occ_i.npy.')
    else:
        num_val = parmt.numval
    
    if parmt.numcon == 'all':
        num_con = num_bands - num_all_val
    elif parmt.numcon > num_bands - num_all_val:
        raise Exception(f'The specified number of conduction bands to include ({parmt.numcon}) is larger than the number of conduction bands obtained from the DFT calculation ({num_bands - num_all_val}). To increase the number of conduction bands, consider using a larger basis set. Check input parameters "numcon" and "mybasis", and DFT output file mo_occ_i.npy.')
    else:
        num_con = parmt.numcon

    ivaltop = num_all_val-1 #index of the highest valence band included
    ivalbot = ivaltop-num_val+1 #index of the lowest valence band included
    iconbot = ivaltop+1 #index of the lowest conduction band included
    icontop = iconbot+num_con-1 #index of the highest conduction band included

    np.save(parmt.store + '/bands.npy', np.array([ivalbot, ivaltop, iconbot, icontop]))

def get_3D_overlaps_blocks(q, G, k2: np.ndarray, blocks: dict, N_AO: int, R_id: np.ndarray, unique_Ri: list[np.ndarray], mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray) -> np.ndarray:
    """
    Notes:
        - last two axes of result are transposed compared to testing script so we have eta_qG(k,i,j) to use in susceptibility function

    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Inputs:
        q:              np.ndarray of shape (3,): one q vector in 1BZ
        G:              np.ndarray of shape (3,): one G vector
        k2:             np.ndarray of shape (N_kpairs, 3):
        blocks:         dict: atomic orbital blocks generated by get_basis_blocks 
        N_AO:           int: number of atomic orbitals
        mo_coeff_i:     np.ndarray of shape (N_kpairs, N_AO, N_valbands)
        mo_coeff_f:     np.ndarray of shape (N_kpairs, N_AO, N_conbands)
        R_id:           np.ndarray of shape (3, N_R_vectors)
        unique_Ri:      np.ndarray of shape (3, N_R_unique
    Outputs:
        eta_qG:         np.ndarray of shape (N_kpairs, N_val_bands (i), N_cond_bands (j)): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    ints = []
    for d in range(3): #435us total
        ints.append(np.load('test_resources/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, (q+G)[d]))[:,None,:,:] * np.exp(-1.j*unique_Ri[d][:,None]*k2[None,:,d])[:,:,None,None])
    ovlp = np.empty((k2.shape[0], N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,:,:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)
            tot = np.ones((R_id.shape[1], k2.shape[0], p2.shape[1], p1.shape[1]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][:,:,p2[d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)
            for i in d1:
                for j in d2:
                    ovlp[:,i,j] = (tot@d1[i])@d2[j]
    return np.einsum('kia,kij,kjb->kba', mo_coeff_f_conj, ovlp, mo_coeff_i, optimize = True)

def get_R_cutoffs(aos, N_primgauss, p):
    """
    Notes:
        - Should precision p be the same as parmt.precision that is fed to pyscf?
        - Currently only implemented in one direction - do we need to worry about 3D? For anisotropic could implement cutoff in terms of number of R cells to include instead of R_cut

    To Do:
        - change load_unique_R and 3D_overlaps so that R_id is only generated for R vectors up to |R| = R_cut(|q+G|)

    Inputs:
        aos:            atomic orbitals
        N_primgauss:    number of primitive gaussians
        p:              precision
    Outputs:
        qG_mag:         magnitudes of q+G sorted from smallest to largest
        R_cut:          largest magnitude of R vector required to obtain precision 
    """
    d = 0

    #determine max normalizations from all AO 
    norm_max = np.zeros(N_primgauss)
    for ao in aos: 
        for i,m in enumerate(ao.prim_indices[d]):
            norm_max[m] = max(norm_max[m],(ao.norm[i])**(1/3)) #updates maximum norm
    norm_max = np.tensordot(norm_max, norm_max, axes=0) #(m,n)

    dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(d)
    Ru = np.load(dir+'Ru.npy')
    files = os.listdir(dir)
    files.remove('Ru.npy')
    qG_mag = np.zeros(len(files))
    R_cut = np.zeros(len(files))
    for i,f in enumerate(files): #(for q:)
        q_1d_integrals = np.load(dir+f)
        q_1d_integrals = norm_max[None,:,:]*q_1d_integrals #multiply by max normalizations
        R_cut[i] = np.max(np.abs(Ru[np.max(q_1d_integrals, axis=(1,2)) > p])) #compares maximum 1d integral for each R_i to precision
        
        qG_mag[i] =  np.abs(float(f[:-4]))
        s = np.argsort(qG_mag)
        qG_mag = qG_mag[s] #sorted magnitude of q+G 
        R_cut = R_cut[s] #corresponding R_cut

    return qG_mag, R_cut

@njit
def get_energy_diff(mo_en_i, mo_en_f, E):
    """
    Notes:
    - numba roughly halves runtime, still only a few ms
    To Do:
    - Account for Re = inf when the energy difference exactly = 0

    Computes contributions to Re(chi) and Im(chi) coming from the energy difference denominator in Eq. (). Imaginary contributions arise when the energy difference is less than parmt.dE. We use a triangle approximation of the dirac delta function with width determined by parmt.dE

    Inputs:
        mo_en_i:    np.ndarray of shape (N_kpairs, N_valbands): initial molecular orbital energies
        mo_en_f:    np.ndarray of shape (N_kpairs, N_conbands): final molecular orbital energies
        E:          np.ndarray of shape (N_E,): energy transferred to electron

    Outputs:
        re_delE:  np.ndarray of shape (k_pairs, i, j, E): Real part of the energy denominator
        im_delE:  np.ndarray of shape (k_pairs, i, j, E): Imaginary part of the denominator
    """
    re_delE = (E[None,None,None,:] - mo_en_f[:,None,:,None] + mo_en_i[:,:,None,None]) #(k_pair,i,j,E)

    im_delE = (np.abs(re_delE) < parmt.dE) * (1.0 - np.abs(re_delE)/parmt.dE)
    return re_delE**(-1), -np.pi*im_delE


def RPA_susceptibility(eta_qG, eta_qGp, re_delE, im_delE):
    """
    Calculates the Kohn-Sham susceptibility chi^0_{GGp}(q,E) for all E

    Inputs:
        eta_qG:     np.ndarray of shape (N_kpairs, N_valbands, N_conbands): all 3D overlaps for q+G generated by get_3D_overlaps
        eta_qGp:    np.ndarray: all 3D overlaps for q+Gp
        re_delE:  np.ndarray of shape (N_kpairs, N_valbands, N_conbands, N_energies): Real part of the energy denominator
        im_delE:  np.ndarray of shape (N_kpairs, N_valbands, N_conbands, N_energies): Imaginary part of the denominator

    Outputs:
        chi:        np.ndarray of shape (N_enegies,): Kohn-Sham susceptibility, Eq.() 
    """
    chi_num = 2*np.conjugate(eta_qGp)*eta_qG #(k_pair,i,j)
    chi = chi_num[:,:,:,None] * (re_delE + 1j*im_delE) #(k_pair,i,j,E)
    return chi.sum(axis=(0,1,2))

def RPA_dielectric(G, q, k_f, mo_coeff_i, mo_coeff_f_conj, re_delE, im_delE, R_id, unique_Ri, blocks, N_AO):
    """
    To Do:
    - Check what factors should multiply chi - need alpha and crystal volume

    Calculates epsilon_{GG}(q, E) for all energies E at single 1BZ q-vector + G-vector. Does not include local field effects.

    Inputs:
        G:          np.ndarray of shape (3,): G-vector
        q:          np.ndarray of shape (3,): 1BZ q-vector
        k_f:        np.ndarray of shape (N_kpairs, 3)
        mo_coeff_i: np.ndarray of shape (N_kpairs, N_AO, N_valbands)
        mo_coeff_f: np.ndarray of shape (N_kpairs, N_AO, N_conbands)
        re_delE:    np.ndarray of shape (N_kpairs, N_valbands, N_conbands, N_energies): Real part of the energy denominator
        im_delE:    np.ndarray of shape (N_kpairs, N_valbands, N_conbands, N_energies): Imaginary part of the denominator
        R_id:       np.ndarray of shape (3, N_R_vectors)
        R_unique:   list of unique R scalars in each dimension
        blocks:     dict: atomic orbital blocks generated by get_basis_blocks
        N_AO:       int: number of atomic orbitals
    Outputs:
        eps:        np.ndarray of shape (N_energies,): epsilon_{GG}(q,E) 
    """
    eta_qG = get_3D_overlaps_blocks(q, G, k_f, blocks, N_AO, R_id, unique_Ri, mo_coeff_i, mo_coeff_f_conj)

    eps = 1 - (4*np.pi)**2 / (np.dot(q,q)+np.dot(G,G)) * RPA_susceptibility(eta_qG, eta_qG, re_delE, im_delE)
    #logging.info('epsilon_GG(q, E) calculated for all G and E for 1BZ q vector {:.5f}. epsilon_q is {:.3f} MB in memory.'.format(q, sys.getsizeof(eps)/10**6))
    return eps

def RPA_dielectric_lfe(q, G_vectors, k_pairs, mo_en_i, mo_en_f, eta_q):
    """
    To Do:
    - Fix np.diag in last step
    - Update for multiprocessing: only take one G, Gp pair

    Calculates epsilon_{GG}(q, E) for all energies E and G-vectors G at single 1BZ q-vector. Includes local field effects by calculating eps_{GGp}(q,E), then inverting and taking diagonal elements. This cannot be done for large q_max: |q+G| < ~5ame advised.

    Inputs:
        q:          np.ndarray of shape (3,): 1BZ q-vector
        G_vectors:
        k_pairs:
        mo_en_i:
        mo_en_f:
        eta_q:      np.ndarray: all 3D overlaps for q generated by get_3D_overlaps
    Outputs:
        eps:        np.array of shape (N_energies,N_G_vectors): epsilon_{GG}(q,E) 
    """
    num_G = G_vectors.shape[0]
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    eps_matrix = np.zeros((E.shape[0], num_G, num_G))
    eps_lfe = np.zeros(E.shape[0], num_G) #(E,G)

    for G_id, G in enumerate(G_vectors):
        for Gp_id, Gp in enumerate(G_vectors):
            eps_matrix[:,G_id,Gp_id] = -(4*np.pi)**2 / np.linalg.norm(q+G) / np.linalg.norm(q+Gp) * RPA_susceptibility(E, G_id, G_id, k_pairs, mo_en_i, mo_en_f, eta_q)
    eps_matrix = eps_matrix + np.identity(num_G)
    eps_lfe = 1/np.diag(np.linalg.inv(eps_matrix)) #need to fix: np.diag does not work with non-2d matrix
    return eps_lfe

@time_wrapper
def initialize_RPA_dielectric(dark_objects, test=False):
    """
    Loads DFT parameters, selects epsilon routine (LFE vs non-LFE), and calculates binned RPA dielectric function, epsilon(q,E).
    """

    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[:,:,ivalbot:ivaltop+1]
    mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')[:,:,iconbot:icontop+1]
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop+1]
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop+1]
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    unique_q = dark_objects['unique_q']
    G_vectors = dark_objects['G_vectors']
    R_vectors = dark_objects['R_vectors']
    N_AO = len(dark_objects['aos'])
    blocks = dark_objects['blocks']

    R_id, unique_Ri = load_unique_R(R_vectors)

    #energy bins
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)

    #initialize spacial bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    tot_bin_eps = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1), dtype='complex')
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)
    
    if parmt.include_lfe:
        RPA_eps = RPA_dielectric_lfe
    else: #currently only implemented for non-LFE
        RPA_eps = RPA_dielectric

    if test:
        #for test case only run for one q vector and a few G vectors
        unique_q = {list(unique_q.keys())[0]: unique_q[list(unique_q.keys())[0]]}
        G_vectors = G_vectors[:1000]

    for q in unique_q.keys():
        k_pairs = np.array(unique_q[q])

        #Only include G_vectors such that |q + G| < q_max
        q = np.array(q)
        G_q = G_vectors[np.linalg.norm(q+G_vectors, axis=1) < parmt.q_max]

        #generate parameters for q
        mo_en_i_q = mo_en_i[k_pairs[:,0]] #(k_pair,i)
        mo_en_f_q = mo_en_f[k_pairs[:,1]] #(k_pair,j)
        mo_coeff_i_q = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
        mo_coeff_f_q_conj = mo_coeff_f[k_pairs[:,1]].conj() #(k_pair,b,j)
        k_f_q = k_f[k_pairs[:,1]]

        #Compute energy differences for denominator of Re(chi), and Im(chi)
        re_delE, im_delE = get_energy_diff(mo_en_i_q, mo_en_f_q, E)

        #Compute epsilon for all G_q
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:  #parallelization over G
            eps = p.map(partial(RPA_eps, q=q, k_f=k_f_q, mo_coeff_i=mo_coeff_i_q, mo_coeff_f_conj=mo_coeff_f_q_conj, re_delE=re_delE, im_delE=im_delE, R_id=R_id, unique_Ri=unique_Ri, blocks=blocks, N_AO=N_AO), G_q)
        eps = np.array(eps) #(G, E)

        #bin eps calculated for all G
        tot_bin_eps, tot_bin_weights = bin.bin_eps_q(q, G_q, eps, bin_centers, tot_bin_eps, tot_bin_weights)

    return tot_bin_eps, tot_bin_weights


def load_unique_R(R_vectors):
    """
    Loads unique R in each dimension from saved 1D intgrals and generates R_id which writes R_vectors in terms of unique R indices. These are used to generate 3D overlaps.
    """
    unique_Ri = []
    R_id = np.zeros_like(R_vectors)
    for dim in range(3):
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        unique_Ri.append(np.load(dir+'Ru.npy'))

        nR = np.round(R_vectors[:,dim,None] - unique_Ri[dim][None,:], 10)
        R_id[:,dim] = np.sum(nR > 0, axis=1)
    R_id = np.transpose(R_id).astype(np.int16) #(dim, R_vec) 
    return R_id, unique_Ri

def get_binned_epsilon(tot_bin_eps, tot_bin_weights):
    """
    Notes: 
    - What to do with nans that occur when eps = weight = 0?

    After all epsilon(q+G), return epsilon(bins)

    Inputs:
        tot_bin_eps: (N_energies, N_bins)
        tot_bin_weights: (N_bins)
    Outputs:
        binned_eps: (N_energies, N_bins)
    """
    binned_eps = tot_bin_eps/tot_bin_weights[:,None]
    #remove nans?
    return(binned_eps)
