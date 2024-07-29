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
import cartesian_moments as cartmoments
import multiprocessing as mp
from functools import partial
import shutil
import input_parameters as parmt
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
    """
    primgauss = np.zeros((0, 7))
    for atom_id, atom in enumerate(cell._atom):
        species, loc = atom[0], atom[1]
        atom_basis = cell._basis[species]
        for basis_element in atom_basis:
            l = basis_element[0]
            elements = basis_element[1:]
            for element in elements:
                for i in range(l+1):
                    primgauss = np.append(primgauss, [[atom_id, l, i, element[0], loc[0], loc[1], loc[2]]], axis = 0)
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
def construct_R_vectors(cell: pbcgto.cell.Cell) -> tuple[np.ndarray, np.ndarray]:
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
    logging.info("{} unique q-vectors found in 1BZ. Storing all unique q-vectors in {} + /unique_q.npy.".format(qu.shape[0], parmt.store))
    return dic

def store_primgauss_1D(dim: int, qG: np.ndarray, results: np.ndarray, Ru: np.ndarray):
    dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
    makedir(dir)
    np.save(dir + 'Ru', Ru)
    results = np.transpose(results, axes = (3, 0, 1, 2))
    for q, res in zip(qG, results):
        np.save(dir + '{:.5f}'.format(q), res)
    return None

@time_wrapper
def primgauss_1D_overlaps(dark_objects: dict):
    """
    Store all 1D primitive gaussians in files. 
    NOTE: Currently returns one numpy array for all overlaps. This is not good and will not work for anisotropic systems.
    NOTE: Implement file stores with name parmt.store + '/1d_overlaps/{}_{}.npy'.format(d, q) 
    Inputs:
        dark_objects:   dict: equivalent to a class object, except not self-referential.
    """
    primindices = dark_objects['primindices']
    atom_locs = dark_objects['atom_locs']
    q, G = np.load(parmt.store + '/unique_q.npy'), dark_objects['G_vectors']
    Rv = dark_objects['R_vectors']
    logging.info("Generating overlaps of 1D primitve gaussians.")
    makedir(parmt.store + '/primgauss_1d_integrals/')
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
            store_primgauss_1D(d, qG, res, Ru)
    logging.info("Generated overlaps of 1D primitive gaussians.")
    return 

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

def dirac_delta(E_minus_delE):
    """
    Triangle approximation of dirac delta function used to calculate imaginary part of dielectric function. The width is determined by parmt.dE.

    Inputs:
        E_minus_delE:   np.ndarray of shape (N_energies,N_k_pairs,N_val_bands,N_con_bands): E - delE = E - (E_{jk'} - E_{ik})
    Outputs:
        d:  np.ndarray of shape (N_energies,N_k_pairs,N_val_bands,N_con_bands): numerical approximation of delta function 
    """

    """
    if delE <= E:
        d = (delE - (E-parmt.dE))/parmt.dE
    else: #delE > E
        d = (E + parmt.dE - delE)/parmt.dE
    """
    #Mask for for 0 <= E - delE < parmt.dE
    left_delta = ((E_minus_delE >= 0) & (E_minus_delE < parmt.dE))*(-E_minus_delE + parmt.dE)/parmt.dE 

    #Mask for 0 > E - delE > -parmt.dE
    right_delta = ((E_minus_delE < 0) & (E_minus_delE > -parmt.dE))*(E_minus_delE + parmt.dE)/parmt.dE

    d = left_delta + right_delta
    
    return d

@time_wrapper
def get_3D_overlaps(q, k_pairs, k_f, mo_coeff_i, mo_coeff_f, dark_objects):
    """
    Work in progress
    - Look into einsum optimizations
    - Implement multiprocessing over G?

    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector using stored 1D overlaps

    Inputs:
        q:              np.ndarray of shape (3,): one q vector in 1BZ
        dark_objects:   dict
        k_pairs:        np.ndarray of shape (N_kpairs, 2): all (k1, k2) pairs such that k2 - k1 = q
        mo_coeff_i:     np.ndarray of shape (N_k1, N_val_bands, N_AO)
        mo_coeff_f:     np.ndarray of shape (N_k2, N_con_bands, N_AO)
        k_f:            np.ndarray of shape (N_kf, 3): 
    Outputs:
        eta_q:          np.ndarray of shape (N_G, N_kpairs, N_val_bands, N_con_bands): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    ao = dark_objects['all_ao']

    #Create 0 padded ao_norm*ao_coef
    for ao_i in ao:
        ao_all_coef = np.zeros((3,dark_objects['primitive_gaussians'].shape[0]))
        for dim in range(3):
            ao_all_coef[dim,ao_i.prim_indices[dim]] = ao_i.norm*ao_i.coef
        ao_i.all_coef = ao_all_coef #(3,N_tot_primgauss)

    unique_Ri = []
    R_id = np.zeros_like(dark_objects['R_vectors'])
    for dim in range(3):
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        unique_Ri.append(np.load(dir+'Ru.npy'))

        nR = dark_objects['R_vectors'][:,dim,None] - unique_Ri[dim][None,:]
        R_id[:,dim] = np.sum(nR > 0, axis=1) 

    eta_q = np.zeros((dark_objects['G_vectors'].shape[0],k_pairs.shape[0],mo_coeff_i.shape[1],mo_coeff_f.shape[1]))

    for G_id,G in enumerate(dark_objects['G_vectors']): #multiprocessing for this step
        #load in relevant q+G 1D overlaps
        qG = q + G
        Ri_coef_sum = np.zeros_like(R_id)
        for dim in range(3):
            dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
            q_1d_integrals = np.load(dir+'{:.5f}.npy'.format(qG[dim]))

            coef_sum = np.einsum('ij,kl,mn,iln,jok,jpm->ijop',np.exp(-1j*np.tensordot(unique_Ri[dim],k_f[k_pairs[:,1],dim],0)), np.array([ao_i.all_coef[dim] for ao_i in ao]), np.array([ao_i.all_coef[dim] for ao_i in ao]), q_1d_integrals, mo_coeff_i[k_pairs[:,0],:,:], np.conjugate(mo_coeff_f[k_pairs[:,1],:,:])) #(i,j,k,l,m,n,o,p)=(Rd,k_pair,a,m,b,n,i,j) -> (i,j,o,p)=(Rd,k_pair,i,j)
            #possible einsum optimization with optimize=...

            Ri_coef_sum[:,dim] = coef_sum[R_id[:,dim],:,:,:] #all elements for R_id in one dimension
        
        #product of x,y,z terms
        eta_q[G_id,:,:,:] = np.sum(np.prod(Ri_coef_sum, axis=1),axis=0) #(G_id,k_pair,i,j)

        logging.info('All {} 3D overlaps generated for 1BZ q vector {:.5f}. eta_q is {:.3f} MB in memory.'.format(np.prod(eta_q.shape), q, sys.getsizeof(eta_q)/10**6))
    return eta_q


def RPA_susceptibility(E, G_id, Gp_id, k_pairs, mo_en_i, mo_en_f, eta_q):
    """
    To Do:
    - Deal with nan and inf entries in chi_re

    Calculates the Kohn-Sham susceptibility chi^0_{GGp}(q,E) for all E

    Inputs:
        E:          np.ndarray of shape (N_energies,): all energies
        G_id:       int: index of G vector from G_vectors
        Gp_id:      int: index of G' vector from G_vectors
        k_pairs:    np.ndarray of shape (N_k_pairs, 2): (ki_id, kf_id) pairs
        mo_en_i:    np.ndarray of shape (N_ki, N_val_bands, N_AO): initial molecular orbital energies
        mo_en_f:    np.ndarray of shape (N_kf, N_con_bands, N_AO): final molecular orbital energies
        eta_q:      np.ndarray: all 3D overlaps for q generated by get_3D_overlaps

    Outputs:
        chi:        np.ndarray of shape (N_enegies,): Kohn-Sham susceptibility, Eq.() 
    """
    delE = mo_en_f[k_pairs[:,1],None,:] - mo_en_i[k_pairs[:,0],:,None] #(k_pair,i,j)
    chi_num = 2*np.conjugate(eta_q[G_id,:,:,:])*eta_q[Gp_id,:,:,:] #(k_pair,i,j)

    E_minus_delE = np.transpose(-delE[:,:,:,None] + E[None,:], axes=(3,0,1,2))
    chi_re = chi_num / E_minus_delE
    #change any nan (0/0) and inf (c/0) entries to 0

    chi_im = -np.pi*dirac_delta(E_minus_delE)*chi_num

    chi = chi_re + 1j*chi_im #(E,k_pair,i,j)

    chi = np.einsum('ijkl->i',chi) #sum over k_pairs, i, j
    return chi

@time_wrapper
def RPA_dielectric(q, G_vectors, k_pairs, mo_en_i, mo_en_f, eta_q):
    """
    To Do:
    - Need to bin q+G as we go instead of storing in eps dict: dict for all q and E gets > 1TB if all are calculated first
    - Implement multiprocessing over G. After each q is completed, run the binning routine to reduce number of stored values from #G -> #bins.

    Calculates epsilon_{GG}(q, E) for all energies E and G-vectors G at single 1BZ q-vector. Does not include local field effects.

    Inputs:
        q:          np.ndarray of shape (3,): 1BZ q-vector
        G_vectors:
        k_pairs:
        mo_en_i:
        mo_en_f:
        eta_q:      np.ndarray: all 3D overlaps for q generated by get_3D_overlaps
    Outputs:
        eps:        np.ndarray of shape (N_energies,N_G_vectors): epsilon_{GG}(q,E) 
    """
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    eps = np.zeros(E.shape[0], G_vectors.shape[0]) #(E,G)

    for G_id, G in enumerate(G_vectors):
        eps[:,G_id] = 1 - (4*np.pi)**2 / (np.dot(q,q)+np.dot(G,G)) * RPA_susceptibility(E, G_id, G_id, k_pairs, mo_en_i, mo_en_f, eta_q)
    logging.info('epsilon_GG(q, E) calculated for all G and E for 1BZ q vector {:.5f}. epsilon_q is {:.3f} MB in memory.'.format(q, sys.getsizeof(eps)/10**6))
    return eps

def RPA_dielectric_lfe(q, G_vectors, k_pairs, mo_en_i, mo_en_f, eta_q):
    """
    To Do:
    - Fix np.diag in last step

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

def initialize_RPA_dielectric(dark_objects):
    """
    Loads DFT parameters, selects epsilon routine (LFE vs non-LFE), and calculates binned RPA dielectric function, epsilon(q,E).
    """

    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[:,ivalbot:ivaltop,:]
    mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')[:,iconbot:icontop,:]
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop]
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop]
    k2 = np.load(parmt.store + '/k-pts_f.npy')

    unique_q = dark_objects['unique_q']
    G_vectors = dark_objects['G_vectors']

    #initialize bins
    
    if parmt.include_lfe:
        RPA_eps = RPA_dielectric
    else:
        RPA_eps = RPA_dielectric_lfe

    for q in unique_q.keys():
        k_pairs = np.array(unique_q[q])

        eta_q = get_3D_overlaps(q, k_pairs, k2, mo_coeff_i, mo_coeff_f, dark_objects) #all 3D overlap integrals #(G_id,k_pair,i,j)
        
        eps = RPA_eps(np.array(q), G_vectors, k_pairs, mo_en_i, mo_en_f, eta_q) #(nE,G_id)
        #still need to implement binning at end of RPA_eps function


def cartesian_to_spherical(cart: np.ndarray) -> np.ndarray:
    """
    Converts all vectors given in cartesian coordinates to corresponding vectors in spherical polar coordinates,
    and remove non-unique vectors.
    Inputs:
        cart:   np.ndarray of shape (N, 3)
                cart[i] = [x, y, z]
    Outputs:
        sph:    np.ndarray of shape (N, 3)
                sph[i] = [r, theta, phi]
                0 <= theta <= pi, -pi <= phi < pi 
    """
    r = np.sqrt(cart[:,0]**2 + cart[:,1]**2 + cart[:,2]**2)
    theta = np.arccos(cart[:,2]/r)
    phi = np.arctan2(cart[:,1],cart[:,0])
    for i,th in enumerate(theta):
        if th == 0 or round(th, 9) == round(np.pi, 9):
            phi[i] = 0
        if round(phi[i],9) == round(np.pi, 9): 
            phi[i] = -np.pi
    return np.transpose([r, theta, phi])

def bin_q(q, G_vectors, bin_centers):
    """
    Bin centers currently generated in tests
    """
    #convert q+G to spherical coords
    qG_sh = cartesian_to_spherical(q + G_vectors)
    #determine closest r bin centers
    np.round(qG_sh[:,0]/parmt.dq)
    r_l = (np.round(qG_sh[:,0]/parmt.dq) - 0.5)*parmt.dq
    r_g = (np.round(qG_sh[:,0]/parmt.dq) + 0.5)*parmt.dq

    #determine closest solid angles 

    #match to bins
    #record list of all bins q+G contributes to with weights
