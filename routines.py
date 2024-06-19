#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tues Apr 09 2024, 14:00:00
Reimagined routines with emphasis on 1-D calculation of primitive gaussian overlaps.
Authors: Megan Hott, Aman Singal
"""

import os, time, itertools, logging, pdb, struct, sys, h5py, functools, psutil
import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.gw as pbcgw
import pyscf.pbc
import cartesian_moments as cartmoments
from multiprocessing import Pool
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
            patchname + " not applied, not an applicable Python version: %s.\n",
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
        logging.info(patchname + " already applied, skipping.\n")
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
            logging.info('Made directory \'' + dirname + '\'.\n')
    except FileExistsError:
        if os.path.isdir(dirname):
            if log:
                logging.info('Directory \''+dirname+'\' already exists.\n')
        else:
            logging.info('File exists with name same as directory, \'' + dirname + '\'. Cannot make directory, raising exception.\n')
            raise FileExistsError('Cannot proceed with making directory \'' + dirname + '\', file exists with same name.')
    return None

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
    logging.info("Further information is in {}.\n".format(cell.output))

    makedir(parmt.store, log = True)

    return cell

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
    logging.info("All 1D primitive gaussians found for the cell.\n\tNumber of unique primitive gaussians = {}.\n\tThis includes all possible angular momentum in one direction.\n".format(primgauss.shape[0]))
    return primgauss

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

    logging.info("Constructed unique exponent for each atom. Total number of exponents at different locations: {}\n".format(indx_arr.shape[0]))
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
        logging.info("Number of unique elements found for {} = {}.\n".format(log_name, unq.size))
    return unq

def get_all_unique_vectors_in_array(array: np.ndarray, round_to: int = None) -> np.ndarray:
    """
    Finds all unique vectors in an (n, 3) array of vectors 
    """
    if not round_to is None:
        array = np.round(array, round_to)
    unq = np.unique(array, axis=0)
    return unq

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
    unR = get_all_unique_nums_in_array(Rvecs, round_to = 9, log_name = None)
    logging.info("\tNumber of unique scalars in R vectors, i.e., unique R_i for R = (R_1, R_2, R_3) = {}.\n".format(unR.size))
    return Rvecs, unR

def make_kpts(cell: pbcgto.cell.Cell) -> pyscf.pbc.lib.kpts.KPoints:
    """
    Function to get the grid in reciprocal unit cell given k_grid density in input_parameters.py.
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Returns:
        k_grid: np.ndarray of shape (N, 3), k vectors generated from the cell.
    """
    kpts = cell.make_kpts(parmt.k_grid, wrap_around=True, with_gamma_point=True, space_group_symmetry=True)
    np.save(parmt.store + '/k-pts.npy', kpts.kpts)
    logging.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.\n".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k_grid.npy', parmt.k_grid[0], parmt.k_grid[1], parmt.k_grid[2]))
    return kpts

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
                all_ao.append(cartmoments.AO(atom_index = atm_indx, exps = exps, coeffs = coeffs, ijk = ijk, primgauss = primgauss))
                i_cart += 1
    logging.info("Generated all cartesian contracted gaussians, number of shells = {}.\n".format(len(all_ao)))
    return all_ao

def KS_density_functional_theory(cell: pbcgto.cell.Cell, kpts: pyscf.pbc.lib.kpts.KPoints = None) -> pbcdft.krks_ksymm.KsymAdaptedKRKS:
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
    dft_path = parmt.store + '/DFT'
    makedir(dft_path)
    if kpts is None:
        kpts = make_kpts(cell)
    kmf = pbcdft.KRKS(cell, kpts).density_fit()
    kmf.xc = parmt.xcfunc
    kmf.kernel()
    if kmf.converged:
        np.save(dft_path + '/mo_en.npy', kpts.transform_mo_energy(kmf.mo_energy))
        np.save(dft_path + '/mo_coeff.npy', kpts.transform_mo_coeff(kmf.mo_coeff))
        np.save(dft_path + '/mo_occ.npy', kpts.transform_mo_occ(kmf.mo_occ))
    else:
        raise ValueError('DFT not converged. Might need to orthogonalize basis before continuing (Not Implemented).')
    logging.info('Electronic structure converged, KS energy is {:.2f} Hartrees.\n\tDFT data is stored to {}/'.format(kmf.e_tot, dft_path))
    return kmf

def KS_NSCF(kmf: pbcdft.krks_ksymm.KsymAdaptedKRKS) -> None:
    """
    Perform an NSCF Calculation on all points (k+q)%G and store N_SCF results to get |j \vec{k}+\vec{q}> and E_{j, \vec{k}+\vec{q}}.
    """
    dft_path = parmt.store + '/DFT/'
    k_nscf = kmf.cell.make_kpts(parmt.nscf_grid, wrap_around=True, with_gamma_point=False, space_group_symmetry=False)
    e_k, c_k = kmf.get_bands(k_nscf)
    np.save(dft_path + 'nscf_ek', e_k)
    np.save(dft_path + 'nscf_ck', c_k)
    logging.info("NSCF Calculation for {} (k+q) points completed. Data stored to {}.".format(k_nscf.size//3, dft_path))
    return

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
    n = int(2*parmt.q_max/norm)
    N_range = list(range(-n,n+1)) 
    triplets = list(itertools.product(N_range, repeat=3))
    mygrid = np.asarray(triplets)
    lattice = np.sum(mygrid[:,:,np.newaxis]*reciprocal_vectors[np.newaxis,:,:], axis = 1).astype('float')
    lattice = lattice[np.linalg.norm(lattice, axis=1)<=parmt.q_max]
    sortindx = np.argsort(np.linalg.norm(lattice, axis=1))
    logging.info('Generated {} G vectors, with maximum q = {:.2f} atomic units.\n'.format(lattice.shape[0], parmt.q_max))
    return lattice[sortindx]