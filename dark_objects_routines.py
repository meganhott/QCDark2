import psutil
import itertools
import numpy as np
import multiprocessing as mp
from functools import partial
import pyscf.pbc.gto as pbcgto

from routines import logger, time_wrapper, makedir, load_unique_R, get_all_unique_nums_in_array
import input_parameters as parmt
import cartesian_moments as cartmoments

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

    logger.info("Built cell object. Note that we only use cartesian gaussians.\nWe have set maximum memory available to pyscf = 90% of available memory of system = {:.2f} MB".format(max_memory))
    logger.info("Parameters fed, in atomic units:")

    logger.info("\tLattice vectors:")
    for a in cell.lattice_vectors():
        logger.info("\t\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1], a[2]))

    logger.info("\tReciprocal Vectors:")
    for a in cell.reciprocal_vectors():
        logger.info("\t\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1], a[2]))

    logger.info("\tAtom locations:")
    for a in cell._atom:
        logger.info("\t\t{}:\t{:.5f}\t{:.5f}\t{:.5f}".format(a[0], a[1][0], a[1][1], a[1][2]))
    
    logger.info("\tBasis set: ")
    for a in cell.basis:
        logger.info("\t\t{}: {}".format(a, cell.basis[a]))
    
    logger.info("\tEffective core potential:")
    if len(cell.ecp):
        for a in cell.ecp:
            logger.info("\t\t{}: {}".format(a, cell.ecp[a]))
    else:
        logger.info("\t\tNone, all-electron calculation.")
    
    logger.info("\tSelected precision: {}".format(cell.precision))
    logger.info("Further information is in {}.".format(cell.output))

    makedir(parmt.store, log = False)

    return cell

@time_wrapper
def construct_R_vectors(cell: pbcgto.cell.Cell) -> np.ndarray:
    """
    Function to construct all R vectors {R_i} relevant to the cell. We use pyscf build in methods and sort the R vectors from there.
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
    logger.info('{} R vectors generated for the cell given precision = {}, and saved to {}.'.format(Rvecs.shape[0], cell.precision, parmt.store + '/R_vectors.npy'))
    return Rvecs

def get_cell_volume(cell: pbcgto.cell.Cell) -> float:
    """
    Returns volume of unit cell in units of Bohr^3.
    """
    R_lat = cell.lattice_vectors()
    V_cell = np.abs(np.dot(R_lat[0], np.cross(R_lat[1], R_lat[2])))
    return V_cell

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
    logger.info('Generated {} G vectors, with maximum q = {:.2f} atomic units.'.format(lattice.shape[0], parmt.q_max))
    lattice = lattice[sortindx]
    np.save(parmt.store + '/G_vectors', lattice)
    return lattice


@time_wrapper
def gen_all_1D_prim_gauss(cell: pbcgto.cell.Cell) -> np.ndarray:
    """
    Generates all primitive gaussians and their 1D components, and places them into an output np.ndarray. Skeleton of the dielectric function procedure, links overlaps calculated in primgauss_1D_overlaps with atomic orbital calculated.
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
        Routine to normalize the basis functions. Note that s and p orbitals have different normalizations for cartesian and spherical shells, while higher orbitals carry same normalization scheme no matter what.
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
    logger.info("All 1D primitive gaussians found for the cell.\n\tNumber of unique primitive gaussians = {}.\n\tThis includes all possible angular momentum in one direction.".format(primgauss.shape[0]))
    return primgauss

@time_wrapper
def gen_prim_gauss_indices(primgauss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function returns index in array containing atom_id, l, exponent, and initial index (corresponding to i = 0) in primgauss object constructed from function gen_all_1D_primgauss.This function generates the objects to be passed into the primgauss_1D_overlaps for generating overlaps.
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

    logger.info("Constructed unique exponent for each atom. Total number of exponents at different locations: {}".format(indx_arr.shape[0]))
    return indx_arr, np.array(locs)

@time_wrapper
def gen_all_atomic_orbitals(cell: pbcgto.cell.Cell, primgauss: np.ndarray) -> list[cartmoments.AO]:
    """
    Given a cell, generate all atomic orbitals in the cell. Note that we, as usual assume Cartesian Gaussians.
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
    logger.info("Generated all cartesian contracted gaussians, number of shells = {}.".format(len(all_ao)))
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
    logger.info('Generated all blocks in 1D primitive gaussians, number of blocks = {}.'.format(len(blocks)))
    return blocks

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
    Function to read both initial and final state k-points, and make all q = k2 - k1. We then project all q to 1BZ and keep all the unique q-points. All unique q are stored as a numpy array file, and the function returns a dictionary object described below. 
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
    logger.info("{} unique q-vectors found in 1BZ. Storing all unique q-vectors in {}/unique_q.npy.".format(qu.shape[0], parmt.store))
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
    logger.info("Generating overlaps of 1D primitve gaussians.")
    makedir(parmt.store + '/primgauss_1d_integrals/')
    vals = []
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        for d in range(3):
            qu, Gu = get_all_unique_nums_in_array(q[:,d], round_to=10), get_all_unique_nums_in_array(G[:,d], round_to=10)
            qG = (qu[:, None] + Gu[None, :]).reshape((-1))
            qG = get_all_unique_nums_in_array(qG, round_to=10)
            qG = qG[np.abs(qG) <= parmt.q_max]
            Ru = get_all_unique_nums_in_array(Rv[:,d], round_to=10)
            logger.info('\tDimension = {}:\n\t\tNumber of unique q = {};\n\t\tNumber of unique R = {}.'.format(d, qG.size, Ru.size))
            res = p.map(partial(cartmoments.primgauss_1D_overlaps_uR, primindices = primindices, q = qG, atom_locs = atom_locs[:,d]), Ru)
            res = np.array(res)
            vals.append(store_primgauss_1D(d, qG, res, Ru, norms))
    vals = np.append(vals[0], np.append(vals[1], vals[2], axis = 0), axis = 0)
    logger.info("Generated overlaps of 1D primitive gaussians.")
    return get_R_cutoffs(vals)

def get_R_id(R_vectors, unique_Ri):
    """
    Generates R_id which writes R_vectors in terms of unique R indices.
    """
    R_id = np.zeros_like(R_vectors)
    for d in range(3):
        nR = np.round(R_vectors[:,d,None] - unique_Ri[d][None,:], 10)
        R_id[:,d] = np.sum(nR > 0, axis=1)
    R_id = np.transpose(R_id).astype(int) #(dim, R_vec) 
    return R_id

@time_wrapper
def store_R_ids(dark_objects: dict):
    """
    Store R_ids for each R_cutoff.
    Inputs:
        Rcutoffs:       np.ndarray of dim = 2, shape (N, 2)
        dark_objects:   dict
    Returns:
        q_cuts:         np.ndarray of dim = 1, shape (N, )
    """
    makedir(parmt.store + '/R_ids')
    R_vecs = dark_objects['R_vectors']
    R_cutoffs = dark_objects['R_cutoffs']
    abs_R = np.linalg.norm(R_vecs, axis = 1)
    unique_Ri = load_unique_R()
    R_id = get_R_id(R_vecs, unique_Ri)
    np.save(parmt.store + '/R_ids/{}'.format(-1), R_id)
    q_cuts = []
    for i, R_cut in enumerate(R_cutoffs):
        q_cuts.append(R_cut[0])
        tR = R_vecs[abs_R <= R_cut[1]]
        R_id = get_R_id(tR, unique_Ri)
        np.save(parmt.store + '/R_ids/{}'.format(i), R_id)
    return np.array(q_cuts)