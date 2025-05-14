import numpy as np
import time
import shutil # for removing working directories after calculation is finished
import multiprocessing as mp
from functools import partial
from numba import njit
import h5py
from scipy.interpolate import LinearNDInterpolator

from dielectric_pyscf.routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import dielectric_pyscf.input_parameters as parmt
import dielectric_pyscf.epsilon_helper as eps
import dielectric_pyscf.binning as bin

def get_RPA_dielectric(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=None):
    if parmt.include_lfe:
        tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_LFE(dark_objects, rank, q_start, q_stop)
    else:
        if parmt.alt_binning: #Temporary
            print('Alternate binning is not updated for general Kramers-Kronig transform')
            exit()
            get_RPA_dielectric_no_LFE_alt_binning(dark_objects)
        else:
            tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_no_LFE(dark_objects, rank, q_start, q_stop)

    return tot_bin_eps, tot_bin_weights, bin_centers

@time_wrapper
def get_RPA_dielectric_no_LFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=None):
    # Reading all relevant data
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']

    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q
    
    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)

        if rank == 0 or rank == None:
            logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {np.array2string(q, precision=5)},')

        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]

        if rank == 0 or rank == None:
            logger.info(f'\t\tnumber of G vectors = {len(G_q)},')

        # Imaginary part of polarizability for E >= 0
        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

        # Imaginary part of polarizability for E < 0 to use in KK
        eps_q_im_neg = -1*get_RPA_dielectric_no_LFE_q(q, mo_en_f[:,ivalbot:ivaltop+1], mo_en_i[:,iconbot:icontop+1], mo_coeff_f_conj[:,:,ivalbot:ivaltop+1], mo_coeff_i[:,:,iconbot:icontop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)
        
        # Flip order of eps_q_im_neg and concatenate along energy axis
        eps_q_im_tot = np.concatenate((np.flip(eps_q_im_neg[:,1:], axis=1), eps_q_im), axis=1)
        # Remove E=0 bin from eps_q_im_neg? - maybe not - add to E=0 bin of eps_q_im?

        if rank == None or rank == 0:
            logger.info(f'\t\t\tIm(eps) calculated for all k and G for energies -E_max <= E <= E_max. Time taken = {(time.time() - start_time):.2f} s.')

        start_time1 = time.time()
        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im_tot, bin_centers, tot_bin_eps_im, tot_bin_weights)

        if rank == 0 or rank == None:
            logger.info(f'\t\t\tIm(eps) binned. Time taken = {(time.time() - start_time1):.2f} s.')
            logger.info(f'\t\tcomplete. Time taken = {(time.time() - start_time):.2f} s.')

    # Removing extra bins
    tot_bin_eps_im = tot_bin_eps_im[:-N_ang_bins, :]
    tot_bin_weights = tot_bin_weights[:-N_ang_bins]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    shutil.rmtree(working_dir) # delete working directory

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

def save_eps(bin_eps, bin_weights, bin_centers):
    f = h5py.File(parmt.store + '/epsilon.hdf5', 'r+') # open hdf5 file

    binned_eps = bin_eps/bin_weights[:, None]
    #f.create_dataset('binned_epsilon', data=binned_eps) # No KK or interpolation - should save only 0 + iIm(eps) for non-LFE

    binned_eps_im = np.imag(binned_eps)
    #binned_eps_kk = eps.kramerskronig_im2re(binned_eps_im) + 1. + 1j*binned_eps_im
    #f.create_dataset('binned_epsilon_kk', data=binned_eps_kk) # Only KK, no interpolation

    #binned_eps_kk_interp = interp_eps(bin_centers, binned_eps)
    #f.create_dataset('binned_epsilon_kk_interp', data=binned_eps_kk_interp) # KK then Interpolation

    binned_eps_im_interp = interp_eps(bin_centers, binned_eps_im)
    binned_eps_interp = eps.kramerskronig_im2re(binned_eps_im_interp) + 1. + 1j*binned_eps_im_interp
    #f.create_dataset('binned_epsilon_interp_kk', data=binned_eps_interp) # Interpolation and then KK # This slightly reduces noise in ELF compared to KK then interpolation

    eps_r = epsilon_r(bin_centers, binned_eps_interp)
    f.create_dataset('epsilon_all', data=eps_r) # angular average, for -E_max to E_max

    #f.create_dataset('bin_centers', data=bin_centers)

    # Add attributes from input parameters
    for name, val in parmt.__dict__.items():
        if not name.startswith('__'): # ignores dunders
            f.attrs[name] = str(val)

    # Add q and E arrays
    q = np.arange(parmt.q_min + 0.5*parmt.dq, parmt.q_max + parmt.dq, parmt.dq)
    E = np.arange(0, parmt.E_max + parmt.dE, parmt.dE)
    
    f.create_dataset('q', data=q)
    f.create_dataset('E', data=E)

    f.create_dataset('epsilon', data=eps_r[:,(E.shape-1):]) # save only for positive energies

    f.close() # Close hdf5 file

    logger.info(f'Calculation done: dielectric function and input parameters stored as hdf5 file at {parmt.store}/epsilon.hdf5')

#Still needs to be updated for MPI
@time_wrapper
def get_RPA_dielectric_no_LFE_alt_binning(dark_objects: dict):

    # Reading all relevant data

    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[:,:,ivalbot:ivaltop+1]
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy')[:,:,iconbot:icontop+1].conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop+1]
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop+1]
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    unique_q = dark_objects['unique_q']
    n_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']

    unique_Ri = load_unique_R()

    eps_im = []
    qG = []

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0],N_AO,N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    makedir(parmt.store + '/working_dir')

    logger.info(f'Total number of q: {n_q}.')
    for i_q, q in enumerate(list(unique_q.keys())):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {np.array2string(q, precision=5)},')
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]
        logger.info(f'\t\tnumber of G vectors = {len(G_q)},')
        qG.append(q + G_q)

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path)

        eps_im.append(eps_q_im)
        
        logger.info(f'\t\tcomplete. Time taken = {(time.time() - start_time):.2f} s.')

    eps_im = np.concatenate(eps_im, axis=0)
    qG = np.concatenate(qG, axis=0)
    np.save(parmt.store + '/eps_im.npy', eps_im)
    np.save(parmt.store + '/qG.npy', qG)

def get_RPA_dielectric_no_LFE_q(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, primgauss_arr, AO_arr, coeff_arr, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray], einsum_path, working_dir, rank) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,a,b)

    # store relevant quantities, perhaps faster to load in each iteration than supply 
    np.save(working_dir + '/im_energy', im_delE)
    np.save(working_dir + '/k_f', k_f)
    np.save(working_dir + '/mo_coeff_i', mo_coeff_i)
    np.save(working_dir + '/mo_coeff_f_conj', mo_coeff_f_conj)

    qG = q[None, :] + G_q

    epsilon_im = prefactor*RPA_Im_eps_external_prefactor_no_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank)

    return epsilon_im

def RPA_Im_eps_external_prefactor_no_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank):
    """
    Returns imaginary part of lim_{n->0} |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2/(E - (E_{j(k+q)} - E_{ik} + i n)), summed over i,j,k. 

    Inputs: 
        qG:         np.ndarray, shape = (N_G, 3), q+G vectors
        primgauss_arr: (N_blocks, N_primgauss, 3) boolean np.ndarray
            Primitive gaussians are included in each block
        AO_arr: (N_blocks, N_AO) boolean np.ndarray
            Atomic orbitals are included in each block
        coeff_arr: (N_AO, N_max_primgauss) complex np.ndarray
            Primgauss coefficients for each AO
        N_AO:       int, number of atomic orbitals
        q_cuts:     np.ndarray, cut-off points in q for different R_ids.
    Outputs:
        eps_delta:  delta function part of epsilon: Re(eps_delta) corresponds to Im(eps)
                                                    Im(eps_delta) corresponds to Re(eps) 
                    shape = (N_G, N_G, int(parmt.E_max/parmt.dE + 1)), prefactor multiplied later.
    """
    # Load data
    im_delE = np.load(working_dir + '/im_energy.npy')
    k_f = np.load(working_dir + '/k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + '/mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + '/mo_coeff_i.npy')

    #make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    #save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelize over k
        eps_im = p.starmap(partial(eps.get_eps_im_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, im_delE=im_delE), k_tup)
    eps_im = np.sum(np.array(eps_im), axis=0) #(E,G): sum over k

    return np.copy(np.transpose(eps_im, (1,0))) #(G,E)

def get_RPA_dielectric_LFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=None):
    # Reading all relevant data
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']

    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eta_qG')

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q

    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)

        if rank == 0 or rank == None:
            logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {np.array2string(q, precision=5)},')

        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]

        if rank == 0 or rank == None:
            logger.info(f'\t\tnumber of G vectors = {len(G_q)},')

        # delta part of polarizability for E >= 0
        eps_delta_q = get_RPA_dielectric_LFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) #(G,G',E)

        # delta part of polarizability for E < 0
        eps_delta_q_neg = -1*get_RPA_dielectric_LFE_q(q, mo_en_f[:,ivalbot:ivaltop+1], mo_en_i[:,iconbot:icontop+1], mo_coeff_f_conj[:,:,ivalbot:ivaltop+1], mo_coeff_i[:,:,iconbot:icontop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

        # Combine both terms for all E
        eps_delta_q_tot = np.concatenate((np.flip(eps_delta_q_neg[:,:,1:], axis=2), eps_delta_q), axis=2)

        #Kramers-Kronig to get transition amplitude parts of Re(eps) and Im(eps)
        eps_pv_q = kramerskronig_lfe(eps_delta_q_tot)

        eps_q =  eps_pv_q + 1j*eps_delta_q_tot + np.identity(eps_delta_q_tot.shape[0])[:,:,None]

        #take inverse to account for LFEs
        eps_lfe = (1/np.diagonal(np.linalg.inv(eps_q.transpose(2,0,1)), axis1=1, axis2=2)).transpose((1,0)) #(G,G',E) -> (E,G,G') -> (G,E)

        #Bin real and imaginary parts - modify binning so both parts done at same time
        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, np.imag(eps_lfe), bin_centers, tot_bin_eps_im, tot_bin_weights)
        tot_bin_eps_re, tot_bin_weights_re = bin.bin_eps_q(q, G_q, np.real(eps_lfe), bin_centers, tot_bin_eps_re, tot_bin_weights_re)
        
        if rank == 0 or rank == None:
            logger.info(f'\t\tcomplete. Time taken = {(time.time() - start_time):.2f} s.')

    # Removing extra bins 
    tot_bin_eps_im = tot_bin_eps_im[:-N_ang_bins, :]
    tot_bin_eps_re = tot_bin_eps_re[:-N_ang_bins, :]
    tot_bin_weights = tot_bin_weights[:-N_ang_bins]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    shutil.rmtree(working_dir) # delete working directory

    return tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper(n_tabs=2)
def get_RPA_dielectric_LFE_q(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, primgauss_arr, AO_arr, coeff_arr, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray], einsum_path, working_dir, rank) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,a,b)

    # store relevant quantities, perhaps faster to load in each iteration than supply 
    np.save(working_dir + '/im_energy', im_delE)
    np.save(working_dir + '/k_f', k_f)
    np.save(working_dir + '/mo_coeff_i', mo_coeff_i)
    np.save(working_dir + '/mo_coeff_f_conj', mo_coeff_f_conj)

    qG = q[None, :] + G_q

    epsilon_delta = prefactor*RPA_Im_eps_external_prefactor_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank)
    
    return epsilon_delta

def RPA_Im_eps_external_prefactor_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank):
    """
    Returns delta function part of lim_{n->0} |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2/(E - (E_{j(k+q)} - E_{ik} + i n)), summed over i,j,k. 

    Inputs: 
        qG ((N_G, 3) np.ndarray): q+G vectors
        primgauss_arr ((N_blocks, N_primgauss, 3) boolean np.ndarray):
            Primitive gaussians are included in each block
        AO_arr ((N_blocks, N_AO) boolean np.ndarray):
            Atomic orbitals are included in each block
        coeff_arr ((N_AO, N_max_primgauss) complex np.ndarray):
            Primgauss coefficients for each AO
        q_cuts ((N_q,) np.ndarray): 
            cut-off points in q for different R_ids.
    Outputs:
        eps_delta ((N_G, N_G, int(parmt.E_max/parmt.dE + 1)) np.ndarray):
            Delta function part of epsilon: Re(eps_delta) corresponds to Im(eps) and Im(eps_delta) corresponds to Re(eps)
    """
    # Load data
    im_delE = np.load(working_dir + '/im_energy.npy')
    k_f = np.load(working_dir + '/k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + '/mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + '/mo_coeff_i.npy')

    N_E = int(parmt.E_max/parmt.dE + 1)

    start_time = time.time()
    #make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    #save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelize over k
        p.starmap(partial(eps.get_3D_overlaps_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, working_dir=working_dir), k_tup) #(G,k,i,j)

    if rank == None or rank == 0:
        logger.info(f'\t\t\teta_qG calculated for all k and G. Time taken = {(time.time() - start_time):.2f} s.')

    start_time = time.time()
    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]), dtype='complex128')
    for i_k in range(k_f.shape[0]):
        eta_qG[i_k] = np.load(working_dir + f'/eta_qG/eta_qG_k{i_k}.npy')
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:]

    if rank == None or rank == 0:
        logger.info(f'\t\t\t3D overlaps loaded from memory for all G. Time taken = {(time.time() - start_time):.2f} s.')

    start_time = time.time()
    eps_delta = np.zeros((N_E, qG.shape[0], qG.shape[0]), dtype='complex')
    for i_k in range(k_f.shape[0]):
        a_ind, b_ind = np.nonzero(im_delE[i_k,0] < N_E)
        eta_qG_ab = eta_qG[i_k, a_ind, b_ind] #(a,b,G) -> (ab,G) #only keeps a,b pairs relevent to delta calculation
        im_delE_ab = im_delE[i_k, :, a_ind, b_ind] #(2, ab)
        eta_qG_sq = np.einsum('ag, ah -> agh', eta_qG_ab, eta_qG_ab.conj()) #(ab,G,G')

        for i in range(eta_qG_sq.shape[0]):
            ind, rem = im_delE_ab[i]
            eps_delta[int(ind)] += rem*eta_qG_sq[i] #already checked ind < nE
            if ind < N_E - 1:
                eps_delta[int(ind+1)] += (1. - rem)*eta_qG_sq[i]

    if rank == None or rank == 0:
        logger.info(f'\t\t\tDelta part of epsilon calculated. Time taken = {(time.time() - start_time):.2f} s.')


    #too slow to load all eps_delta from memory, will only get worse with larger N_G
    """
    #save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelize over k
        p.starmap(partial(eps.get_eps_delta_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, im_delE=im_delE), k_tup) #(G,k,i,j)

    logger.info('\t\t\teps_delta calculated for all k and G. Time taken = {:.2f} s.'.format(time.time()-start_time))

    #load and add eps_delta for all k
    start_time = time.time()
    eps_delta = np.zeros((N_E, qG.shape[0], qG.shape[0]), dtype='complex')
    for i_k in range(k_f.shape[0]):
        eps_delta += np.load(parmt.store + f'/working_dir/eps_delta/eps_delta_k{i_k}.npy')

    logger.info('\t\t\teps_delta loaded from memory and summed for all k. Time taken = {:.2f} s.'.format(time.time()-start_time))
    """

    return np.copy(np.transpose(eps_delta, (1,2,0))) #(G,G',E)

@njit
def delta_energy(mo_en_i, mo_en_f):
    """
    Get the lower energy bin and the remainder to feed into that particular bin.
    """
    nk, nmo, nmu = mo_en_f.shape[0], mo_en_i.shape[1], mo_en_f.shape[1]
    delE = np.abs((mo_en_f[:, None, :] - mo_en_i[:, :, None])/parmt.dE)
    arr = np.zeros((nk, 2, nmo, nmu), dtype = np.float32)
    arr[:,0,:,:] = delE//1
    arr[:,1,:,:] = 1. - delE%1
    return arr

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

@time_wrapper(n_tabs=2)
def kramerskronig_lfe_causal(eps_delta):
    """
    Calculates principal value part of Re(eps) and Im(eps) for LFEs. This function is parallelized over G-vectors (over first G-vector of eps_delta).

    Input:
        eps_delta: np.ndarray of shape (N_G, N_G, N_E): Dirac delta part of dielectric function

    Output: 
        epa_pv: np.ndarray of shape (N_G, N_G, N_E): Principal value part of dielectric function
    
    """
    eps_delta_re = np.real(eps_delta)
    eps_delta_im = np.imag(eps_delta)
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        eps_pv_re = p.map(eps.kramerskronig_im2re_causal, eps_delta_re)
        eps_pv_im = p.map(eps.kramerskronig_re2im_causal, eps_delta_im)
    eps_pv = np.array(eps_pv_re) - 1j*np.array(eps_pv_im)
    return eps_pv

@time_wrapper(n_tabs=2)
def kramerskronig_lfe(eps_delta):
    """
    Calculates principal value part of Re(eps) and Im(eps) for LFEs. This function is parallelized over G-vectors (over first G-vector of eps_delta).

    Input:
        eps_delta: np.ndarray of shape (N_G, N_G, N_E): Dirac delta part of dielectric function

    Output: 
        epa_pv: np.ndarray of shape (N_G, N_G, N_E): Principal value part of dielectric function
    
    """
    eps_delta_re = np.real(eps_delta)
    eps_delta_im = np.imag(eps_delta)
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        eps_pv_re = p.map(eps.kramerskronig_im2re, eps_delta_re)
        eps_pv_im = p.map(eps.kramerskronig_re2im, eps_delta_im)
    eps_pv = np.array(eps_pv_re) - 1j*np.array(eps_pv_im)
    return eps_pv

#May want to put functions below into separate post-processing module? 

def epsilon_r(bin_centers, binned_eps, eps_dtype='complex'):
    """
    Calculate angular averaged dielectric function eps(|q|, E) from binned epsilon.
    """
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    r = np.unique(bin_centers[:,0])
    eps_r = np.zeros((r.shape[0], binned_eps.shape[1]), dtype=eps_dtype)
    for i, r_i in enumerate(r):
        eps_ri = binned_eps[i*N_ang_bins:(i+1)*N_ang_bins]
        eps_r[i] = np.nansum(eps_ri, axis=0) / (N_ang_bins - np.sum(np.isnan(eps_ri).astype(int), axis=0)) #treats nans as 0, want to average over all non-nan entries
    return eps_r

def interp_eps(bin_centers_sph, binned_eps):
    """
    Interpolate missing bins of epsilon. Can be used to interpolate Im(eps) before computing Re(eps) with Kramers-Kronig, or to interpolate Re(eps) and Im(eps) in the LFE case.

    bin_centers input should be in spherical coordinates
    """
    bin_centers = bin.spherical_to_cartesian(bin_centers_sph)

    nan_loc = np.where(np.isnan(binned_eps[:,0]))[0] #Find indices of missing bins
    nan_bins = bin_centers[nan_loc]

    interp_loc = np.where(np.invert(np.isnan(binned_eps[:,0])))[0] #Use remaining bins for interpolation input
    interp_bins = bin_centers[interp_loc]
    interp_eps = binned_eps[interp_loc]

    binned_eps_interp = binned_eps.copy()

    interp = LinearNDInterpolator(interp_bins, interp_eps)(nan_bins)
    binned_eps_interp[nan_loc] = interp #replace nans with interpolated data

    return binned_eps_interp