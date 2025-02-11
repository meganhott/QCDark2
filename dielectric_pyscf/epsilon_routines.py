import numpy as np
import time
import psutil
from os import getpid
import multiprocessing as mp
from functools import partial
from numba import njit, prange
from scipy.interpolate import LinearNDInterpolator

from dielectric_pyscf.routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import dielectric_pyscf.input_parameters as parmt
import dielectric_pyscf.epsilon_helper as eps
import dielectric_pyscf.binning as bin

def get_RPA_dielectric(dark_objects: dict):
    if parmt.include_lfe:
        get_RPA_dielectric_LFE(dark_objects)
    else:
        if parmt.alt_binning: #Temporary
            get_RPA_dielectric_no_LFE_alt_binning(dark_objects)
        else:
            get_RPA_dielectric_no_LFE(dark_objects)

@time_wrapper
def get_RPA_dielectric_no_LFE(dark_objects: dict):

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
    blocks = dark_objects['blocks']

    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    
    # Make working directory
    makedir(parmt.store + '/working_dir')

    logger.info('Total number of q: {}.'.format(n_q))

    if parmt.q_start is not None: # Starts from intermediate q vector if specified
        q_start = parmt.q_start
        q_keys = list(unique_q.keys())[q_start-1:]
        # Load saved intermediate calculations
        try:
            tot_bin_eps_im = np.load(parmt.store + '/working_dir/tot_bin_eps_im.npy')
            tot_bin_weights = np.load(parmt.store + '/working_dir/tot_bin_weights.npy')
        except FileNotFoundError:
            raise Exception('Intermediate calculations were not found in {}. To start from a specified q vector, input_parameters.save_temp_eps must be set to True. To run calculation for all q vectors, set input_parameters.q_start = None'.format(parmt.store+'/working_dir'))
    else: # Otherwise perform calculation for all q
        q_start = 1
        q_keys = unique_q.keys()
        tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
        tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info('\ti_q: {}\n\t\tq = {},'.format(i_q, np.array2string(q, precision=5)))
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri)

        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)

        if parmt.save_temp_eps:
            np.save(parmt.store + '/working_dir/tot_bin_eps_im.npy', tot_bin_eps_im)
            np.save(parmt.store + '/working_dir/tot_bin_weights.npy', tot_bin_weights)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 

    #Interpolating missing Im(eps) bins and then performing Kramers-Kronig transformation to get Re(eps)
    binned_eps_im_interp = interp_eps(bin_centers, binned_eps_im)
    
    np.save(parmt.store+'/binned_eps.npy', eps.kramerskronig_im2re(binned_eps_im) + 1. + 1j*binned_eps_im) #no interpolation 

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
    blocks = dark_objects['blocks']

    unique_Ri = load_unique_R()

    # Generating energy centers & bins
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    bin_centers = bin.gen_bin_centers(cartesian=True)
    #N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    #tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
    #tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)
    eps_im = []
    qG = []
    
    # Make working directory
    makedir(parmt.store + '/working_dir')

    logger.info('Total number of q: {}.'.format(n_q))
    for i_q, q in enumerate(list(unique_q.keys())):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info('\ti_q: {}\n\t\tq = {},'.format(i_q+1, np.array2string(q, precision=5)))
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))
        qG.append(q + G_q)

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri)

        eps_im.append(eps_q_im)

        #tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    eps_im = np.concatenate(eps_im, axis=0)
    qG = np.concatenate(qG, axis=0)
    np.save(parmt.store + '/eps_im.npy', eps_im)
    np.save(parmt.store + '/qG.npy', qG)


    #binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 

    #Interpolating missing Im(eps) bins and then performing Kramers-Kronig transformation to get Re(eps)
    start_time = time.time()
    binned_eps_im = LinearNDInterpolator(qG, eps_im)(bin_centers)
    logger.info('Interpolation to bins complete. Time taken = {:.2f} s.'.format(time.time() - start_time))
    np.save(parmt.store+'/binned_eps_altbin.npy', eps.kramerskronig_im2re(binned_eps_im) + 1. + 1j*binned_eps_im)

def get_RPA_dielectric_no_LFE_q(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, blocks: dict, N_AO: int, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray]) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f)

    # store relevant quantities, perhaps faster to load in each iteration than supply 
    working_dir = parmt.store + '/working_dir/'
    np.save(working_dir + 'im_energy', im_delE)
    np.save(working_dir + 'k_f', k_f)
    np.save(working_dir + 'mo_coeff_i', mo_coeff_i)
    np.save(working_dir + 'mo_coeff_f_conj', mo_coeff_f_conj)

    qG = q[None, :] + G_q

    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        epsilon_im = p.map(partial(eps.RPA_Im_eps_external_prefactor_no_LFE, unique_Ri=unique_Ri, blocks=blocks, N_AO=N_AO, q_cuts=q_cuts), qG)
    epsilon_im = prefactor*np.array(epsilon_im)

    return epsilon_im

def make_blocks_arrays(blocks):
    block_keys = list(blocks.keys())
    block_values = list(blocks.values())
    N_blocks = len(block_keys)
    N_primgauss = max([bij for bi in [aij for ai in block_keys for aij in ai] for bij in bi]) + 1
    N_AO = max([aij for ai in (list(b.keys()) for b in block_values) for aij in ai]) + 1
    N_maxcoeff = max([aij.shape[0] for ai in (list(b.values()) for b in block_values) for aij in ai])

    primgauss_arr = np.zeros((N_blocks, N_primgauss, 3), dtype='bool')
    for i, b in enumerate(block_keys):
        for d in range(3):
            primgauss_arr[i,np.array(b)[d],d] = 1

    AO_arr = np.zeros((N_blocks, N_AO), dtype='bool')
    AOs = [list(b.keys()) for b in block_values]
    for i, a in enumerate(AOs): #n_block, AOs
        AO_arr[i,a] = 1

    coeff_arr = np.zeros((N_AO, N_maxcoeff), dtype='complex') #actually just real floats, but need to match complex dtype for numba optimization later 
    coeff = [list(b.values()) for b in block_values]
    for b in block_values:
        for a in list(b.keys()): #[AO1, AO2, ...]
            coeff = b[a]
            coeff_arr[a, :coeff.shape[0]] = coeff

    return primgauss_arr, AO_arr, coeff_arr

def get_RPA_dielectric_LFE(dark_objects: dict):
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
    blocks = dark_objects['blocks']

    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)

    # Generate arrays from blocks dict (temporary)
    primgauss_arr, AO_arr, coeff_arr = make_blocks_arrays(blocks)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0],N_AO,N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    makedir(parmt.store + '/working_dir')
    makedir(parmt.store + '/working_dir/eta_qG')

    logger.info('Total number of q: {}.'.format(n_q))

    if parmt.q_start is not None: # Starts from intermediate q vector if specified
        q_start = parmt.q_start
        q_keys = list(unique_q.keys())[q_start-1:]
        # Load saved intermediate calculations
        try:
            tot_bin_eps_im = np.load(parmt.store + '/working_dir/tot_bin_eps_im.npy')
            tot_bin_eps_re = np.load(parmt.store + '/working_dir/tot_bin_eps_re.npy')
            tot_bin_weights = np.load(parmt.store + '/working_dir/tot_bin_weights.npy')
            tot_bin_weights_re = tot_bin_weights
        except FileNotFoundError:
            raise Exception('Intermediate calculations were not found in {}. To start from a specified q vector, input_parameters.save_temp_eps must be set to True. To run calculation for all q vectors, set input_parameters.q_start = None'.format(parmt.store+'/working_dir'))
    else: # Otherwise perform calculation for all q
        q_start = 1
        q_keys = unique_q.keys()

        tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
        tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

        tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
        tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info('\ti_q: {}\n\t\tq = {},'.format(i_q, np.array2string(q, precision=5)))
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))

        eps_delta_q = get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path) #(G,G',E)

        #Kramers-Kronig to get transition amplitude parts of Re(eps) and Im(eps)
        eps_pv_q= kramerskronig_lfe(eps_delta_q)

        eps_q =  eps_pv_q + 1j*eps_delta_q + np.identity(eps_delta_q.shape[0])[:,:,None]

        #take inverse to account for LFEs
        eps_lfe = (1/np.diagonal(np.linalg.inv(eps_q.transpose(2,0,1)), axis1=1, axis2=2)).transpose((1,0)) #(G,G',E) -> (E,G,G') -> (G,E)

        #Bin real and imaginary parts - modify binning so both parts done at same time
        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, np.imag(eps_lfe), bin_centers, tot_bin_eps_im, tot_bin_weights)
        tot_bin_eps_re, tot_bin_weights_re = bin.bin_eps_q(q, G_q, np.real(eps_lfe), bin_centers, tot_bin_eps_re, tot_bin_weights_re)

        if parmt.save_temp_eps:
            np.save(parmt.store + '/working_dir/tot_bin_eps_im.npy', tot_bin_eps_im)
            np.save(parmt.store + '/working_dir/tot_bin_eps_re.npy', tot_bin_eps_re)
            np.save(parmt.store + '/working_dir/tot_bin_weights.npy', tot_bin_weights)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 
    binned_eps_re = tot_bin_eps_re[:-N_ang_bins, :]/tot_bin_weights_re[:-N_ang_bins, None] #removing extra bins 

    #Could interpolate eps_im and re-compute eps_re from KK
    #binned_eps_re_kk = kramerskronig(binned_eps_im)

    np.save(parmt.store+'/binned_eps_lfe.npy', binned_eps_re + 1j*binned_eps_im)
    #np.save(parmt.store+'/binned_eps_re_kk.npy', binned_eps_re_kk + 1j*binned_eps_im)

@time_wrapper(n_tabs=2)
def get_RPA_dielectric_LFE_q(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, primgauss_arr, AO_arr, coeff_arr, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray], einsum_path) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #change shape in future!
    im_delE = np.copy(np.transpose(im_delE, (0,3,1,2))) #(k,2,a,b)

    # store relevant quantities, perhaps faster to load in each iteration than supply 
    working_dir = parmt.store + '/working_dir/'
    np.save(working_dir + 'im_energy', im_delE)
    np.save(working_dir + 'k_f', k_f)
    np.save(working_dir + 'mo_coeff_i', mo_coeff_i)
    np.save(working_dir + 'mo_coeff_f_conj', mo_coeff_f_conj)

    qG = q[None, :] + G_q

    epsilon_delta = prefactor*RPA_Im_eps_external_prefactor_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path)
    
    return epsilon_delta

def RPA_Im_eps_external_prefactor_LFE(qG, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path):
    """
    Returns delta function part of lim_{n->0} |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2/(E - (E_{j(k+q)} - E_{ik} + i n)), summed over i,j,k. 

    Inputs: 
        qG:         np.ndarray, shape = (N_G, 3), q+G vectors
        blocks:     dict, atomic orbital blocks generated by get_basis_blocks
        N_AO:       int, number of atomic orbitals
        q_cuts:     np.ndarray, cut-off points in q for different R_ids.
    Outputs:
        eps_delta:  delta function part of epsilon: Re(eps_delta) corresponds to Im(eps)
                                                    Im(eps_delta) corresponds to Re(eps) 
                    shape = (N_G, N_G, int(parmt.E_max/parmt.dE + 1)), prefactor multiplied later.
    """
    # Load data
    working_dir = parmt.store + '/working_dir/'
    im_delE = np.load(working_dir + 'im_energy.npy')


    k_f = np.load(working_dir + 'k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + 'mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + 'mo_coeff_i.npy')

    N_E = int(parmt.E_max/parmt.dE + 1)

    start_time = time.time()
    #make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    #save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelize over k
        p.starmap(partial(eps.get_3D_overlaps_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path), k_tup) #(G,k,i,j)

    logger.info('\t\t\teta_qG calculated for all k and G. Time taken = {:.2f} s.'.format(time.time()-start_time))

    start_time = time.time()
    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]), dtype='complex128')
    for i_k in range(k_f.shape[0]):
        eta_qG[i_k] = np.load(parmt.store + f'/working_dir/eta_qG/eta_qG_k{i_k}.npy')
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:]

    logger.info('\t\t\t3D overlaps loaded from memory for all G. Time taken = {:.2f} s.'.format(time.time()-start_time))

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

    logger.info('\t\t\tDelta part of epsilon calculated. Time taken = {:.2f} s.'.format(time.time()-start_time))


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
    delE = (mo_en_f[:, None, :] - mo_en_i[:, :, None])/parmt.dE
    arr = np.zeros((nk, nmo, nmu, 2), dtype = np.float32)
    arr[:,:,:,0] = delE//1
    arr[:,:,:,1] = 1. - delE%1
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

def get_Nk(unique_Ri, blocks, N_AO):
    """
    Returns number of k pairs to include in 3D overlaps multiprocessing based on the available memory and number of cores. Used for LFE calculations.
    """
    max_memory = psutil.virtual_memory().available*0.9
    init_memory = psutil.Process(getpid()).memory_info()[0] #memory currently used 
    N_cpus = mp.cpu_count()

    N_uniqueRi = sum([Ri.shape[0] for Ri in unique_Ri])
    N_PG = max(np.concatenate([k1 for k in blocks.keys() for k1 in k])) + 1
    N_max_block = max([len(k1) for k in blocks.keys() for k1 in k])
    N_Rid = np.load(parmt.store+'/R_ids/-1.npy').shape[1]
    overlaps_memory = (N_uniqueRi*N_PG**2 + N_AO**2 + N_uniqueRi*N_PG*N_max_block + N_Rid*N_max_block**2 + N_uniqueRi*N_max_block**2) * 16
    
    Nk = int((max_memory - init_memory*N_cpus) / overlaps_memory / N_cpus)
    return Nk

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

    interp = LinearNDInterpolator(interp_bins, interp_eps)(nan_bins)
    binned_eps[nan_loc] = interp #replace nans with interpolated data

    return binned_eps