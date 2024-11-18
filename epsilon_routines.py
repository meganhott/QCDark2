import numpy as np
import time
import psutil
from os import getpid
import multiprocessing as mp
from functools import partial
from numba import njit, prange
from scipy.interpolate import LinearNDInterpolator

from routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import input_parameters as parmt
import epsilon_helper as eps
import binning as bin

def get_RPA_dielectric(dark_objects: dict):
    if parmt.include_lfe:
        get_RPA_dielectric_LFE(dark_objects)
    else:
        get_RPA_dielectric_no_LFE_alt_binning(dark_objects)

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

    # Generating energy centers & bins
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)
    
    # Make working directory
    makedir(parmt.store + '/working_dir')

    logger.info('Total number of q: {}.'.format(n_q))
    for i_q, q in enumerate(unique_q.keys()):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info('\ti_q: {}\n\t\tq = {},'.format(i_q+1, np.array2string(q, precision=5)))
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri)

        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)
        
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

    # Generating energy centers & bins
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
    tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)
    
    # Make working directory
    makedir(parmt.store + '/working_dir')

    logger.info('Total number of q: {}.'.format(n_q))

    for i_q, q in enumerate(unique_q.keys()):

        k_pairs = np.array(unique_q[q])
        q = np.array(q)
        logger.info('\ti_q: {}\n\t\tq = {},'.format(i_q+1, np.array2string(q, precision=5)))
        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))

        eps_delta_q = get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri) #(G,G',E)

        #Kramers-Kronig to get transition amplitude parts of Re(eps) and Im(eps)
        eps_pv_q= kramerskronig_lfe(eps_delta_q)

        eps_q =  eps_pv_q + 1j*eps_delta_q + np.identity(eps_delta_q.shape[0])[:,:,None]

        #take inverse to account for LFEs
        eps_lfe = (1/np.diagonal(np.linalg.inv(eps_q.transpose(2,0,1)), axis1=1, axis2=2)).transpose((1,0)) #(G,G',E) -> (E,G,G') -> (G,E)

        #Bin real and imaginary parts - modify binning so both parts done at same time
        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, np.imag(eps_lfe), bin_centers, tot_bin_eps_im, tot_bin_weights)
        tot_bin_eps_re, tot_bin_weights_re = bin.bin_eps_q(q, G_q, np.real(eps_lfe), bin_centers, tot_bin_eps_re, tot_bin_weights_re)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 
    binned_eps_re = tot_bin_eps_re[:-N_ang_bins, :]/tot_bin_weights_re[:-N_ang_bins, None] #removing extra bins 

    #Could interpolate eps_im and re-compute eps_re from KK
    #binned_eps_re_kk = kramerskronig(binned_eps_im)

    np.save(parmt.store+'/binned_eps_lfe.npy', binned_eps_re + 1j*binned_eps_im)
    #np.save(parmt.store+'/binned_eps_re_kk.npy', binned_eps_re_kk + 1j*binned_eps_im)

@time_wrapper(n_tabs=2)
def get_RPA_dielectric_LFE_q(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, blocks: dict, N_AO: int, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray]) -> np.ndarray:
    
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

    epsilon_delta = prefactor*RPA_Im_eps_external_prefactor_LFE(qG, blocks, N_AO, q_cuts, unique_Ri)
    
    return epsilon_delta

def RPA_Im_eps_external_prefactor_LFE(qG, blocks, N_AO, q_cuts, unique_Ri):
    """
    Returns delta function part of lim_{n->0} |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2/(E - (E_{j(k+q)} - E_{ik} + i n)), summed over i,j,k. 

    Notes:
        - Can we parallelize or otherwise speed up loop over k? Pretty slow when including many G-vectors
        - Speed up with numba? Is saving+loading data and using for loops or keeping data in memory and optimizing for loops with numba faster?

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
    energy_arr = np.load(working_dir + 'im_energy.npy')
    k_f = np.load(working_dir + 'k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + 'mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + 'mo_coeff_i.npy')

    nE = int(parmt.E_max/parmt.dE + 1)
    eps_delta = np.zeros((qG.shape[0], qG.shape[0], int(parmt.E_max/parmt.dE + 1)), dtype='complex')

    Nk = get_Nk(unique_Ri, blocks, N_AO) #number of k such that calculation of eta(G,k,i,j) can be run in parallel without exceeding memory  
    partitions = np.append(np.arange(0, k_f.shape[0], Nk), [k_f.shape[0]])
    start_time = time.time()
    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]), dtype='complex') #(k,i,j,G)
    for p in range(len(partitions) - 1):
        i, f = partitions[p], partitions[p+1]
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:
            eta_qG_p = p.map(partial(eps.get_3D_overlaps_blocks, k_f=k_f[i:f], blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i:f], mo_coeff_f_conj=mo_coeff_f_conj[i:f], unique_Ri=unique_Ri, q_cuts=q_cuts), qG) #(G,k,i,j)
        eta_qG[i:f] = np.array(eta_qG_p).transpose((1,2,3,0))
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:] #(k,i,j,G)

    logger.info('\t\t\t3D overlaps loaded with {} k partitions. Time taken = {:.2f} s.'.format(len(partitions)-1, time.time()-start_time))
    
    start_time = time.time()
    for i_k in range(k_f.shape[0]):
        eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG[i_k], eta_qG[i_k].conj())
        for a in range(mo_coeff_i.shape[2]):
            for b in range(mo_coeff_f_conj.shape[2]):
                ind, rem = tuple(energy_arr[i_k, a, b])
                if ind < nE:
                    eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
                if ind < nE - 1:
                    eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]

    logger.info('\t\t\tDelta part of epsilon calculated. Time taken = {:.2f} s.'.format(time.time()-start_time))

    return eps_delta #(G,G',E)

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
    N_Rid = np.load('test_resources/R_ids/-1.npy').shape[1]
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