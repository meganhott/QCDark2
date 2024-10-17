import numpy as np
import time
import multiprocessing as mp
from functools import partial
from numba import njit
from scipy.interpolate import LinearNDInterpolator

from routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import input_parameters as parmt
import epsilon_helper as eps
import binning as bin

@time_wrapper
def initialize_RPA_dielectric(dark_objects):
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
    n_q = len(unique_q) #number of 1BZ q vectors
    G_vectors = dark_objects['G_vectors']
    R_vectors = dark_objects['R_vectors']
    N_AO = len(dark_objects['aos'])
    blocks = dark_objects['blocks']
    V_cell = dark_objects['V_cell']
    q_cuts = dark_objects['R_cutoff_q_points']

    unique_Ri = load_unique_R()

    #energy bins
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)

    #initialize spacial bins
    bin_centers = bin.gen_bin_centers()
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    tot_bin_eps = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1), dtype='complex')
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)
    
    if parmt.include_lfe:
        raise NotImplementedError()
        RPA_eps = eps.RPA_dielectric_lfe
    else:
        RPA_eps = eps.RPA_dielectric

    for i_q, q in enumerate(unique_q.keys()):
        k_pairs = np.array(unique_q[q])
        N_kpairs = k_pairs.shape[0]

        #Only include G_vectors such that |q + G| < q_max
        q = np.array(q)
        G_q = G_vectors[np.linalg.norm(q+G_vectors, axis=1) < parmt.q_max]

        #generate parameters for q
        mo_en_i_q = mo_en_i[k_pairs[:,0]] #(k_pair,i)
        mo_en_f_q = mo_en_f[k_pairs[:,1]] #(k_pair,j)
        mo_coeff_i_q = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
        mo_coeff_f_q_conj = mo_coeff_f[k_pairs[:,1]].conj() #(k_pair,b,j)
        k_f_q = k_f[k_pairs[:,1]]

        #Compute and store energy differences for denominator of Re(chi), and Im(chi)
        get_energy_diff(mo_en_i_q, mo_en_f_q, E)

        start_eps_q = time.time()
        
        #Compute epsilon for all G_q
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:  #parallelization over G
            epsilon = p.map(partial(RPA_eps, q=q, k_f=k_f_q, mo_coeff_i=mo_coeff_i_q, mo_coeff_f_conj=mo_coeff_f_q_conj, unique_Ri=unique_Ri, blocks=blocks, N_AO=N_AO, q_cuts=q_cuts, N_kpairs=N_kpairs, V_cell=V_cell), G_q)
        epsilon = np.array(epsilon) #(G, E)

        #bin eps calculated for all G
        tot_bin_eps, tot_bin_weights = bin.bin_eps_q(q, G_q, epsilon, bin_centers, tot_bin_eps, tot_bin_weights)
        
        end_eps_q = time.time()
        
        logger.info('epsilon_GG(q, E) calculated and binned for all G and E for 1BZ q vector {} ({}/{}). Time taken = {:.2f} s.\n'.format(np.array2string(q, precision=5), i_q+1, n_q, end_eps_q - start_eps_q))
    return tot_bin_eps, tot_bin_weights

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

@time_wrapper
def get_RPA_dielectric_no_LFE(dark_objects: dict) -> tuple[np.ndarray, np.ndarray]:

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
    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1), dtype='complex')
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
        logger.info('\t\tnumber of G vectors = {},'.format(len(G_q)))

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri)

        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 

    #Eventually want to add interpolation of missing bins for Im(eps) before performing Kramers-Kronig transformation to get Re(eps)

    binned_eps_re = kramerskronig(binned_eps_im)
    
    return binned_eps_re, binned_eps_im

def get_RPA_dielectric(dark_objects: dict) -> tuple[np.ndarray, np.ndarray]:
    
    if parmt.include_lfe:
        raise NotImplementedError('Local Field Effects are not yet implemented. The optimization will be done via a different route and needs to be done at a later stage.')
    else:
        return get_RPA_dielectric_no_LFE(dark_objects)

def get_energy_diff(mo_en_i, mo_en_f, E):
    """
    Notes:
    - numba roughly halves runtime, still only a few ms, but cannot use njit with errstate
    - Faster to save and load im_delE and re_delE instead of passing to multiprocessing, speed up is even better for larger arrays
    To Do:
    - Account for Re = inf when the energy difference exactly = 0

    Computes contributions to Re(chi) and Im(chi) coming from the energy difference denominator in Eq. (). Imaginary contributions arise when the energy difference is less than parmt.dE. We use a triangle approximation of the dirac delta function with width determined by parmt.dE

    Inputs:
        mo_en_i:    np.ndarray of shape (N_kpairs, N_valbands): initial molecular orbital energies
        mo_en_f:    np.ndarray of shape (N_kpairs, N_conbands): final molecular orbital energies
        E:          np.ndarray of shape (N_E,): energy transferred to electron

    Stored:
        re_delE:  np.ndarray of shape (k_pairs, i, j, E): Real part of the energy denominator
        im_delE:  np.ndarray of shape (k_pairs, i, j, E): Imaginary part of the denominator
    """
    re_delE = (E[None,None,None,:] - mo_en_f[:,None,:,None] + mo_en_i[:,:,None,None]) #(k_pair,i,j,E)
    im_delE = (np.abs(re_delE) < parmt.dE) * (1.0 - np.abs(re_delE)/parmt.dE)

    with np.errstate(divide='ignore'): #need this to avoid ZeroDivisionError
        re_delE = np.where(re_delE == 0, 0, re_delE**(-1)) #gives inverse of energy difference unless delE == 0, in which case 0 is returned since there is no real part
    np.save(parmt.store+'/re_delE.npy', re_delE)
    np.save(parmt.store+'/im_delE.npy', -np.pi*im_delE)

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

@time_wrapper
def kramerskronig(eps_im):
    """
    Notes:
        Add warning for E_max < plasmon?
        Optimize with numba
    """
    E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)
    eps_re = np.empty_like(eps_im)
    for n, En in enumerate(E):
        E_pv = np.delete(E, n) #removes Ei = En for principal value
        eps_im_pv = np.delete(eps_im, n, axis=1)

        eps_re[:,n] = 2/np.pi*parmt.dE*(np.sum(E_pv[None,:] * eps_im_pv / (E_pv[None,:]**2 - En**2), axis=1) - 0.5*(E_pv[None,0]*eps_im_pv[:,0]/(E_pv[None,0]**2-En**2) + E_pv[None,-1]*eps_im_pv[:,-1]/(E_pv[None,-1]**2-En**2))) #trapezoid rule
    return eps_re + 1

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
    Interpolate missing bins of Im(eps) before computing Re(eps) with Kramers-Kronig. Returns interpolated real and imaginary parts of binned epsilon.

    bin_centers input should be in spherical coordinates
    """
    eps_im = np.imag(binned_eps)
    bin_centers = bin.spherical_to_cartesian(bin_centers_sph)

    nan_loc = np.where(np.isnan(eps_im[:,0]))[0] #Find indices of missing bins
    nan_bins = bin_centers[nan_loc]

    interp_loc = np.where(np.invert(np.isnan(eps_im[:,0])))[0] #Use remaining bins for interpolation input
    interp_bins = bin_centers[interp_loc]
    interp_eps = eps_im[interp_loc]

    interp = LinearNDInterpolator(interp_bins, interp_eps)(nan_bins)
    eps_im[nan_loc] = interp #replace nans with interpolated data

    return kramerskronig(eps_im) + 1j*eps_im