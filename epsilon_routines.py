import numpy as np
import time
import multiprocessing as mp
from functools import partial

from routines import logger, time_wrapper, load_unique_R
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

        #Compute energy differences for denominator of Re(chi), and Im(chi)
        re_delE, im_delE = get_energy_diff(mo_en_i_q, mo_en_f_q, E)

        start_eps_q = time.time()
        
        #Compute epsilon for all G_q
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:  #parallelization over G
            epsilon = p.map(partial(RPA_eps, q=q, k_f=k_f_q, mo_coeff_i=mo_coeff_i_q, mo_coeff_f_conj=mo_coeff_f_q_conj, re_delE=re_delE, im_delE=im_delE, unique_Ri=unique_Ri, blocks=blocks, N_AO=N_AO, q_cuts=q_cuts, N_kpairs=N_kpairs, V_cell=V_cell), G_q)
        epsilon = np.array(epsilon) #(G, E)

        #bin eps calculated for all G
        tot_bin_eps, tot_bin_weights = bin.bin_eps_q(q, G_q, epsilon, bin_centers, tot_bin_eps, tot_bin_weights)
        
        end_eps_q = time.time()
        
        logger.info('epsilon_GG(q, E) calculated and binned for all G and E for 1BZ q vector {} ({}/{}). Time taken = {:.2f} s.\n'.format(np.array2string(q, precision=5), i_q+1, n_q, end_eps_q - start_eps_q))
    return tot_bin_eps, tot_bin_weights

def get_energy_diff(mo_en_i, mo_en_f, E):
    """
    Notes:
    - numba roughly halves runtime, still only a few ms, but cannot use njit with errstate
    To Do:
    - Account for Re = inf when the energy difference exactly = 0
    - Is it faster to save and load this?

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

    with np.errstate(divide='ignore'): #need this to avoid ZeroDivisionError
        re_delE = np.where(re_delE == 0, 0, re_delE**(-1)) #gives inverse of energy difference unless delE == 0, in which case 0 is returned since there is no real part
    return re_delE, -np.pi*im_delE

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

def epsilon_r(bin_centers, binned_eps):
    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    r = np.unique(bin_centers[:,0])
    eps_r = np.zeros((r.shape[0], binned_eps.shape[1]), dtype='complex')
    for i, r_i in enumerate(r):
        eps_ri = binned_eps[i*N_ang_bins:(i+1)*N_ang_bins]
        eps_r[i] = np.nansum(eps_ri, axis=0) / (N_ang_bins - np.sum(np.isnan(eps_ri).astype(int), axis=0)) #treats nans as 0, want to average over all non-nan entries
    return eps_r