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

from epsilon_routines import kramerskronig, kramerskronig_lfe, delta_energy


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

        eps_q_im = get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri) #(G,G',E)

        #Kramers-Kronig to get eps_re
        eps_q_re = kramerskronig_lfe(eps_q_im)

        #need eps_re to calculate inverse matrix, but can we do kramers-kronig over each GG' separately? Interpolate first?
        #Doesn't make sense to bin before doing lfe calculation since that averages over space

        #take inverse to account for LFEs
        eps_lfe = 1/np.diagonal(np.linalg.inv(eps_q_re + 1j*eps_q_im), axis1=0, axis2=1) #check that this is correct diagonal

        #Could now bin just imaginary part and then use KK to calculate real part again, or bin over both real and imaginary parts

        tot_bin_eps_im, tot_bin_weights = bin.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)
        
        logger.info('\t\tcomplete. Time taken = {:.2f} s.'.format(time.time() - start_time))

    binned_eps_im = tot_bin_eps_im[:-N_ang_bins, :]/tot_bin_weights[:-N_ang_bins, None] #removing extra bins 

    #Eventually want to add interpolation of missing bins for Im(eps) before performing Kramers-Kronig transformation to get Re(eps)

    binned_eps_re = kramerskronig(binned_eps_im)

    return binned_eps_re, binned_eps_im

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

    epsilon_im = prefactor*RPA_Im_eps_external_prefactor_LFE(qG, blocks, N_AO, q_cuts, unique_Ri) #(G, G', E)
    
    return epsilon_im

def RPA_Im_eps_external_prefactor_LFE(qG, blocks, N_AO, q_cuts, unique_Ri):
    """
    Returns |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2 summed over i,j,k

    Notes:
        -Speed up with numba? Is saving+loading data and using for loops or keeping data in memory and optimizing for loops with numba faster?

    Inputs: 
        qG:         np.ndarray, shape = (N_G, 3), q+G vectors
        blocks:     dict, atomic orbital blocks generated by get_basis_blocks
        N_AO:       int, number of atomic orbitals
        q_cuts:     np.ndarray, cut-off points in q for different R_ids.
    Outputs:
        im_eps:     Imaginary part of RPA dielectric function, 
                    shape = (N_G, N_G, int(parmt.E_max/parmt.dE + 1)), prefactor multiplied later.
    """
    # Load data
    working_dir = parmt.store + '/working_dir/'
    energy_arr = np.load(working_dir + 'im_energy.npy')
    k_f = np.load(working_dir + 'k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + 'mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + 'mo_coeff_i.npy')

    nk = len(energy_arr)
    nE = int(parmt.E_max/parmt.dE + 1)
    im_eps = np.zeros(qG.shape[0], qG.shape[0], int(parmt.E_max/parmt.dE + 1))
    if nk > 16:
        partitions = np.append(np.arange(0, nk, 16), [nk])
        for p in range(len(partitions) - 1):
            i, f = partitions[p], partitions[p+1]
            with mp.get_context('fork').Pool(mp.cpu_count()) as p:
                eta_qG = p.map(partial(eps.get_3D_overlaps_blocks, k_f=k_f[i:f], blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i:f], mo_coeff_f_conj=mo_coeff_f_conj[i:f], unique_Ri=unique_Ri, q_cuts=q_cuts), qG)
            eta_qG = np.array(eta_qG) / np.linalg.norm(qG, axis=1)[None, None, None, :]
            eta_qG_sq = np.einsum('kijg, kijh -> kijgh', eta_qG*eta_qG.conj())
            del eta_qG
            for k in range(i, f):
                for a in range(mo_coeff_i.shape[2]):
                    for b in range(mo_coeff_f_conj.shape[2]):
                        ind, rem = tuple(energy_arr[k, a, b])
                        if ind < nE:
                            im_eps[:,:,int(ind)] += rem*eta_qG_sq[k - i, a, b]
                        if ind < nE - 1:
                            im_eps[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[k - i, a, b]
    else:
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:
            eta_qG = p.map(partial(eps.get_3D_overlaps_blocks, k_f=k_f, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, unique_Ri=unique_Ri, q_cuts=q_cuts), qG)
        eta_qG = np.array(eta_qG) / np.linalg.norm(qG, axis=1)[None, None, None, :]
        eta_qG_sq = np.einsum('kijg, kijh -> kijgh', eta_qG*eta_qG.conj())
        del eta_qG
        for k in range(nk):
            for a in range(mo_coeff_i.shape[2]):
                for b in range(mo_coeff_f_conj.shape[2]):
                    ind, rem = tuple(energy_arr[k, a, b])
                    if ind < nE:
                        im_eps[:,:,int(ind)] += rem*eta_qG_sq[k, a, b]
                    if ind < nE - 1:
                        im_eps[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[k, a, b]
    return im_eps
