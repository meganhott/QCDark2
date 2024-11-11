import numpy as np
import time
import os
import psutil
import multiprocessing as mp
from functools import partial
from numba import njit, prange
from scipy.interpolate import LinearNDInterpolator

from routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import input_parameters as parmt
import binning as bin

from epsilon_routines import kramerskronig_im2re, kramerskronig_re2im, delta_energy
from epsilon_helper import get_3D_overlaps_blocks
from dielectric_functions import *

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

        eps_q_delta = get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, blocks, N_AO, q_cuts, VCell, G_q, unique_Ri) #(G,G',E)

        #Kramers-Kronig to get transition amplitude parts of Re(eps) and Im(eps)
        eps_q_ta = kramerskronig_lfe(eps_q_delta)

        eps_q =  eps_q_ta + 1j*eps_q_delta + np.identity(eps_q_delta.shape[0])[:,:,None]

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
        -Speed up with numba? Is saving+loading data and using for loops or keeping data in memory and optimizing for loops with numba faster?

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
            eta_qG_p = p.map(partial(get_3D_overlaps_blocks, k_f=k_f[i:f], blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i:f], mo_coeff_f_conj=mo_coeff_f_conj[i:f], unique_Ri=unique_Ri, q_cuts=q_cuts), qG) #(G,k,i,j)
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

@time_wrapper(n_tabs=2)
def kramerskronig_lfe(eps_delta):
    """
    eps_delta: (N_G, N_G, N_E)
    Parallelize over G1
    """
    eps_delta_re = np.real(eps_delta)
    eps_delta_im = np.imag(eps_delta)
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        eps_re_ta = p.map(kramerskronig_im2re, eps_delta_re)
        eps_im_ta = p.map(kramerskronig_re2im, eps_delta_im)
    eps_ta = np.array(eps_re_ta) - 1j*np.array(eps_im_ta)
    return eps_ta

def get_Nk(unique_Ri, blocks, N_AO):
    """
    Returns number of k pairs to include in 3D overlaps multiprocessing based on the available memory and number of cores
    """
    max_memory = psutil.virtual_memory().available*0.9
    init_memory = psutil.Process(os.getpid()).memory_info()[0] #memory currently used 
    N_cpus = mp.cpu_count()

    N_uniqueRi = sum([Ri.shape[0] for Ri in unique_Ri])
    N_PG = max(np.concatenate([k1 for k in blocks.keys() for k1 in k])) + 1
    N_max_block = max([len(k1) for k in blocks.keys() for k1 in k])
    N_Rid = np.load('test_resources/R_ids/-1.npy').shape[1]
    overlaps_memory = (N_uniqueRi*N_PG**2 + N_AO**2 + N_uniqueRi*N_PG*N_max_block + N_Rid*N_max_block**2 + N_uniqueRi*N_max_block**2) * 16
    
    Nk = int((max_memory - init_memory*N_cpus) / overlaps_memory / N_cpus)
    return Nk

def main():
    cell, dark_objects = initialize_cell()
    dark_objects = dielectric_RPA(cell, dark_objects)
    get_RPA_dielectric_LFE(dark_objects)

if __name__ == '__main__':
    main()
