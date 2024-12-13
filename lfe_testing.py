import numpy as np
import time
import os
import re
import psutil
import multiprocessing as mp
from functools import partial
from numba import njit
import numba as nb
from scipy.interpolate import LinearNDInterpolator

from routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
import input_parameters as parmt
import binning as bin

from epsilon_routines import delta_energy
from epsilon_helper import get_3D_overlaps_blocks, kramerskronig_im2re, kramerskronig_re2im
from dielectric_functions import *

#from line_profiler import profile
#from memory_profiler import profile

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
    makedir(parmt.store + '/working_dir/eps_delta')

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

    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0],N_AO,N_AO)), mo_coeff_i)[0] #generate optimal path for 3D overlaps calculation

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,i,j,2)
    del mo_en_i, mo_en_f

    # store relevant quantities, perhaps faster to load in each iteration than supply - still allocates memory for full coeff arrays when loading in k slices so makes more sense to keep full arrays here
    """
    np.save(working_dir + 'im_energy', im_delE)
    np.save(working_dir + 'k_f', k_f)
    np.save(working_dir + 'mo_coeff_i', mo_coeff_i)
    np.save(working_dir + 'mo_coeff_f_conj', mo_coeff_f_conj)
    del mo_coeff_i, mo_coeff_f_conj
    """

    qG = q[None, :] + G_q

    #create list of band ids -> eta_qG, im_delE
    # np.hstack([b[:,i] for i in range(b.shape[1])])
    #eta_qG = np.hstack([eta_qG[:,i] for i in range(eta_qG.shape[1])]) #collapsing i,j indices #(k,(i,j),G)
    #im_delE = np.hstack([im_delE[:,i] for i in range(im_delE.shape[1])]) #(k,(i,j),2)

    N_E = int(parmt.E_max/parmt.dE + 1)
    
    #Determine number of processes to run in parallel based on memory limitations
    N_p = get_mp_N(N_AO, k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], G_q.shape[0], N_E)
    """
    with mp.get_context('fork').Pool(N_p) as p:
        p.map(partial(get_eps_delta_k, k_f=k_f, qG=qG, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, blocks=blocks, unique_Ri=unique_Ri, q_cuts=q_cuts, einsum_path=einsum_path, im_delE=im_delE, N_AO=N_AO, N_E=N_E), range(k_f.shape[0]))
    """
    #get_eps_delta_k(0, k_f=k_f, qG=qG, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, blocks=blocks, unique_Ri=unique_Ri, q_cuts=q_cuts, einsum_path=einsum_path, im_delE=im_delE, N_AO=N_AO, N_E=N_E)

    start_time = time.time()
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex128')
    get_eps_delta_k(0, k_f, qG, mo_coeff_i, mo_coeff_f_conj, blocks, unique_Ri, q_cuts, einsum_path, im_delE, N_AO, N_E)
    #eta_qG = np.array(eta_qG).transpose((1,2,0)) #(G,i,j) -> (i,j,G)
    print('old 3D overlaps: ', time.time() - start_time)

    start_time = time.time()
    blocks = blocks_typed_dict2(blocks)
    print('blocks conversion: ', time.time() - start_time)

    start_time = time.time()

    #testing for first k
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex128')
    for i, qG_i in enumerate(qG):
        eta_qG[i] = get_3D_overlaps_k_full_numba(qG_i, k_f=k_f[0], blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[0], mo_coeff_f_conj=mo_coeff_f_conj[0], unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path)
    eta_qG = np.array(eta_qG).transpose((1,2,0)) #(G,i,j) -> (i,j,G)
    print('numba 3D overlaps: ', time.time() - start_time)


    exit()

    #read and add together all delta_eps(k) for given q
    eps_delta = np.zeros((qG.shape[0], qG.shape[0], N_E), dtype='complex')
    for i_k in range(k_f.shape[0]):
        eps_delta += np.load(parmt.store + '/working_dir/eps_delta/k_{}.npy'.format(i_k))

    # The above is actually slightly slower than implementation in epsilon_routines which calculates eps_delta in serial. It might be worth it to parallelize than calculation over a few k at a time and implement numba optimizations instead of multiprocessing whole calculation over k and saving to disk. for 512 k, eps_delta takes up 72 GB!


    '''
    for i_k, k in enumerate([k_f[0]]):
        
        #Calculate eta_qG for one k
        start_time = time.time()
        """
        with mp.get_context('spawn').Pool(mp.cpu_count()) as p:
            eta_qG = p.map(partial(lfe_testing_helper.get_3D_overlaps_k, k_f=k, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path), qG) #might want to run this in chunks so less initialization time
        """
        
        #numba experiment - does not work with multiprocessing
        """
        blocks = blocks_typed_dict(blocks) #convert to numba typed dictionary
        eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[1], mo_coeff_f_conj.shape[1]), dtype='complex128')
        start_time = time.time()
        for i, qG_i in enumerate(qG):
            eta_qG[i] = get_3D_overlaps_k_numba(qG_i, k_f=k, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path)
        print(f'eta_qG(k) calculated in {time.time() - start_time}s, numba')
        """
        eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex128')
        for i, qG_i in enumerate(qG):
            eta_qG[i] = get_3D_overlaps_k(qG_i, k_f=k, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i_k], mo_coeff_f_conj=mo_coeff_f_conj[i_k], unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path)
        eta_qG = np.array(eta_qG).transpose((1,2,0)) #(G,i,j) -> (i,j,G)
        print(f'eta_qG(k) calculated in {time.time() - start_time}s, serial')

        start_time = time.time()
        eps_delta = np.zeros((qG.shape[0], qG.shape[0], nE), dtype='complex')
        eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG, eta_qG.conj())
        for a in range(mo_coeff_i.shape[2]):
            for b in range(mo_coeff_f_conj.shape[2]):
                ind, rem = tuple(im_delE[i_k, a, b])
                if ind < N_E:
                    eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
                if ind < N_E - 1:
                    eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
        np.save(working_dir + '/eps_delta/k_' + str(i_k), eps_delta)
        print(f'eps_delta(k) calculated in {time.time() - start_time}s, serial')

        """
        exit()
        #Collapse (i,j) 
        eta_qG = np.stack([eta_qG[i,j] for i in range(eta_qG.shape[0]) for j in range(eta_qG.shape[1])]) #collapsing i,j indices #((i,j),G) #rewrite so don't have to transpose above

        ij = [(eta_qG_ij, im_delE_ij) for eta_qG_ij, im_delE_ij in zip(eta_qG, im_delE[i_k])] #create tuples for starmap
        start_time = time.time()
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:
            eps_delta_k = p.starmap(get_eps_delta_LFE, ij) #split over (i,j)
        eps_delta_k = np.sum(np.array(eps_delta_k), axis=0) #(G,G',E)
        print(f'eps_delta(k) calculated in {time.time() - start_time}s')

        eps_delta = eps_delta + eps_delta_k
        """
    '''
    eps_delta = prefactor*eps_delta
    
    return eps_delta

def get_eps_delta_k(i_k, k_f, qG, mo_coeff_i, mo_coeff_f_conj, blocks, unique_Ri, q_cuts, einsum_path, im_delE, N_AO, N_E):
    #Calculate eta_qG for one k
    k = k_f[i_k]
    start_time = time.time()
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex128')
    for i, qG_i in enumerate(qG):
        eta_qG[i] = get_3D_overlaps_k(qG_i, k_f=k, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i_k], mo_coeff_f_conj=mo_coeff_f_conj[i_k], unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path)
    eta_qG = np.array(eta_qG).transpose((1,2,0)) #(G,i,j) -> (i,j,G)
    print(f'eta_qG(k) calculated in {time.time() - start_time}s, serial')

    start_time = time.time()
    eps_delta = np.zeros((qG.shape[0], qG.shape[0], N_E), dtype='complex')
    eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG, eta_qG.conj())
    for a in range(mo_coeff_i.shape[2]):
        for b in range(mo_coeff_f_conj.shape[2]):
            ind, rem = tuple(im_delE[i_k, a, b])
            if ind < N_E:
                eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
            if ind < N_E - 1:
                eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
    np.save(parmt.store + '/working_dir/eps_delta/k_{}.npy'.format(i_k), eps_delta)
    print(f'eps_delta(k) calculated in {time.time() - start_time}s, serial')

def get_3D_overlaps_k(qG: np.ndarray, k_f: np.ndarray, blocks: dict, N_AO: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, unique_Ri:list[np.ndarray], q_cuts: np.ndarray, path) -> np.ndarray:
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Inputs:
        qG:             np.ndarray of shape (3,): q+G
        k_f:            np.ndarray of shape (3, ): a single k' corresponding to q
        blocks:         dict: atomic orbital blocks generated by get_basis_blocks 
        N_AO:           int: number of atomic orbitals
        mo_coeff_i:     np.ndarray of shape (N_AO, N_valbands)
        mo_coeff_f:     np.ndarray of shape (N_AO, N_conbands)
        unique_Ri:      list of [R_unique_x, R_unique_y, R_unique_z)
        q_cuts:         np.ndarray of shape (N_q, ): |q+G| at which R_cutoff change
    Outputs:
        eta_qG:         np.ndarray of shape (N_val_bands (i), N_cond_bands (j)): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): 
        ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)

            tot = np.ones((R_id.shape[1], p2.shape[1], p1.shape[1]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][:,p2[d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)
            for i in d1:
                for j in d2:
                    ovlp[i,j] = (tot@d1[i])@d2[j]
            
    return np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0] #This is faster with added k index for some reason??

def get_mp_N(N_AO, N_k, N_val, N_con, N_qG, N_E):
    """
    Returns number of simultaneous processes to run for eps_delta calculation based on the available memory and number of cores
    """
    M_max = psutil.virtual_memory().available*0.9
    M_init = psutil.Process(os.getpid()).memory_info()[0] #memory currently used 
    N_cpus = mp.cpu_count()

    N_mo_coeff_i = N_AO * N_val * N_k
    N_mo_coeff_f = N_AO * N_con * N_k
    N_im_delE = N_k * N_val * N_con * 2
    N_eta_qG = N_qG * N_val * N_con
    N_eta_qG_sq = N_eta_qG * N_qG
    N_eps_delta = N_qG * N_qG * N_E

    M_p = (N_mo_coeff_i + N_mo_coeff_f + N_im_delE + N_eta_qG + N_eta_qG_sq + N_eps_delta) * 16 + M_init #memory for one process in bytes
    N_p = min(int(M_max / M_p), N_cpus) #number of processes to run in parallel

    logger.info('\t\t\tUsing {} cores with memory per process = {:.1f} GiB'.format(N_p, M_p/2**30))
    return N_p

def get_eps_delta_LFE(eta_qG_kij, im_delE_kij):
    """
    Computes delta part of epsilon for all G vectors for one k and one (i,j)
    eta_qG_kij: (G, )
    im_delE_kij: (2, )
    """
    N_E = int(parmt.E_max/parmt.dE + 1)
    N_G = eta_qG_kij.shape[0]
    eps_delta = np.zeros((N_G, N_G, N_E), dtype='complex') #(G,G',E)

    eta_qG_sq = np.einsum('g,h -> gh', eta_qG_kij, eta_qG_kij.conj()) #optimize

    ind, rem = tuple(im_delE_kij)
    if ind < N_E:
        eps_delta[:,:,int(ind)] += rem*eta_qG_sq #might be faster to have eps_delta be (E,G,G)?
    if ind < N_E - 1:
        eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq
    return eps_delta #(G,G',E)

def get_3D_overlaps_LFE(qG, blocks, N_AO, q_cuts, unique_Ri):
    # Load data
    working_dir = parmt.store + '/working_dir/'
    energy_arr = np.load(working_dir + 'im_energy.npy')
    k_f = np.load(working_dir + 'k_f.npy')
    mo_coeff_f_conj = np.load(working_dir + 'mo_coeff_f_conj.npy')
    mo_coeff_i = np.load(working_dir + 'mo_coeff_i.npy')

    nk = len(energy_arr)
    nE = int(parmt.E_max/parmt.dE + 1)
    im_eps = np.zeros(int(parmt.E_max/parmt.dE + 1))
    if nk > 16:
        eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex')
        partitions = np.append(np.arange(0, nk, 16), [nk])
        for p in range(len(partitions) - 1):
            i, f = partitions[p], partitions[p+1]
            eta_qG[i:f] = get_3D_overlaps_blocks(qG, k_f[i:f], blocks, N_AO, mo_coeff_i[i:f], mo_coeff_f_conj[i:f], unique_Ri, q_cuts)
    else:
        eta_qG = get_3D_overlaps_blocks(qG, k_f, blocks, N_AO, mo_coeff_i, mo_coeff_f_conj, unique_Ri, q_cuts)

    return eta_qG / np.linalg.norm(qG) #(k,i,j)

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

    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]), dtype='complex') #(k,i,j,G)

    Nk, Np = get_mp_N(unique_Ri, blocks, N_AO, mo_coeff_i.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]) #number of k such that calculation of eta(G,k,i,j) can be run in parallel without exceeding memory  

    #print(psutil.Process(os.getpid()).memory_info()[0]/2**20)
    """
    start_time = time.time()
    partitions = np.append(np.arange(0, k_f.shape[0], Nk), [k_f.shape[0]])
    for p in range(len(partitions) - 1):
        i, f = partitions[p], partitions[p+1]
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:
            eta_qG_p = p.map(partial(get_3D_overlaps_blocks, k_f=k_f[i:f], blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i:f], mo_coeff_f_conj=mo_coeff_f_conj[i:f], unique_Ri=unique_Ri, q_cuts=q_cuts), qG) #(G,k,i,j)
        eta_qG[i:f] = np.array(eta_qG_p).transpose((1,2,3,0))
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:] #(k,i,j,G)

    logger.info('\t\t\t3D overlaps loaded with {} k partitions. Time taken = {:.2f} s.'.format(len(partitions)-1, time.time()-start_time))
    """
    eta_qG = np.load(parmt.store+'/eta_qG.npy')
    start_time = time.time()
    """
    nE = int(parmt.E_max/parmt.dE + 1)
    eps_delta = np.zeros((qG.shape[0], qG.shape[0], int(parmt.E_max/parmt.dE + 1)), dtype='complex')
    for i_k in range(k_f.shape[0]):
        eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG[i_k], eta_qG[i_k].conj())
        for a in range(mo_coeff_i.shape[2]):
            for b in range(mo_coeff_f_conj.shape[2]):
                ind, rem = tuple(energy_arr[i_k, a, b])
                if ind < nE:
                    eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
                if ind < nE - 1:
                    eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
    """
    """
    Np = min(Np, mp.cpu_count())
    print(Np)
    partitions = np.append(np.arange(0, k_f.shape[0], Np), [k_f.shape[0]])
    eps_delta_tot = np.zeros((eta_qG.shape[3], eta_qG.shape[3], int(parmt.E_max/parmt.dE + 1)), dtype='complex')
    for p in [0]: #range(len(partitions) - 1):
        i, f = partitions[p], partitions[p+1]
        start_time = time.time()
        with mp.get_context('fork').Pool(Np) as p:
            eps_delta = p.map(partial(get_eps_delta, eta_qG=eta_qG, energy_arr=energy_arr), range(8))
        #eps_delta = np.sum(eps_delta, axis=0)
        #eps_delta_tot += eps_delta
    """
    #Rewrite
    
    Np = min(Np, mp.cpu_count())
    partitions = np.append(np.arange(0, k_f.shape[0], int(k_f.shape[0]/Np)), [k_f.shape[0]])
    partitions = np.array([partitions[:-1], partitions[1:]]).transpose() #(k_ni, k_n(i+1))

    start_time = time.time()
    with mp.get_context('fork').Pool(Np) as p:
        eps_delta = p.map(partial(get_eps_delta, eta_qG=eta_qG, energy_arr=energy_arr), partitions)
    print(time.time()-start_time)
    start_time = time.time()
    eps_delta = np.array(eps_delta)
    eps_delta = np.sum(eps_delta, axis=0)
    print(time.time()-start_time)
    logger.info('\t\t\tDelta part of epsilon calculated. Time taken = {:.2f} s.'.format(time.time()-start_time))
    exit()

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

'''
def get_mp_N(unique_Ri, blocks, N_AO, Nk_tot, N_val, N_con, N_qG):
    """
    Returns number of k pairs to include in 3D overlaps multiprocessing and number of simultaneous processes to run for eps_delta calculation based on the available memory and number of cores
    """
    max_memory = psutil.virtual_memory().available*0.9
    init_memory = psutil.Process(os.getpid()).memory_info()[0] #memory currently used 
    N_cpus = mp.cpu_count()

    N_uniqueRi = sum([Ri.shape[0] for Ri in unique_Ri])
    N_PG = max(np.concatenate([k1 for k in blocks.keys() for k1 in k])) + 1
    N_max_block = max([len(k1) for k in blocks.keys() for k1 in k])
    N_Rid = np.load(parmt.store + '/R_ids/-1.npy').shape[1]
    overlaps_memory = (N_uniqueRi*N_PG**2 + N_AO**2 + N_uniqueRi*N_PG*N_max_block + N_Rid*N_max_block**2 + N_uniqueRi*N_max_block**2) * 16
    
    Nk = int((max_memory - init_memory*N_cpus) / overlaps_memory / N_cpus)

    eps_delta_memory = (Nk_tot*N_val*N_con*N_qG + N_val*N_con*N_qG**2 + 2*N_qG**2*int(parmt.E_max/parmt.dE + 1)) * 16 + init_memory
    Np = int(max_memory/eps_delta_memory)
    return Nk, Np
'''
'''
# multiprocessing over k experiment vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

@time_wrapper(n_tabs=2)
def get_RPA_dielectric_LFE_q_k_parallel(q: np.ndarray, mo_en_f: np.ndarray, mo_en_i: np.ndarray, mo_coeff_f_conj: np.ndarray, mo_coeff_i: np.ndarray, k_f: np.ndarray, k_pairs: np.ndarray, blocks: dict, N_AO: int, q_cuts: np.ndarray, VCell: float, G_q: np.ndarray, unique_Ri: list[np.ndarray]) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0],N_AO,N_AO)), mo_coeff_i)[0] #generate optimal path for 3D overlaps calculation

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,i,j,2)
    del mo_en_i, mo_en_f

    qG = q[None, :] + G_q

    N_E = int(parmt.E_max/parmt.dE + 1)
    
    #Determine number of processes to run in parallel based on memory limitations
    N_p = get_mp_N_k_parallel(N_AO, k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], G_q.shape[0], N_E)
    with mp.get_context('fork').Pool(N_p) as p:
        p.map(partial(get_eps_delta_k, k_f=k_f, qG=qG, mo_coeff_i=mo_coeff_i, mo_coeff_f_conj=mo_coeff_f_conj, blocks=blocks, unique_Ri=unique_Ri, q_cuts=q_cuts, einsum_path=einsum_path, im_delE=im_delE, N_AO=N_AO, N_E=N_E), range(k_f.shape[0]))

    #read and add together all delta_eps(k) for given q
    eps_delta = np.zeros((qG.shape[0], qG.shape[0], N_E), dtype='complex')
    for i_k in range(k_f.shape[0]):
        eps_delta += np.load(parmt.store + '/working_dir/eps_delta/k_{}.npy'.format(i_k))

    # The above is actually slightly slower than implementation in epsilon_routines which calculates eps_delta in serial. It might be worth it to parallelize than calculation over a few k at a time and implement numba optimizations instead of multiprocessing whole calculation over k and saving to disk. for 512 k, eps_delta takes up 72 GB!

    eps_delta = prefactor*eps_delta
    
    return eps_delta

def get_mp_N_k_parallel(N_AO, N_k, N_val, N_con, N_qG, N_E):
    """
    Returns number of simultaneous processes to run for eps_delta calculation based on the available memory and number of cores
    """
    M_max = psutil.virtual_memory().available*0.9
    M_init = psutil.Process(os.getpid()).memory_info()[0] #memory currently used 
    N_cpus = mp.cpu_count()

    N_mo_coeff_i = N_AO * N_val * N_k
    N_mo_coeff_f = N_AO * N_con * N_k
    N_im_delE = N_k * N_val * N_con * 2
    N_eta_qG = N_qG * N_val * N_con
    N_eta_qG_sq = N_eta_qG * N_qG
    N_eps_delta = N_qG * N_qG * N_E

    M_p = (N_mo_coeff_i + N_mo_coeff_f + N_im_delE + N_eta_qG + N_eta_qG_sq + N_eps_delta) * 16 + M_init #memory for one process in bytes
    N_p = min(int(M_max / M_p), N_cpus) #number of processes to run in parallel

    logger.info('\t\t\tUsing {} cores with memory per process = {:.1f} GiB'.format(N_p, M_p/2**30))
    return N_p

def get_eps_delta_k(i_k, k_f, qG, mo_coeff_i, mo_coeff_f_conj, blocks, unique_Ri, q_cuts, einsum_path, im_delE, N_AO, N_E):
    #Calculate eta_qG for one k
    k = k_f[i_k]
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2]), dtype='complex128')
    for i, qG_i in enumerate(qG):
        eta_qG[i] = get_3D_overlaps_k(qG_i, k_f=k, blocks=blocks, N_AO=N_AO, mo_coeff_i=mo_coeff_i[i_k], mo_coeff_f_conj=mo_coeff_f_conj[i_k], unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path)
    eta_qG = np.array(eta_qG).transpose((1,2,0)) #(G,i,j) -> (i,j,G)

    eps_delta = np.zeros((qG.shape[0], qG.shape[0], N_E), dtype='complex')
    eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG, eta_qG.conj())
    for a in range(mo_coeff_i.shape[2]):
        for b in range(mo_coeff_f_conj.shape[2]):
            ind, rem = tuple(im_delE[i_k, a, b])
            if ind < N_E:
                eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
            if ind < N_E - 1:
                eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
    np.save(parmt.store + '/working_dir/eps_delta/k_{}.npy'.format(i_k), eps_delta)

def get_3D_overlaps_k(qG: np.ndarray, k_f: np.ndarray, blocks: dict, N_AO: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, unique_Ri:list[np.ndarray], q_cuts: np.ndarray, path) -> np.ndarray:
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Inputs:
        qG:             np.ndarray of shape (3,): q+G
        k_f:            np.ndarray of shape (3, ): a single k' corresponding to q
        blocks:         dict: atomic orbital blocks generated by get_basis_blocks 
        N_AO:           int: number of atomic orbitals
        mo_coeff_i:     np.ndarray of shape (N_AO, N_valbands)
        mo_coeff_f:     np.ndarray of shape (N_AO, N_conbands)
        unique_Ri:      list of [R_unique_x, R_unique_y, R_unique_z)
        q_cuts:         np.ndarray of shape (N_q, ): |q+G| at which R_cutoff change
    Outputs:
        eta_qG:         np.ndarray of shape (N_val_bands (i), N_cond_bands (j)): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): 
        ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)

            tot = np.ones((R_id.shape[1], p2.shape[1], p1.shape[1]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][:,p2[d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)
            for i in d1:
                for j in d2:
                    ovlp[i,j] = (tot@d1[i])@d2[j]
            
    return np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0] #This is faster with added k index for some reason??

# multiprocessing over k experiment ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''


# numba experiments vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def get_3D_overlaps_k_numba(qG: np.ndarray, k_f: np.ndarray, blocks: dict, N_AO: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, unique_Ri:list[np.ndarray], q_cuts: np.ndarray, path) -> np.ndarray:
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Notes:
    - Calculating ovlp with numba speeds up this function by ~30-40% but we cannot have a numba-typed blocks dict sent to multiprocessing, so this must be run in serial since it takes way to long to convert blocks each time. Overall, multiprocessing without numba is faster

    Inputs:
        qG:             np.ndarray of shape (3,): q+G
        k_f:            np.ndarray of shape (3, ): a single k' corresponding to q
        blocks:         dict: atomic orbital blocks generated by get_basis_blocks 
        N_AO:           int: number of atomic orbitals
        mo_coeff_i:     np.ndarray of shape (N_AO, N_valbands)
        mo_coeff_f:     np.ndarray of shape (N_AO, N_conbands)
        unique_Ri:      list of [R_unique_x, R_unique_y, R_unique_z)
        q_cuts:         np.ndarray of shape (N_q, ): |q+G| at which R_cutoff change
    Outputs:
        eta_qG:         np.ndarray of shape (N_val_bands (i), N_cond_bands (j)): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): 
        ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)
            ovlp = ovlp_sum(ovlp, d1, p1, d2, p2, ints_i[0], ints_i[1], ints_i[2], R_id) #numba optimized function
            
    return np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0] #This is faster with added k index for some reason??

def get_3D_overlaps_k_full_numba(qG: np.ndarray, k_f: np.ndarray, blocks: dict, N_AO: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, unique_Ri:list[np.ndarray], q_cuts: np.ndarray, path) -> np.ndarray:
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Notes:
    - Calculating ovlp with numba speeds up this function by ~30-40% but we cannot have a numba-typed blocks dict sent to multiprocessing, so this must be run in serial since it takes way to long to convert blocks each time. Overall, multiprocessing without numba is faster

    Inputs:
        qG:             np.ndarray of shape (3,): q+G
        k_f:            np.ndarray of shape (3, ): a single k' corresponding to q
        blocks:         dict: atomic orbital blocks generated by get_basis_blocks 
        N_AO:           int: number of atomic orbitals
        mo_coeff_i:     np.ndarray of shape (N_AO, N_valbands)
        mo_coeff_f:     np.ndarray of shape (N_AO, N_conbands)
        unique_Ri:      list of [R_unique_x, R_unique_y, R_unique_z)
        q_cuts:         np.ndarray of shape (N_q, ): |q+G| at which R_cutoff change
    Outputs:
        eta_qG:         np.ndarray of shape (N_val_bands (i), N_cond_bands (j)): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): 
        ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
    ovlp = ovlp_sum_full(blocks, ints[0], ints[1], ints[2], R_id, N_AO) #numba optimized function

    return np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0] #This is faster with added k index for some reason??


@njit(nb.types.Array(nb.types.complex128, 2, 'C')(nb.types.DictType(nb.types.unicode_type, nb.types.DictType(nb.types.int64, nb.types.Array(nb.types.complex128, 1, 'C'))), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.int64, 2, 'F'), nb.types.int64))
def ovlp_sum_full(blocks, ints_x, ints_y, ints_z, R_id, N_AO):
    def str_to_int(s): #https://github.com/numba/numba/issues/5650
        final_index, result = len(s) - 1, 0
        for i,v in enumerate(s):
            result += (ord(v) - 48) * (10 ** (final_index - i))
        return result

    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 

    for p1 in blocks:
        d1 = blocks[p1]
        p1 = [[b.replace(',', '') for b in p1.split('), ')[i].replace('(', '').replace(')', '').split(', ')] for i in range(3)] #now have to transform this from string 
        p1 = [[str_to_int(b)for b in p1[i]] for i in range(3)]
        p1 = np.array(p1) #have to do this is multiple steps for numba otherwise memory issues

        ints_i_x = ints_x[:,:,p1[0]] #numba hates lists
        ints_i_y = ints_y[:,:,p1[1]]
        ints_i_z = ints_z[:,:,p1[2]]

        for p2 in blocks:
            d2 = blocks[p2]
            p2 = [[b.replace(',', '') for b in p2.split('), ')[i].replace('(', '').replace(')', '').split(', ')] for i in range(3)] #now have to transform this from string 
            p2 = [[str_to_int(b)for b in p2[i]] for i in range(3)]
            p2 = np.array(p2)

            tot = np.ones((R_id.shape[1], p2.shape[1], p1.shape[1]), dtype = np.complex128)
            for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
                ints_ij = ints_i[:,p2[d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)

            for i in d1:
                for j in d2:
                    ovlp[i,j] = (tot@d1[i])@d2[j]

    return ovlp


#make dicts within blocks into numba typed dicts
def blocks_typed_dict(blocks):
    for p in blocks:
        d = blocks[p] #untyped dict
        typed_d = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.Array(dtype=nb.types.complex128, ndim=1, layout='C'))
        for p1 in d:
            typed_d[p1] = d[p1].astype('complex128') #need to be complex when multiplying against 1D integrals for numba optimization
        blocks[p] = typed_d #replace with typed dict
    return blocks

def blocks_typed_dict2(blocks):
    # Make whole blocks dict into numba typed dict
    # To get tuples (now arrays) keys back: [[int(c) for c in re.findall('\d+', b)] for b in a.split('),')] where a is a tuple string
    start_time = time.time()
    blocks_typed = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.DictType(nb.types.int64, nb.types.Array(dtype=nb.types.complex128, ndim=1, layout='C')))
    for p in blocks:
        d = blocks[p] #untyped dict
        typed_d = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.Array(dtype=nb.types.complex128, ndim=1, layout='C'))
        for p1 in d:
            typed_d[p1] = d[p1].astype('complex128') #need to be complex when multiplying against 1D integrals for numba optimization
        blocks_typed[str(p)] = typed_d #replace with typed dict
    print(time.time()-start_time)
    return blocks_typed

def get_blocks():
    cell, dark_objects = initialize_cell()
    blocks = dark_objects['blocks']
    return blocks

#njit main part of 3D overlaps code
@njit(nb.types.Array(nb.types.complex128, 2, 'C')(nb.types.Array(nb.types.complex128, 2, 'C'), nb.types.DictType(nb.types.int64, nb.types.Array(nb.types.complex128, 1, 'C')), nb.types.Array(nb.types.int64, 2, 'C'), nb.types.DictType(nb.types.int64, nb.types.Array(nb.types.complex128, 1, 'C')), nb.types.Array(nb.types.int64, 2, 'C'), nb.types.Array(nb.types.complex128, 3, 'A'), nb.types.Array(nb.types.complex128, 3, 'A'), nb.types.Array(nb.types.complex128, 3, 'A'), nb.types.Array(nb.types.int64, 2, 'F')))
def ovlp_sum(ovlp, d1, p1, d2, p2, ints_i_x, ints_i_y, ints_i_z, R_id): #have to split up ints_i because numba hates lists
    tot = np.ones((R_id.shape[1], p2.shape[1], p1.shape[1]), dtype = np.complex128)
    for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
        ints_ij = ints_i[:,p2[d]]
        tot *= ints_ij[R_id[d]]
    tot = tot.sum(axis = 0)

    for i in d1:
        for j in d2:
            ovlp[i,j] = (tot@d1[i])@d2[j]
    return ovlp

# numba experiments ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def main():
    cell, dark_objects = initialize_cell()
    new_dft, dft_params = dft_routines.save_dft()
    dark_objects = dielectric_RPA(cell, dark_objects, dft_params)
    get_RPA_dielectric_LFE(dark_objects)

if __name__ == '__main__':
    main()
