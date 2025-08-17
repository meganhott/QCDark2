import numpy as np
import time
import os
import psutil
import shutil # for removing working directories after calculation is finished
import multiprocessing as mp
from functools import partial
from numba import njit
import h5py

from dielectric_pyscf.routines import logger, time_wrapper, load_unique_R, makedir, alpha, me
from dielectric_pyscf.epsilon_utils import epsilon_r, interp_eps
import dielectric_pyscf.input_parameters as parmt
import dielectric_pyscf.epsilon_helper as eps
import dielectric_pyscf.binning as binning
import dielectric_pyscf.kramers_kronig as kk


def get_RPA_dielectric(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    if parmt.include_lfe:
        tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_LFE(dark_objects, rank, q_start, q_stop)
    else:
        if parmt.dir_1d is not None: # only compute along one direction
            tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_no_LFE_1d(dark_objects, rank, q_start, q_stop)
        elif parmt.binning_1d:
            tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_no_LFE_1d_binning(dark_objects, rank, q_start, q_stop)
        else:
            tot_bin_eps, tot_bin_weights, bin_centers = get_RPA_dielectric_no_LFE(dark_objects, rank, q_start, q_stop)

    return tot_bin_eps, tot_bin_weights, bin_centers

@time_wrapper
def get_RPA_dielectric_no_LFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
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
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
    N_ang_bins = parmt.N_phi*(parmt.N_theta-2) + 2

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eps_im')

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q
    
    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(parmt.E_max/parmt.dE)+1))
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
            logger.info(f'\t\tNumber of G vectors = {len(G_q)},')

        # Imaginary part of polarizability for E >= 0
        if rank == 0 or rank == None:
            logger.info(f'\t\tStarting calculation of Im(eps) for 0 < E <= {parmt.E_max} eV')
        start_time1 = time.time()

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

        if rank == 0 or rank == None:
            logger.info(f'\t\tFinished calculation of Im(eps). Time taken = {(time.time() - start_time1):.2f} s')

        start_time1 = time.time()
        tot_bin_eps_im, tot_bin_weights = binning.bin_eps_q(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)

        if rank == 0 or rank == None:
            logger.info(f'\t\tIm(eps) binned. Time taken = {(time.time() - start_time1):.2f} s.')
            logger.info(f'\t\tCompleted i_q = {i_q + 1}. Time taken = {(time.time() - start_time):.2f} s.')

        # Save to working directory
        np.save(f'{working_dir}/tot_bin_eps_im.npy', tot_bin_eps_im)
        np.save(f'{working_dir}/tot_bin_weights.npy', tot_bin_weights)

    # Removing extra bins
    tot_bin_eps_im = tot_bin_eps_im[:-N_ang_bins, :]
    tot_bin_weights = tot_bin_weights[:-N_ang_bins]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    #shutil.rmtree(working_dir) # delete working directory

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper
def get_RPA_dielectric_no_LFE_1d(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
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

    dir_1d = parmt.dir_1d
    dir_sph = binning.cartesian_to_spherical(np.array([dir_1d]))[0]

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi, dir=True)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eps_im')

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q
    
    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+1, int(parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+1)

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(unique_q[q])
        q = np.array(q)

        if rank == 0 or rank == None:
            logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {np.array2string(q, precision=5)},')

        start_time = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]

        qpG_sph = binning.cartesian_to_spherical(q + G_q)
        if parmt.dir_1d_exact_angle: # exactly match angle
            G_q = G_q[(np.round(qpG_sph[:,1],4) == np.round(dir_sph[1], 4)) & (np.round(qpG_sph[:,2],4) == np.round(dir_sph[2], 4))] # select G along specified direction
        else: # just match within N_theta and N_phi accuracy
            costheta_l = np.cos(dir_sph[1] + np.pi/parmt.N_theta/2)
            costheta_g = np.cos(dir_sph[1] - np.pi/parmt.N_theta/2)

            if np.round(costheta_g, 4) == np.round(costheta_l, 4):
                costheta_g = 1 #for [0,0,1] - make more general later

            phi_l = dir_sph[2] - np.pi/parmt.N_phi
            phi_g = dir_sph[2] + np.pi/parmt.N_phi

            G_q = G_q[((qpG_sph[:,2] < phi_g) & (qpG_sph[:,2] > phi_l) & (np.cos(qpG_sph[:,1]) <= costheta_g) & (np.cos(qpG_sph[:,1]) >= costheta_l))]

        if rank == 0 or rank == None:
            logger.info(f'\t\tNumber of G vectors = {len(G_q)},')

        if len(G_q) != 0:
            # Imaginary part of polarizability for E >= 0
            if rank == 0 or rank == None:
                logger.info(f'\t\tStarting calculation of Im(eps) for 0 < E <= {parmt.E_max} eV')
            start_time1 = time.time()

            eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

            if rank == 0 or rank == None:
                logger.info(f'\t\tFinished calculation of Im(eps). Time taken = {(time.time() - start_time1):.2f} s')

            start_time1 = time.time()
            tot_bin_eps_im, tot_bin_weights = binning.bin_eps_q_1d(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)

            if rank == 0 or rank == None:
                logger.info(f'\t\tIm(eps) binned. Time taken = {(time.time() - start_time1):.2f} s.')
                logger.info(f'\t\tCompleted i_q = {i_q + 1}. Time taken = {(time.time() - start_time):.2f} s.')

            # Save to working directory
            np.save(f'{working_dir}/tot_bin_eps_im.npy', tot_bin_eps_im)
            np.save(f'{working_dir}/tot_bin_weights.npy', tot_bin_weights)

    # Removing extra bins
    tot_bin_eps_im = tot_bin_eps_im[:-1, :]
    tot_bin_weights = tot_bin_weights[:-1]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper
def get_RPA_dielectric_no_LFE_1d_binning(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
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
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi, dir=True)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]
    
    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eps_im')

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q
    
    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+1, int(parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+1)

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
            logger.info(f'\t\tNumber of G vectors = {len(G_q)},')

        # Imaginary part of polarizability for E >= 0
        if rank == 0 or rank == None:
            logger.info(f'\t\tStarting calculation of Im(eps) for 0 < E <= {parmt.E_max} eV')
        start_time1 = time.time()

        eps_q_im = get_RPA_dielectric_no_LFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

        if rank == 0 or rank == None:
            logger.info(f'\t\tFinished calculation of Im(eps). Time taken = {(time.time() - start_time1):.2f} s')

        start_time1 = time.time()
        tot_bin_eps_im, tot_bin_weights = binning.bin_eps_q_1d(q, G_q, eps_q_im, bin_centers, tot_bin_eps_im, tot_bin_weights)

        if rank == 0 or rank == None:
            logger.info(f'\t\tIm(eps) binned. Time taken = {(time.time() - start_time1):.2f} s.')
            logger.info(f'\t\tCompleted i_q = {i_q + 1}. Time taken = {(time.time() - start_time):.2f} s.')

        # Save to working directory
        np.save(f'{working_dir}/tot_bin_eps_im.npy', tot_bin_eps_im)
        np.save(f'{working_dir}/tot_bin_weights.npy', tot_bin_weights)

    # Removing extra bins
    tot_bin_eps_im = tot_bin_eps_im[:-1, :]
    tot_bin_weights = tot_bin_weights[:-1]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

def save_eps(bin_eps, bin_weights, bin_centers):
    f = h5py.File(parmt.store + '/epsilon.hdf5', 'r+') # open hdf5 file

    binned_eps = bin_eps/bin_weights[:, None]
    binned_eps_im = np.imag(binned_eps)
    binned_eps_im_interp = interp_eps(bin_centers, binned_eps_im)

    if parmt.include_lfe:
        kk_function = kk.kramerskronig_im2re
    else:
        kk_function = kk.kramerskronig_im2re_causal

    binned_eps_interp = kk_function(binned_eps_im_interp, parmt.E_max, parmt.dE) + 1. + 1j*binned_eps_im_interp

    f.create_dataset('binned_eps', data=bin_eps) #before interpolation
    if parmt.dir_1d == None:
        if parmt.binning_1d:
            eps_r = binned_eps_interp
        else:
            if parmt.save_3d: # Saves 3-dimensional binned epsilon
                f.create_dataset('binned_eps_interp', data=binned_eps_interp)
                f.create_dataset('bin_centers', data=bin_centers)
                f.create_dataset('bin_weights', data=bin_weights)

            eps_r = epsilon_r(bin_centers, binned_eps_interp) # angular average, for -E_max to E_max
    else:
        eps_r = binned_eps_interp

    f.create_dataset('epsilon_all', data=eps_r)

    # Add attributes from input parameters
    for name, val in parmt.__dict__.items():
        if not name.startswith('__'): # ignores dunders
            f.attrs[name] = str(val)

    # Add q and E arrays
    q = np.arange(parmt.q_min + 0.5*parmt.dq, parmt.q_max + parmt.dq, parmt.dq)
    E = np.arange(0, parmt.E_max + parmt.dE, parmt.dE)
    
    f.create_dataset('q', data=q)
    f.create_dataset('E', data=E)

    f.create_dataset('epsilon', data=eps_r[:,-E.shape[0]:]) # save only for positive energies

    f.close() # Close hdf5 file

    logger.info(f'Calculation done: dielectric function and input parameters stored as hdf5 file at {parmt.store}/epsilon.hdf5')

def get_RPA_dielectric_no_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) -> np.ndarray:
    
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,a,b)

    qG = q[None, :] + G_q

    epsilon_im = prefactor*RPA_Im_eps_external_prefactor_no_LFE(qG, mo_coeff_f_conj, mo_coeff_i, k_f, im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank)

    return epsilon_im

def RPA_Im_eps_external_prefactor_no_LFE(qG, mo_coeff_f_conj, mo_coeff_i, k_f, im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank):
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
    # Make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    # Save Im(eps) for each k, then load and combine after calculating for all k
    start_time = time.time()
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
        p.starmap(partial(eps.get_eps_im_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, im_delE=im_delE, working_dir=working_dir), k_tup)

    if rank == None or rank == 0:
        logger.info(f'\t\t\tIm(eps) calculated and saved for all k. Time taken = {(time.time() - start_time):.2f} s.')

    #eps_im = np.sum(np.array(eps_im), axis=0) #(E,G): sum over k
    
    # Load eps_im to memory
    start_time = time.time()
    N_E = int(parmt.E_max/parmt.dE + 1)
    eps_im = np.zeros((qG.shape[0], N_E), dtype='float')
    for i_k in range(k_f.shape[0]):
        eps_im += np.load(working_dir + f'/eps_im/eps_im_k{i_k}.npy')

    if rank == None or rank == 0:
        logger.info(f'\t\t\tIm(eps) loaded to memory and summed over k. Time taken = {(time.time() - start_time):.2f} s.')

    return eps_im #(G,E)

def get_RPA_dielectric_LFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
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

    N_E = int(parmt.E_max/parmt.dE*2 + 1) # number of energies to calculate eps at. This in includes -E_max <= E <= E_max

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
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

        start_time_full = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        N_G = G_q.shape[0]

        if rank == 0 or rank == None:
            logger.info(f'\t\tnumber of G vectors = {len(G_q)},')

        #Creating hdf5 file to store large intermediate array
        eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'w') #eps_delta with be stored to hdf5 for each energy
        # Determining chunks used to store data
        """
        chunk_size = 2**30 #1 GB chunks
        N_chunks = 16 * N_E * N_G**2 / chunk_size
        E_per_chunk = np.floor(N_E / np.sqrt(N_chunks))
        G_per_chunk = np.floor(N_G / np.sqrt(N_chunks))
        """
        chunk_slice_size = 2**30 # want to read/write in 1 GB chunks later
        E_per_chunk = int(np.floor(chunk_slice_size / (16 * N_G**2)))
        G_per_chunk = int(np.floor(np.sqrt(chunk_slice_size / (16 * N_E))))
        if G_per_chunk > N_G: 
            if E_per_chunk > N_E:
                eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex') # no chunking
                E_per_chunk = N_E
                G_per_chunk = N_G
            else:
                if E_per_chunk < 1:
                    E_per_chunk = 1
                eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, N_G, N_G)) # only chunking along E
                G_per_chunk = N_G
        else:
            eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, N_G, G_per_chunk))

        if rank == 0 or rank == None:
            logger.info(f'\t\tAn hdf5 dataset has been created to store intermediate results. The shape is (N_E={N_E}, N_G={N_G}, N_G={N_G}) saved in chunks of ({E_per_chunk}, {G_per_chunk}, {G_per_chunk}).')

        eps_delta_h5.close()

        get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) # calculate and store delta part of polarizability (E,G,G')

        #Perform Kramers-Kronig transformation to get transition amplitude parts of Re(eps) and Im(eps)
        N_G_chunks = int(np.ceil(N_G/G_per_chunk))
        if rank == 0 or rank == None:
            logger.info(f"\t\tBeginning Kramers-Kronig transform of eps_delta. This will be performed in {N_G_chunks**2} batches of (N_E={N_E}, N_G={G_per_chunk}, N_G'={G_per_chunk}) arrays.")

        G_start = np.arange(0, G_per_chunk*N_G_chunks, G_per_chunk)
        G_stop = np.append(np.arange(G_per_chunk, G_per_chunk*N_G_chunks, G_per_chunk), N_G)

        eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'r+')
        eps_delta = eps_delta_h5['eps_delta']
        start_time = time.time()
        for i in range(N_G_chunks):
            for j in range(N_G_chunks):
                start_time1 = time.time()
                eps_delta_chunk = eps_delta[:, G_start[i]:G_stop[i], G_start[j]:G_stop[j]] #(E,G_chunk,G'_chunk)
                
                eps_lfe = kk.kramerskronig_lfe(eps_delta_chunk, parmt.E_max, parmt.dE)

                if parmt.debug_logging and (rank == 0 or rank == None):
                    logger.info(f'\t\t\t\tKK finished in {time.time() - start_time1} s.')

                start_time2 = time.time()
                eps_delta[:, G_start[i]:G_stop[i], G_start[j]:G_stop[j]] = eps_lfe + np.identity(N_G)[None, G_start[i]:G_stop[i], G_start[j]:G_stop[j]] #add eps_pv and identity to existing eps_lfe

                if parmt.debug_logging and (rank == 0 or rank == None):
                    logger.info(f'\t\t\t\tWriting to hdf5 finished in {time.time() - start_time2} s.')
                    logger.info(f'\t\t\tBatch {(i+1)*N_G_chunks + (j+1)} finished in {time.time() - start_time1} s.')

        if rank == 0 or rank == None:
            logger.info(f'\t\tKramers-Kronig transform completed in {(time.time() - start_time):.2f} s.')

        #take inverse in chunks of energy
        N_E_chunks = int(np.ceil(N_E/E_per_chunk))

        if rank == 0 or rank == None:
            logger.info(f"\t\tBeginning inversion of eps_delta to obtain eps_LFE. This will be performed in {N_E_chunks} batches of (N_E={E_per_chunk}, N_G={N_G}, N_G'={N_G}) arrays.")

        E_start = np.arange(0, E_per_chunk*N_E_chunks, E_per_chunk)
        E_stop = np.append(np.arange(E_per_chunk, E_per_chunk*N_E_chunks, E_per_chunk), N_E)

        eps_lfe = np.empty((N_E, N_G), dtype='complex')
        start_time = time.time()
        for i in range(N_E_chunks):
            start_time1 = time.time()
            eps_delta_chunk = eps_delta[E_start[i]:E_stop[i]]

            if E_per_chunk == 1: # eps_delta_chunk will be 2d
                eps_lfe[E_start[i]:E_stop[i]] = (1/np.diagonal(np.linalg.inv(eps_delta_chunk), axis1=0, axis2=1))
            else: # eps_delta_chunk will be 3d
                eps_lfe[E_start[i]:E_stop[i]] = (1/np.diagonal(np.linalg.inv(eps_delta_chunk), axis1=1, axis2=2)) #make general diagonal function? write this whole function with numba? Is inv faster if we transpose first? (E,G,G') -> (G,E)

            if parmt.debug_logging and (rank == 0 or rank == None):
                logger.info(f'\t\t\tBatch {i+1} finished in {time.time() - start_time1} s.')

        eps_delta_h5.close()
        eps_lfe = eps_lfe.transpose((1,0)).copy() #(G,E)

        if rank == 0 or rank == None:
            logger.info(f'\t\tInversion completed and eps_LFE calculated in {(time.time() - start_time):.2f} s.')

        #Bin real and imaginary parts - modify binning so both parts done at same time
        start_time = time.time()
        tot_bin_eps_im, tot_bin_weights = binning.bin_eps_q(q, G_q, np.imag(eps_lfe), bin_centers, tot_bin_eps_im, tot_bin_weights)
        tot_bin_eps_re, tot_bin_weights_re = binning.bin_eps_q(q, G_q, np.real(eps_lfe), bin_centers, tot_bin_eps_re, tot_bin_weights_re)

        if rank == 0 or rank == None:
            logger.info(f'\t\tBinning of eps_LFE completed in {(time.time() - start_time):.2f} s.')

        # Save to working directory
        np.save(f'{working_dir}/tot_bin_eps_im_q.npy', tot_bin_eps_im)
        np.save(f'{working_dir}/tot_bin_eps_re_q.npy', tot_bin_eps_re)
        np.save(f'{working_dir}/tot_bin_weights_q.npy', tot_bin_weights)

        if rank == 0 or rank == None:
            logger.info(f'\tcomplete. Time taken = {(time.time() - start_time_full):.2f} s.')

    # Removing extra bins 
    tot_bin_eps_im = tot_bin_eps_im[:-N_ang_bins, :]
    tot_bin_eps_re = tot_bin_eps_re[:-N_ang_bins, :]
    tot_bin_weights = tot_bin_weights[:-N_ang_bins]

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    # delete large working directory files
    shutil.rmtree(f'{working_dir}/eta_qG')
    os.remove(f'{working_dir}/eps_delta.h5')

    return tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper(n_tabs=2)
def get_RPA_dielectric_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank):
    
    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]
    qG = q[None, :] + G_q

    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    valbot, valtop, conbot, contop = np.load(parmt.store + '/bands.npy')
    # For E >=0, initial states are occupied and final states are unoccupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV')
    start_time = time.time()
    im_delE = delta_energy(mo_en_i[:,valbot:valtop+1], mo_en_f[:,conbot:contop+1]) #(k,2,a,b) # Calculating the delta function in energy
    RPA_Im_eps_external_prefactor_LFE(qG, k_f, mo_coeff_i[:,:,valbot:valtop+1], mo_coeff_f_conj[:,:,conbot:contop+1], im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor, neg_E=False)
    
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV. Time taken = {(time.time() - start_time):.2f} s')

    # For E < 0, initial states are unoccupied and final states are occupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV')
    start_time = time.time()
    im_delE = delta_energy(mo_en_i[:,conbot:contop+1], mo_en_f[:,valbot:valtop+1]) #(k,2,a,b)
    #negate ind?
    RPA_Im_eps_external_prefactor_LFE(qG, k_f, mo_coeff_i[:,:,conbot:contop+1], mo_coeff_f_conj[:,:,valbot:valtop+1], im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor, neg_E=True)

    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV. Time taken = {(time.time() - start_time):.2f} s')


def RPA_Im_eps_external_prefactor_LFE(qG, k_f, mo_coeff_i, mo_coeff_f_conj, im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor, neg_E):
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
    if neg_E:
        ind_sign = -1
    else:
        ind_sign = 1

    im_delE_ind = im_delE[:,0,:,:].astype('int') # energy indices (k,i,j)
    im_delE_rem = im_delE[:,1,:,:] # remainders (k,i,j)

    N_E = int(parmt.E_max/parmt.dE + 1) # Number of E for E <=0 or E>= 0

    start_time = time.time()
    #make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    #save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelize over k
        p.starmap(partial(eps.get_3D_overlaps_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, working_dir=working_dir), k_tup) #(G,k,i,j)

    if rank == None or rank == 0:
        logger.info(f'\t\t\t\teta_qG calculated for all k and G. Time taken = {(time.time() - start_time):.2f} s.')

    # Load 3D overlaps to memory
    start_time = time.time()
    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], qG.shape[0]), dtype='complex128')
    for i_k in range(k_f.shape[0]):
        eta_qG[i_k] = np.load(working_dir + f'/eta_qG/eta_qG_k{i_k}.npy')
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:]

    if rank == None or rank == 0:
        logger.info(f'\t\t\t\t3D overlaps loaded to memory for all G. Time taken = {(time.time() - start_time):.2f} s.')

    # Calculate eps_delta
    start_time = time.time()
    
    if neg_E: #For E < 0
        prefactor = -1*prefactor

    eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'r+') #eps_delta hdf5 file
    eps_delta = eps_delta_h5['eps_delta']
    
    chunks = eps_delta.chunks
    if chunks is not None:
        E_per_chunk = chunks[0]
    else:
        E_per_chunk = 1
    N_E_chunks = int(np.ceil(N_E/E_per_chunk))
    E_start = np.arange(0, E_per_chunk*N_E_chunks, E_per_chunk)
    E_stop = np.append(np.arange(E_per_chunk, E_per_chunk*N_E_chunks, E_per_chunk), N_E)

    if rank == 0 or rank == None:
        logger.info(f"\t\t\t\tBeginning eps_delta. This will be performed in {N_E_chunks} batches of N_E = {E_per_chunk}")

    eps_delta_n_min_1 = np.zeros((qG.shape[0], qG.shape[0]), dtype='complex')
    for n_E_chunk in range(N_E_chunks):
        start_time1 = time.time()
        E_i = E_start[n_E_chunk]
        E_f = E_stop[n_E_chunk]
        eps_delta_chunk = np.empty((E_f-E_i, qG.shape[0], qG.shape[0]), dtype='complex')
        for i in range(E_f - E_i):
            n_E = E_i + i
            k_ind, i_ind, j_ind = np.where(im_delE_ind == n_E)
            eta_qG_kij = eta_qG[k_ind, i_ind, j_ind] #(k,i,j,G) -> (kij,G) #only keeps k,i,j elements relevent to delta calculation
            rem_kij = im_delE_rem[k_ind, i_ind, j_ind]

            eta_qG_kij_sq = np.empty((k_ind.shape[0], eta_qG_kij.shape[1], eta_qG_kij.shape[1]), dtype='complex')
            eta_qG_kij_sq = eps.gen_outer(eta_qG_kij.conj(), eta_qG_kij, eta_qG_kij_sq, prefactor) #numba optimized function

            eps_delta_n = np.tensordot(rem_kij, eta_qG_kij_sq, axes=(0,0)) 
            eps_delta_chunk[i] = eps_delta_n + eps_delta_n_min_1

            eps_delta_n_min_1 = np.tensordot(1 - rem_kij, eta_qG_kij_sq, axes=(0,0)) # (1 - rem)*eta_sq
        
        if parmt.debug_logging and (rank == 0 or rank == None):
            logger.info(f'\t\t\t\t\t\tBatch {n_E_chunk} of eps_delta calculated. Time taken = {(time.time() - start_time1):.2f} s.')

        # write to hdf5 only after entire chunk has been calculated
        start_time1 = time.time()
        if ind_sign == -1:
            eps_delta[(ind_sign*E_f + int(parmt.E_max/parmt.dE)):(ind_sign*E_i + int(parmt.E_max/parmt.dE))] = np.flip(eps_delta_chunk, axis=0)
        else: # ind_sign = 1
            eps_delta[(ind_sign*E_i + int(parmt.E_max/parmt.dE)):(ind_sign*E_f + int(parmt.E_max/parmt.dE))] = eps_delta_chunk
        
        if parmt.debug_logging and (rank == 0 or rank == None):
            logger.info(f'\t\t\t\t\t\tBatch {n_E_chunk} of eps_delta saved to hdf5. Time taken = {(time.time() - start_time1):.2f} s.')

    eps_delta_h5.close()
            
    """
    for n_E in range(N_E):
        k_ind, i_ind, j_ind = np.where(im_delE_ind == n_E)
        eta_qG_kij = eta_qG[k_ind, i_ind, j_ind] #(k,i,j,G) -> (kij,G) #only keeps k,i,j elements relevent to delta calculation
        rem_kij = im_delE_rem[k_ind, i_ind, j_ind]

        eta_qG_kij_sq = np.empty((k_ind.shape[0], eta_qG_kij.shape[1], eta_qG_kij.shape[1]), dtype='complex')
        eta_qG_kij_sq = eps.gen_outer(eta_qG_kij, eta_qG_kij.conj(), eta_qG_kij_sq, prefactor) #numba optimized function

        eps_delta_n = np.tensordot(rem_kij, eta_qG_kij_sq, axes=(0,0)) 
        eps_delta[ind_sign*n_E + int(parmt.E_max/parmt.dE)] = eps_delta_n + eps_delta_n_min_1 #write to hdf5. Index is offset because this array contains both positive and negative energies
        eps_delta_n_min_1 = np.sum(eta_qG_kij_sq, axis=0) - eps_delta_n # (1 - rem)*eta_sq
    eps_delta_h5.close()
    """

    if rank == None or rank == 0:
        logger.info(f'\t\t\t\tDelta part of epsilon calculated. Time taken = {(time.time() - start_time):.2f} s.')

    #return np.copy(np.transpose(eps_delta, (1,2,0))) #(G,G',E)

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