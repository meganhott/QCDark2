import numpy as np
import time
import os
import shutil # for removing working directories after calculation is finished
import multiprocessing as mp
from functools import partial
from numba import njit
import h5py

from dielectric_pyscf.routines import logger, time_wrapper, makedir, alpha, me
from dielectric_pyscf.epsilon_utils import epsilon_r, interp_eps
import dielectric_pyscf.input_parameters as parmt
import dielectric_pyscf.epsilon_helper as eps
import dielectric_pyscf.binning as binning
import dielectric_pyscf.kramers_kronig as kk

optical_limit_override = True

@time_wrapper
def get_RPA_dielectric(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    if parmt.dir_1d is not None:
        raise NotImplementedError('1D dielectric function has not yet been added to general RPA function')

    # Load DFT results
    band_ids = np.load(parmt.store + '/bands.npy')
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    k_i = np.load(parmt.store + '/k-pts_i.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    dk = np.max(np.linalg.norm(dark_objects['cell'].reciprocal_vectors(), axis=1)/np.array(parmt.k_grid)) # maximum k-grid spacing

    if parmt.optical_limit:
        mo_coeff_f_conj = mo_coeff_i.conj()
        mo_en_f = mo_en_i
    else:
        mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
        mo_en_f = np.load(dft_path + 'mo_en_f.npy')

    # Prefactor for susceptibility
    V_cell = dark_objects['V_cell']
    N_k = k_i.shape[0]
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(V_cell*N_k*parmt.dE)

    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: # MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)

    # Generate optimal path for 3D overlaps calculation
    N_AO = len(dark_objects['aos'])
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff_f_conj, np.ones((k_f.shape[0], N_AO, N_AO)), mo_coeff_i)[0]

    if parmt.dir_1d is not None: # only compute along one direction
        N_ang_bins = 1
    else:
        N_ang_bins = parmt.N_phi*(parmt.N_theta-2) + 2

    if parmt.include_lfe: # LFE
        N_E = int(2*parmt.E_max/parmt.dE) + 1 # -E_max < E < E_max for LFEs
        makedir(working_dir + '/eta_qG')
        RPA_function = RPA_LFE_gen_q
    else: # noLFE
        N_E = int(parmt.E_max/parmt.dE) + 1 # 0 < E < E_max for noLFE
        makedir(working_dir + '/eps_im')
        RPA_function = RPA_noLFE_gen_q
    
    # Generate bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, N_E))
    tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, N_E))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)
    tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)

    N_q = len(dark_objects['unique_q'])
    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q
    
    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    q_keys = list(dark_objects['unique_q'].keys())[q_start:q_stop]
    G_vectors = dark_objects['G_vectors']

    for i_q, q in enumerate(q_keys, start=q_start):
        k_pairs = np.array(dark_objects['unique_q'][q])
        q = np.array(q)

        if rank == 0 or rank == None:
            logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {np.array2string(q, precision=5)},')

        start_time_q = time.time()

        mo_coeff_i_q = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
        mo_coeff_f_conj_q = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
        mo_en_i_q = mo_en_i[k_pairs[:,0]] #(k_pair,i)
        mo_en_f_q = mo_en_f[k_pairs[:,1]] #(k_pair,j)
        k_i_q = k_i[k_pairs[:,0]] #(k,3)
        k_f_q = k_f[k_pairs[:,1]] #(k,3)

        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]

        # Incorporate 1D later
        """
        if parmt.dir_1d is not None: # 1D calculation selects G-vectors along specific direction, dir_1d
            dir_sph = binning.cartesian_to_spherical(np.array(parmt.dir_1d)[np.newaxis])[0]
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
        """
        global optical_limit_override

        if (np.linalg.norm(q) < parmt.dq) and (parmt.dq < dk):
            # Remove G = 0 if |q| < dq and dq < dk (if it would contribute to first bin)
            # This also removes some contributions to second bins, but these would be inaccurate anyway?
            G_q = G_q[1:] 
            optical_limit = True # compute G = 0 contribution with optical limit
        elif optical_limit_override:
            optical_limit = True
        else:
            optical_limit = False

        G_q = G_q[np.linalg.norm(q[None, :]+G_q, axis=1) > parmt.q_min - 0.5*parmt.dq]

        if rank == 0 or rank == None:
            logger.info(f'\t\tNumber of G vectors = {len(G_q)},')

        if G_q.shape[0] == 0 and not optical_limit:
            logger.info('\t\tNo contributions from this q')
        else:
            # Send to RPA function
            eps_q, bins_q = RPA_function(q, G_q, dark_objects, mo_en_i_q, mo_en_f_q, mo_coeff_i_q, mo_coeff_f_conj_q, k_i_q, k_f_q, band_ids, N_E, bin_centers[:N_ang_bins], einsum_path, working_dir, rank, optical_limit, prefactor)

            # Bin epsilon
            start_time = time.time()
            tot_bin_eps_im, tot_bin_weights = binning.bin_eps_q(bins_q, np.imag(eps_q), bin_centers, tot_bin_eps_im, tot_bin_weights)
            if parmt.include_lfe:
                tot_bin_eps_re, tot_bin_weights_re = binning.bin_eps_q(bins_q, np.real(eps_q), bin_centers, tot_bin_eps_re, tot_bin_weights_re) # tot_bin_weights_re is same as tot_bin_weights

            if rank == 0 or rank == None:
                logger.info(f'\t\tBinning of epsilon completed in {(time.time() - start_time):.2f} s.')

            # Save to working directory
            np.save(f'{working_dir}/tot_bin_eps.npy', tot_bin_eps_re + 1j*tot_bin_eps_im)
            np.save(f'{working_dir}/tot_bin_weights.npy', tot_bin_weights)

        if rank == 0 or rank == None:
            logger.info(f'\t\tCompleted i_q = {i_q + 1}. Time taken = {(time.time() - start_time_q):.2f} s.')

    # Removing extra bins 
    tot_bin_eps_im = tot_bin_eps_im[:-N_ang_bins, :]
    tot_bin_eps_re = tot_bin_eps_re[:-N_ang_bins, :]
    tot_bin_weights = tot_bin_weights[:-N_ang_bins]

    # Delete large working directory files
    if parmt.include_lfe:
        shutil.rmtree(f'{working_dir}/eta_qG')
        os.remove(f'{working_dir}/eps_delta.h5')
    else:
        shutil.rmtree(f'{working_dir}/eps_im')

    return tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

def RPA_noLFE_gen_q(q, G_q, dark_objects, mo_en_i_q, mo_en_f_q, mo_coeff_i_q, mo_coeff_f_conj_q, k_i_q, k_f_q, band_ids, N_E, first_bins, einsum_path, working_dir, rank, optical_limit, prefactor):
    """
    Calculates macroscopic dielectric function eps(q+G,E) *without* local field effects for one q-vector and all G-vectors. 

    Inputs: 
        q: (3,): q-vector in 1BZ
        G: (N_G, 3): G-vectors
        dark_objects: dict
        mo_en_i_q: (k,i): initial state energies for q
        mo_en_f_q: (k,j): final state energies for q
        mo_coeff_i_q: (k,a,i): initial state MO coefficients
        mo_coeff_f_conj_q: (k,b,j) final state MO coefficients
        k_i_q: (k,3): initial state k-points
        k_f_q: (k,3): final state k-points
        band_ids: list of int
            Lowest and highest bands to include for initial/final states. Corresponds to relevant valence and conduction bands.
        N_E: int: number of energies
        first_bins: (N_ang_bins,3): bins closest to origin - need to be treated in optical limit
        einsum_path: optimal einsum path for 3D overlaps
        working_dir: str
        rank: int or None: MPI rank
        optical_limit: bool: determines if proper q -> 0 limit is taken for first bins
        prefactor: float
    Outputs:
        eps_q: (bins,E): imaginary part of macroscopic dielectric function for q
        bins_q: (N_first_bins+N_G,3): q+G locations for binning
    """
    # Retrieving arrays from dark_objects
    primgauss_arr = dark_objects['primgauss_arr']
    AO_arr = dark_objects['AO_arr']
    coeff_arr = dark_objects['coeff_arr']
    q_cuts = dark_objects['R_cutoff_q_points']
    unique_Ri = dark_objects['unique_Ri']

    if rank == 0 or rank == None:
        logger.info(f'\t\tStarting calculation of Im(eps) for 0 < E <= {parmt.E_max} eV')
    start_time1 = time.time()
    
    ivalbot, ivaltop, iconbot, icontop = band_ids
    qG = q[None, :] + G_q

    if G_q.shape[0] == 0:
        eps_q = np.zeros((0,N_E), dtype='complex')
    else:
        # Calculating the delta function in energy
        im_delE = delta_energy(mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_f_q[:,iconbot:icontop+1]) #(k,2,a,b)

        # Make k_f, coeff tuples for starmap
        N_k = k_f_q.shape[0]
        k_tup = []
        for i_k in range(N_k):
            k_tup.append( (i_k, k_f_q[i_k], mo_coeff_i_q[i_k,:,ivalbot:ivaltop+1], mo_coeff_f_conj_q[i_k,:,iconbot:icontop+1]) )

        # Save Im(eps) for each k, then load and combine after calculating for all k
        start_time = time.time()
        with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
            p.starmap(partial(eps.get_eps_im_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, im_delE=im_delE, working_dir=working_dir), k_tup)

        if rank == None or rank == 0:
            logger.info(f'\t\t\tIm(eps) calculated and saved for all k. Time taken = {(time.time() - start_time):.2f} s.')
        
        # Load eps_im to memory and sum over k
        start_time = time.time()
        N_E = int(parmt.E_max/parmt.dE + 1)
        eps_im = np.zeros((G_q.shape[0], N_E), dtype='float')
        for i_k in range(N_k):
            eps_im += np.load(working_dir + f'/eps_im/eps_im_k{i_k}.npy')
        eps_q = 1j*eps_im
        
        if rank == None or rank == 0:
            logger.info(f'\t\t\tIm(eps) loaded to memory and summed over k. Time taken = {(time.time() - start_time):.2f} s.')

    bins_q = qG

    if optical_limit:
        start_time = time.time()
        eps_head = 1j * alpha**4 * me**2 * RPA_head(dark_objects['cell'], k_i_q, mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_i_q[:,iconbot:icontop+1], mo_coeff_i_q[:,:,ivalbot:ivaltop+1], mo_coeff_i_q[:,:,iconbot:icontop+1], first_bins)

        bins_q = np.concatenate((binning.spherical_to_cartesian(first_bins), bins_q), axis=0)
        eps_q = np.concatenate((eps_head, eps_q), axis=0)

        logger.info(f'\t\t\tOptical limit computed for bins closest to origin. Time taken = {(time.time() - start_time):.2f} s.')

        global optical_limit_override
        optical_limit_override = False # only computes optical limit once

    if rank == None or rank == 0:
        logger.info(f'\t\tFinished calculation of Im(eps). Time taken = {(time.time() - start_time1):.2f} s')
   
    return prefactor*eps_q, bins_q

"""
def RPA_noLFE_optical_limit(dark_objects, mo_en_i_q, mo_coeff_i_q, k_i_q, bands_ids, first_bins, prefactor):
    start_time = time.time()
    eps_head = 1j * alpha**4 * me**2 * RPA_head(dark_objects['cell'], k_i_q, mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_i_q[:,iconbot:icontop+1], mo_coeff_i_q[:,:,ivalbot:ivaltop+1], mo_coeff_i_q[:,:,iconbot:icontop+1], first_bins)

    logger.info(f'\t\t\tOptical limit computed for bins closest to origin. Time taken = {(time.time() - start_time):.2f} s.')
    return prefactor*eps_head, first_bins
"""

def RPA_LFE_gen_q(q, G_q, dark_objects, mo_en_i_q, mo_en_f_q, mo_coeff_i_q, mo_coeff_f_conj_q, k_i_q, k_f_q, band_ids, N_E, first_bins, einsum_path, working_dir, rank, optical_limit, prefactor):
    """
    Calculates macroscopic dielectric function eps(q+G,E) including local field effects for one q-vector and all G-vectors. 

    Inputs: 
        q: (3,): q-vector in 1BZ
        G: (N_G, 3): G-vectors
        dark_objects: dict
        mo_en_i_q: (k,i): initial state energies for q
        mo_en_f_q: (k,j): final state energies for q
        mo_coeff_i_q: (k,a,i): initial state MO coefficients
        mo_coeff_f_conj_q: (k,b,j) final state MO coefficients
        k_i_q: (k,3): initial state k-points
        k_f_q: (k,3): final state k-points
        band_ids: list of int
            Lowest and highest bands to include for initial/final states. Corresponds to relevant valence and conduction bands.
        N_E: int: number of energies
        first_bins: (N_ang_bins,3): bins closest to origin - need to be treated in optical limit
        einsum_path: optimal einsum path for 3D overlaps
        working_dir: str
        rank: int or None: MPI rank
        optical_limit: bool: determines if proper q -> 0 limit is taken for first bins
        prefactor: float
    Outputs:
        eps_q: (bins,E): macroscopic dielectric function for q
        bins_q: (N_first_bins+N_G,3): q+G locations for binning
    """
    N_G = G_q.shape[0]
    qG = q[None, :] + G_q
    bins_q = qG
    ivalbot, ivaltop, iconbot, icontop = band_ids

    if optical_limit:
        print(f'Warning: directional optical limit is not yet implemented for LFEs. The dielectric function for q -> will be calculated in the q_shift_dir = {parmt.q_shift_dir} direction only. This direction can be specified in the input file.')

        # check if material is iostropic - to be implemented

        N_G = N_G + 1 # determine how many "extra" N_G required for optical bins - just one for now
        N_G_skip = 1 # skip first G indices of eps_delta when saving body

        bins_q = np.concatenate((first_bins, bins_q), axis=0)
    else:
        N_G_skip = 0

    # **Creating hdf5 file to store large intermediate array**
    eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'w') # eps_delta with be stored to hdf5 for each energy

    # Determining chunks used to store data
    E_per_chunk, G_per_chunk = get_hdf5_chunks(N_E, N_G, 2**30) # want to read/write in 1 GB chunks later
    eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, G_per_chunk, G_per_chunk))

    if rank == 0 or rank == None:
        logger.info(f'\t\tAn hdf5 dataset has been created to store intermediate results. The shape is (N_E={N_E}, N_G={N_G}, N_G={N_G}) saved in chunks of ({E_per_chunk}, {G_per_chunk}, {G_per_chunk}).')

    eps_delta_h5.close()

    # **Calculate and store spectral part of polarizability (E,G,G')**

    # For E >=0, initial states are occupied and final states are unoccupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV')
    start_time = time.time()

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_f_q[:,iconbot:icontop+1]) #(k,2,a,b)

    RPA_body_LFE(qG, k_f_q, mo_coeff_i_q[:,:,ivalbot:ivaltop+1], mo_coeff_f_conj_q[:,:,iconbot:icontop+1], im_delE, dark_objects, einsum_path, working_dir, rank, prefactor, neg_E=False, N_G_skip=N_G_skip)
    
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV. Time taken = {(time.time() - start_time):.2f} s')

    # For E < 0, initial states are unoccupied and final states are occupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV')
    start_time = time.time()

    im_delE = delta_energy(mo_en_i_q[:,iconbot:icontop+1], mo_en_f_q[:,ivalbot:ivaltop+1]) #(k,2,a,b)

    RPA_body_LFE(qG, k_f_q, mo_coeff_i_q[:,:,iconbot:icontop+1], mo_coeff_f_conj_q[:,:,ivalbot:ivaltop+1], im_delE, dark_objects, einsum_path, working_dir, rank, prefactor, neg_E=True, N_G_skip=N_G_skip)

    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV. Time taken = {(time.time() - start_time):.2f} s')

    if optical_limit:
        q_dir = np.array(parmt.q_shift_dir)
        q_dir = q_dir / np.linalg.norm(q_dir) 

        start_time = time.time()
        if rank == 0 or rank == None:
            logger.info(f'\t\t\tStarting calculation of head and wings (q -> 0 limit).')

        VCell = dark_objects['V_cell']
        N_k = k_i_q.shape[0]
        prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2
        prefactor_wings = 8.*(np.pi**2)*(alpha**3)*me/(VCell*N_k)/parmt.dE * alpha*me

        # Calculating head for 0 <= E <= E_max
        eps_im_head = prefactor_head * RPA_head(dark_objects['cell'], k_i_q, mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_i_q[:,iconbot:icontop+1], mo_coeff_i_q[:,:,ivalbot:ivaltop+1], mo_coeff_i_q[:,:,iconbot:icontop+1], q_dir[np.newaxis,:]) #(E,)

        # Head for -E_max <= E <= E_max
        eps_delta_head = np.concatenate((-1*np.flip(eps_im_head)[:-1], eps_im_head)).astype('complex') #(E,)

        # Wings
        eps_delta_wings_p = prefactor_wings * RPA_wings(G_q, k_i_q, mo_en_i_q[:,ivalbot:ivaltop+1], mo_en_i_q[:,iconbot:icontop+1], mo_coeff_i_q[:,:,ivalbot:ivaltop+1], mo_coeff_i_q[:,:,iconbot:icontop+1], q_dir[np.newaxis,:], dark_objects, einsum_path)[0] #(E,G)
        # indexing first axis since we currently only calculate in one direction

        eps_delta_wings_n = -1 * prefactor_wings * RPA_wings(G_q, k_i_q, mo_en_i_q[:,iconbot:icontop+1], mo_en_i_q[:,ivalbot:ivaltop+1], mo_coeff_i_q[:,:,iconbot:icontop+1], mo_coeff_i_q[:,:,ivalbot:ivaltop+1], q_dir[np.newaxis,:], dark_objects, einsum_path)[0]

        # Concatenate wings along energy and write to hdf5 along with head
        eps_delta_wings = np.concatenate((np.flip(eps_delta_wings_n, axis=0)[:-1], eps_delta_wings_p), axis=0) #(E,G)

        eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'r+')
        eps_delta = eps_delta_h5['eps_delta']

        eps_delta[:,0,0] = eps_delta_head

        eps_delta[:,0,1:] = eps_delta_wings.conj()
        eps_delta[:,1:,0] = eps_delta_wings

        eps_delta_h5.close()

        if rank == 0 or rank == None:
            logger.info(f'\t\t\tFinished calculation of head and wings (q -> 0 limit). Time taken = {(time.time() - start_time):.2f} s')

        global optical_limit_override
        optical_limit_override = False # only computes optical limit once

    # **Perform Kramers-Kronig transformation to get PV parts of Re(eps) and Im(eps)**
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
            eps_delta[:, G_start[i]:G_stop[i], G_start[j]:G_stop[j]] = eps_lfe + np.identity(N_G)[None, G_start[i]:G_stop[i], G_start[j]:G_stop[j]] # add eps_pv and identity to existing eps_lfe

            if parmt.debug_logging and (rank == 0 or rank == None):
                logger.info(f'\t\t\t\tWriting to hdf5 finished in {time.time() - start_time2} s.')
                logger.info(f'\t\t\tBatch {(i+1)*N_G_chunks + (j+1)} finished in {time.time() - start_time1} s.')

    if rank == 0 or rank == None:
        logger.info(f'\t\tKramers-Kronig transform completed in {(time.time() - start_time):.2f} s.')

    # **Perform matrix inverse and take inverse of diagonal**
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
            eps_lfe[E_start[i]:E_stop[i]] = (1/np.diagonal(np.linalg.inv(eps_delta_chunk), axis1=1, axis2=2))

        if parmt.debug_logging and (rank == 0 or rank == None):
            logger.info(f'\t\t\tBatch {i+1} finished in {time.time() - start_time1} s.')

    eps_delta_h5.close()
    eps_lfe = eps_lfe.transpose((1,0)).copy() #(G,E)

    if rank == 0 or rank == None:
        logger.info(f'\t\tInversion completed and eps_LFE calculated in {(time.time() - start_time):.2f} s.')

    if optical_limit:
        # anisotropic optical limit not yet implemented - optical limit in q_shift_dir will be written to all first bins
        N_first_bins = first_bins.shape[0]
        eps_q = np.empty((N_first_bins + N_G-1), dtype='complex')
        eps_q[:N_first_bins] = eps_lfe[0]
        eps_q[N_first_bins:] = eps_lfe[N_first_bins+1:]
    else:
        eps_q = eps_lfe

    return eps_q, bins_q

def RPA_body_LFE(qG, k_f, mo_coeff_i, mo_coeff_f_conj, im_delE, dark_objects, einsum_path, working_dir, rank, prefactor, neg_E, N_G_skip=0):
    """
    Writes spectral part of lim_{n->0} <ik|exp(-i(q+G)r)|j(k+q)> <j(k+q)|exp(i(q+G')r)|ik> / |q+G| / |q+G'| /(E - (E_{j(k+q)} - E_{ik} +/- i n)), summed over i,j,k, to hdf5 file. 

    Inputs: 
        qG: (N_G, 3): q + G-vectors
        k_f: (k,3): final state k-points
        mo_coeff_i: (k,a,i): initial state MO coefficients
        mo_coeff_f_conj: (k,b,j) final state MO coefficients
        im_delE: (k,2,i,j): energy delta function
        dark_objects: dict
        einsum_path: optimal einsum path for 3D overlaps
        working_dir: str
        rank: int or None: MPI rank
        prefactor: float
        neg_E: bool: True if E_f > E_i (for calculating negative energy transfer)
        N_G_skip: int: 
            Number of extra G vectors to include. Used for q -> 0 to skip G=0, G'=0 entries which must be treated separately 
    Outputs:
        eps_delta (N_G,N_G,N_E):
            Spectral part of dielectric function. Re(eps_delta) corresponds to Im(eps) and Im(eps_delta) corresponds to Re(eps).
    """
    # Retrieving arrays from dark_objects
    primgauss_arr = dark_objects['primgauss_arr']
    AO_arr = dark_objects['AO_arr']
    coeff_arr = dark_objects['coeff_arr']
    q_cuts = dark_objects['R_cutoff_q_points']
    unique_Ri = dark_objects['unique_Ri']

    if neg_E:
        ind_sign = -1
    else:
        ind_sign = 1

    im_delE_ind = im_delE[:,0,:,:].astype('int') # energy indices (k,i,j)
    im_delE_rem = im_delE[:,1,:,:] # remainders (k,i,j)

    N_E = int(parmt.E_max/parmt.dE + 1) # number of E for E <=0 or E>= 0
    N_G = qG.shape[0] # number of G-vectors

    # Make k_f, coeff tuples for starmap
    start_time = time.time()
    k_tup = []
    for i_k in range(k_f.shape[0]):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    # Save eta for each k, then load and combine after calculating for all k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
        p.starmap(partial(eps.get_3D_overlaps_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, working_dir=working_dir), k_tup) #(G,k,i,j)

    if rank == None or rank == 0:
        logger.info(f'\t\t\t\teta_qG calculated for all k and G. Time taken = {(time.time() - start_time):.2f} s.')

    # Load 3D overlaps to memory
    start_time = time.time()
    eta_qG = np.empty((k_f.shape[0], mo_coeff_i.shape[2], mo_coeff_f_conj.shape[2], N_G), dtype='complex128')
    for i_k in range(k_f.shape[0]):
        eta_qG[i_k] = np.load(working_dir + f'/eta_qG/eta_qG_k{i_k}.npy')
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,None,:]

    if rank == None or rank == 0:
        logger.info(f'\t\t\t\t3D overlaps loaded to memory for all G. Time taken = {(time.time() - start_time):.2f} s.')

    # Calculate eps_delta
    start_time = time.time()
    
    if neg_E: # for E < 0
        prefactor = -1*prefactor

    eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'r+')
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

    eps_delta_n_min_1 = np.zeros((N_G, N_G), dtype='complex')
    for n_E_chunk in range(N_E_chunks):
        start_time1 = time.time()
        E_i = E_start[n_E_chunk]
        E_f = E_stop[n_E_chunk]
        
        eps_delta_chunk, eps_delta_n_min_1 = eps.delta_GG(im_delE_ind, im_delE_rem, eta_qG, E_i, E_f, N_G, prefactor, eps_delta_n_min_1)
        
        if parmt.debug_logging and (rank == 0 or rank == None):
            logger.info(f'\t\t\t\t\t\tBatch {n_E_chunk} of eps_delta calculated. Time taken = {(time.time() - start_time1):.2f} s.')

        # Write to hdf5 only after entire chunk has been calculated
        start_time1 = time.time()
        if ind_sign == -1:
            eps_delta[(ind_sign*E_f + int(parmt.E_max/parmt.dE) + 1):(ind_sign*E_i + int(parmt.E_max/parmt.dE) + 1), N_G_skip:, N_G_skip:] = np.flip(eps_delta_chunk, axis=0)
        else: # ind_sign = 1
            eps_delta[(ind_sign*E_i + int(parmt.E_max/parmt.dE)):(ind_sign*E_f + int(parmt.E_max/parmt.dE)), N_G_skip:, N_G_skip:] = eps_delta_chunk
        
        if parmt.debug_logging and (rank == 0 or rank == None):
            logger.info(f'\t\t\t\t\t\tBatch {n_E_chunk} of eps_delta saved to hdf5. Time taken = {(time.time() - start_time1):.2f} s.')

    eps_delta_h5.close()
    
    if rank == None or rank == 0:
        logger.info(f'\t\t\t\tDelta part of epsilon calculated. Time taken = {(time.time() - start_time):.2f} s.')

def get_nabla_ovlps(cell, k, mo_coeff_i, mo_coeff_f):
    """
    Returns <ik| nabla |jk> MO integrals built from int1e_ipovlp AO integrals for head and wings of dielectric function.

    Inputs:
        cell: pyscf.pbc.gto.cell.Cell
        k: (k,3): k-points
        mo_coeff_i: (k,a,i): initial state MO coefficients
        mo_coeff_f: (k,b,j) final state MO coefficients
    Output:
        mo_ovlp: (k,3,i,j): nabla overlaps
    """
    ao_ovlp = np.array(cell.pbc_intor('int1e_ipovlp', kpts=k)) #(k,3,a,b)
    mo_ovlp = np.einsum('kai,kbj,knab -> knij', mo_coeff_i.conj(), mo_coeff_f, ao_ovlp) #(k,3,i,j)
    return mo_ovlp #(k,3,i,j)

def RPA_head(cell, k, mo_en_i, mo_en_f, mo_coeff_i, mo_coeff_f, first_bins):
    """
    Calculate the head of the dielectric fuction: G = G' = 0 in the q = 0 limit along each first_bin direction.

    Inputs:
        cell: pyscf.pbc.gto.cell.Cell
        k: (k,3): k-points
        mo_en_i: (k,i): initial state energies
        mo_en_f: (k,j): final state energies
        mo_coeff_i: (k,a,i): initial state MO coefficients
        mo_coeff_f: (k,b,j) final state MO coefficients
        first_bins: (N_ang_bins,3): bins closest to origin
    Output:
        eps_im: (bins, E): imaginary part (spectral part) of head of dielectric function
    """
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,i,j)

    en_denom = 1 / (mo_en_f[:,None,:] - mo_en_i[:,:,None]) #(k,i,j)

    mo_ovlp = get_nabla_ovlps(cell, k, mo_coeff_i, mo_coeff_f) #(k,3,i,j)
    ovlp = en_denom[:,None,:,:] * mo_ovlp #(k,3,i,j)

    # Following same algorithm as get_eps_im_k 
    # Make k_f, coeff tuples for starmap
    N_k = k.shape[0]
    k_tup = []
    for i_k in range(N_k):
        k_tup.append( (i_k, ovlp[i_k]) )

    # Calculate Im(eps) for each k
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
        eps_im_t = p.starmap(partial(eps.get_eps_im_k_head, im_delE=im_delE), k_tup) #(k,E,3,3)
    eps_im_t = np.sum(np.array(eps_im_t), axis=0) #(k,E,3,3) -> (E,3,3)

    # Optical limit for all first bins around origin
    q_dir = first_bins.copy()
    q_dir[:,0] = 1 # setting magnitude to 1 since we want unit vectors
    q_dir = binning.spherical_to_cartesian(q_dir) # convert back to cartesian for dot product
    
    eps_im = np.einsum('bx,by,exy -> be', q_dir, q_dir, eps_im_t)

    return eps_im #(first bins, E)

def RPA_wings(G, k, mo_en_i, mo_en_f, mo_coeff_i, mo_coeff_f, first_bins, dark_objects, path):
    """
    Calculate the wings of the dielectric fuction: G' = 0, G != 0 in the q = 0 limit along each first_bin direction.

    Inputs:
        G: (G,3): G-vectors
        k: (k,3): k-points
        mo_en_i: (k,i): initial state energies
        mo_en_f: (k,j): final state energies
        mo_coeff_i: (k,a,i): initial state MO coefficients
        mo_coeff_f: (k,b,j) final state MO coefficients
        first_bins: (N_ang_bins,3): bins closest to origin
        dark_objects: dict
        path: optimal einsum path
    Output:
        eps_delta: (bins, E, G): spectral part of wings of dielectric function
    """
    # Retrieving arrays from dark_objects
    cell = dark_objects['cell']
    primgauss_arr = dark_objects['primgauss_arr']
    AO_arr = dark_objects['AO_arr']
    coeff_arr = dark_objects['coeff_arr']
    q_cuts = dark_objects['R_cutoff_q_points']
    unique_Ri = dark_objects['unique_Ri']

    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,i,j)
    en_denom = 1 / (mo_en_f[:,None,:] - mo_en_i[:,:,None]) #(k,i,j)

    mo_ovlp = get_nabla_ovlps(cell, k, mo_coeff_i, mo_coeff_f) #(k,3,i,j)
    ovlp = en_denom[:,None,:,:] * mo_ovlp #(k,3,i,j) # <ik|nabla|jk>/(Ej-Ei)

    N_k = k.shape[0]
    k_tup = []
    for i_k in range(N_k):
        k_tup.append( (i_k, k[i_k], ovlp[i_k], mo_coeff_i[i_k], mo_coeff_f[i_k].conj()) )

    with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
        eps_delta_k = p.starmap(partial(eps.get_eps_delta_k_wings, G=G, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=path, im_delE=im_delE), k_tup)
    eps_delta = np.sum(np.array(eps_delta_k), axis=0) #(k,3,E,G) -> (3,E,G)

    # Optical limit for all first bins around origin
    # for now there will just be one bin
    first_bins[:,0] = 1 # setting magnitude to 1 since we want unit vectors
    q_dir = first_bins

    eps_delta = np.einsum('bx,xeg -> beg', q_dir, eps_delta)

    return eps_delta #(first bins, E, G)

def get_hdf5_chunks(N_E, N_G, chunk_slice_size):
    """
    Returns number of energies and G-vectors per (N_E_chunk,N_G_chunk,N_G_chunk) chunk of eps_delta hdf5 file of total size (N_E,N_G,N_G) to obtain chunk sizes of chunk_slice_size.

    Inputs:
        N_E: number of energies
        N_G: int: number of G-vectors
        chunk_slice_size: int: size of desired hdf5 chunks in bytes
    Outputs:
        E_per_chunk: int: number of energies per chunk
        G_per_chunk: int: number of G-vectors per chunk
    """
    E_per_chunk = int(np.floor(chunk_slice_size / (16 * N_G**2)))
    G_per_chunk = int(np.floor(np.sqrt(chunk_slice_size / (16 * N_E))))

    if G_per_chunk > N_G: 
        if E_per_chunk > N_E:
            # no chunking
            E_per_chunk = N_E
            G_per_chunk = N_G
        else:
            if E_per_chunk < 1:
                E_per_chunk = 1
            # only chunking along E
            G_per_chunk = N_G

    return E_per_chunk, G_per_chunk

@njit
def delta_energy(mo_en_i, mo_en_f):
    """
    Implementation of the energy delta function as a "triangle function". Gets the lower energy bin and the remainder to feed into that particular bin.

    Inputs:
        mo_en_i (k,i): initial state energies
        mo_en_f (k,j): final state energies
    Output:
        im_delE (k,2,i,j): energy delta function, where im_delE[:,0] are bin indices and im_delE[:,1] are remainders
    """
    nk, nmo, nmu = mo_en_f.shape[0], mo_en_i.shape[1], mo_en_f.shape[1]
    delE = np.abs((mo_en_f[:, None, :] - mo_en_i[:, :, None])/parmt.dE)
    arr = np.zeros((nk, 2, nmo, nmu), dtype = np.float32)
    arr[:,0,:,:] = delE//1
    arr[:,1,:,:] = 1. - delE%1
    return arr

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

    f.create_dataset('binned_eps', data=bin_eps) # before interpolation
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

# ------------- Below are all old functions -------------

'''
@time_wrapper
def RPA_noLFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    # Reading all relevant data
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')
    k_i = np.load(parmt.store + '/k-pts_i.npy')
    N_k = k_i.shape[0]

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    cell = dark_objects['cell']
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

        eps_q_im = RPA_noLFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

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

    if parmt.optical_first_bins:
        # Calculate first bins with optical limit
        start_time1 = time.time()
        q_dir = np.array(parmt.q_shift_dir)
        q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector
        prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2
        eps_im_head_q_dir = RPA_head(cell, k_i, q_dir, mo_en_i[:,ivalbot:ivaltop+1], mo_en_i[:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], mo_coeff_i[:,:,iconbot:icontop+1]) #(3,3,E)

        q_dir = parmt.q_shift_dir
        if q_dir == 'all': # calculate first bins separately
            q_dirs = bin_centers[:N_ang_bins]
            q_dirs[:,0] = 1 # make unit vectors
            q_dirs = binning.spherical_to_cartesian(q_dirs) #convert to cartesian
        else: # calculate q -> in one direction only
            q_dir = np.array(q_dir)
            q_dirs = np.repeat((q_dir/np.linalg.norm(q_dir))[np.newaxis,:], N_ang_bins, axis=0) # normalize to get unit vector

        eps_im_head = prefactor_head * np.einsum('bi,bj,ije -> be', q_dirs, q_dirs, eps_im_head_q_dir) #(bins, E)

        tot_bin_eps_im[:N_ang_bins] = eps_im_head
        tot_bin_weights[:N_ang_bins] = 1

        if rank == 0 or rank == None:
                logger.info(f'\tFirst bins recalculated with optical limit. Time taken = {(time.time() - start_time1):.2f} s.')

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    #shutil.rmtree(working_dir) # delete working directory

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper
def RPA_noLFE_1d(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    # Reading all relevant data
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')
    k_i = np.load(parmt.store + '/k-pts_i.npy')
    N_k = k_i.shape[0]

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    cell = dark_objects['cell']
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

            eps_q_im = RPA_noLFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

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

    # Calculate first bin with optical limit
    start_time1 = time.time()
    q_dir = np.array(parmt.q_shift_dir)
    q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector
    prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2
    eps_im_head = prefactor_head * RPA_head(cell, k_i, q_dir, mo_en_i[:,ivalbot:ivaltop+1], mo_en_i[:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], mo_coeff_i[:,:,iconbot:icontop+1]) #(E,)
    tot_bin_eps_im[0] = eps_im_head
    tot_bin_weights[0] = 1

    if rank == 0 or rank == None:
        logger.info(f'\tFirst bin recalculated with optical limit. Time taken = {(time.time() - start_time1):.2f} s.')

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper
def RPA_noLFE_1d_binning(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
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

        eps_q_im = RPA_noLFE_q(q, mo_en_f[:,iconbot:icontop+1], mo_en_i[:,ivalbot:ivaltop+1], mo_coeff_f_conj[:,:,iconbot:icontop+1], mo_coeff_i[:,:,ivalbot:ivaltop+1], k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

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

def RPA_noLFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) -> np.ndarray:
    """
    Returns imaginary part of lim_{n->0} |<j(k+q)|exp(i(q+G)r)|ik>|^2/|q+G|^2/(E - (E_{j(k+q)} - E_{ik} + i n)) for one q-vector, summed over i,j,k. 

    Inputs: 
        q: (3,) q-vector
        mo_en_f: (k,j): MO energies of final k-grid
        mo_en_i: (k,i): MO energies of initial k-grid
        mo_coeff_f_conj: (k,j): MO coefficients of final k-grid (conjugated)
        mo_coeff_i: (k,i): MO coefficients of intial k-grid
        k_f: (k,3): final k-grid
        k_pairs: (k,2): k-grid indices for q-vector
        primgauss_arr: (N_blocks, N_primgauss, 3) boolean np.ndarray: 
            Primitive gaussians are included in each block
        AO_arr: (N_blocks, N_AO) boolean np.ndarray: 
            Atomic orbitals are included in each block
        coeff_arr: (N_AO, N_max_primgauss) complex np.ndarray
            Primgauss coefficients for each AO
        q_cuts: cut-off points in q for different R_ids

    Outputs:
        eps_delta:  delta function part of epsilon: Re(eps_delta) corresponds to Im(eps)
                                                    Im(eps_delta) corresponds to Re(eps) 
                    shape = (N_G, N_G, int(parmt.E_max/parmt.dE + 1)), prefactor multiplied later.
    """
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]
    N_k = k_f.shape[0]
    qG = q[None, :] + G_q

    # Calculating the delta function in energy
    im_delE = delta_energy(mo_en_i, mo_en_f) #(k,2,a,b)

    # Make k_f, coeff tuples for starmap
    k_tup = []
    for i_k in range(N_k):
        k_tup.append( (i_k, k_f[i_k], mo_coeff_i[i_k], mo_coeff_f_conj[i_k]) )

    # Save Im(eps) for each k, then load and combine after calculating for all k
    start_time = time.time()
    with mp.get_context('fork').Pool(mp.cpu_count()) as p: # parallelize over k
        p.starmap(partial(eps.get_eps_im_k, qG=qG, primgauss_arr=primgauss_arr, AO_arr=AO_arr, coeff_arr=coeff_arr, unique_Ri=unique_Ri, q_cuts=q_cuts, path=einsum_path, im_delE=im_delE, working_dir=working_dir), k_tup)

    if rank == None or rank == 0:
        logger.info(f'\t\t\tIm(eps) calculated and saved for all k. Time taken = {(time.time() - start_time):.2f} s.')
    
    # Load eps_im to memory and sum over k
    start_time = time.time()
    N_E = int(parmt.E_max/parmt.dE + 1)
    eps_im = np.zeros((qG.shape[0], N_E), dtype='float')
    for i_k in range(N_k):
        eps_im += np.load(working_dir + f'/eps_im/eps_im_k{i_k}.npy')

    if rank == None or rank == 0:
        logger.info(f'\t\t\tIm(eps) loaded to memory and summed over k. Time taken = {(time.time() - start_time):.2f} s.')

    return prefactor * eps_im #(G,E)

def RPA_LFE(dark_objects: dict, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    # Reading all relevant data
    N_AO = len(dark_objects['aos'])
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy').conj()
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    k_f = np.load(parmt.store + '/k-pts_f.npy')
    k_i = np.load(parmt.store + '/k-pts_i.npy')
    N_k = k_i.shape[0]

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    VCell = dark_objects['V_cell']
    cell = dark_objects['cell']
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
        E_per_chunk, G_per_chunk = get_hdf5_chunks(N_E, N_G, 2**30) # want to read/write in 1 GB chunks later
        eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, G_per_chunk, G_per_chunk))

        if rank == 0 or rank == None:
            logger.info(f'\t\tAn hdf5 dataset has been created to store intermediate results. The shape is (N_E={N_E}, N_G={N_G}, N_G={N_G}) saved in chunks of ({E_per_chunk}, {G_per_chunk}, {G_per_chunk}).')

        eps_delta_h5.close()

        RPA_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) # calculate and store delta part of polarizability (E,G,G')

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

    # Calculate first bin with optical limit
    start_time1 = time.time()
    q_dir = np.array(parmt.q_shift_dir)
    q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector
    prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2

    if rank is None: # No MPI: dielectric function has been calculated for all q and can be saved
        save_eps(tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers)

    # delete large working directory files
    shutil.rmtree(f'{working_dir}/eta_qG')
    os.remove(f'{working_dir}/eps_delta.h5')

    return tot_bin_eps_re + 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

@time_wrapper(n_tabs=2)
def RPA_LFE_q(q, mo_en_f, mo_en_i, mo_coeff_f_conj, mo_coeff_i, k_f, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank):
    # Prepare computation
    prefactor = 8.*(np.pi**2)*(alpha**2)*me/(VCell*len(k_pairs))/parmt.dE

    mo_en_i = mo_en_i[k_pairs[:,0]] #(k_pair,i)
    mo_en_f = mo_en_f[k_pairs[:,1]] #(k_pair,j)
    mo_coeff_i = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
    mo_coeff_f_conj = mo_coeff_f_conj[k_pairs[:,1]] #(k_pair,b,j)
    k_f = k_f[k_pairs[:,1]]
    qG = q[None, :] + G_q

    valbot, valtop, conbot, contop = np.load(parmt.store + '/bands.npy')

    # For E >=0, initial states are occupied and final states are unoccupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV')
    start_time = time.time()

    im_delE = delta_energy(mo_en_i[:,valbot:valtop+1], mo_en_f[:,conbot:contop+1]) #(k,2,a,b) # Calculating the delta function in energy
    RPA_body_LFE(qG, k_f, mo_coeff_i[:,:,valbot:valtop+1], mo_coeff_f_conj[:,:,conbot:contop+1], im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor, neg_E=False)
    
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for 0 < E <= {parmt.E_max} eV. Time taken = {(time.time() - start_time):.2f} s')

    # For E < 0, initial states are unoccupied and final states are occupied
    if rank == 0 or rank == None:
        logger.info(f'\t\t\tStarting calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV')
    start_time = time.time()
    im_delE = delta_energy(mo_en_i[:,conbot:contop+1], mo_en_f[:,valbot:valtop+1]) #(k,2,a,b)
    #negate ind?
    RPA_body_LFE(qG, k_f, mo_coeff_i[:,:,conbot:contop+1], mo_coeff_f_conj[:,:,valbot:valtop+1], im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor, neg_E=True)

    if rank == 0 or rank == None:
        logger.info(f'\t\t\tFinished calculation of delta part of polarizability for -{parmt.E_max} <= E <= 0 eV. Time taken = {(time.time() - start_time):.2f} s')


def RPA_noLFE_0q(dark_objects, q_dir):
    """
    Only calculates head and not wings for noLFE in the proper q -> 0 limit
    """
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    
    mo_coeff = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en = np.load(dft_path + 'mo_en_i.npy')
    kpts = np.load(parmt.store + '/k-pts_i.npy')

    N_k = kpts.shape[0]
    VCell = dark_objects['V_cell'] 
    cell = dark_objects['cell']

    mo_en_val = mo_en[:,ivalbot:ivaltop+1] #(k,i)
    mo_en_con = mo_en[:,iconbot:icontop+1] #(k,j)
    mo_coeff_val = mo_coeff[:,:,ivalbot:ivaltop+1] #(k,a,i)
    mo_coeff_con = mo_coeff[:,:,iconbot:icontop+1] #(k,b,j)

    #logger.info(f'\tStarting calculation of Im(eps(q -> 0)) for 0 < E <= {parmt.E_max} eV along q = {q_dir} direction.')

    start_time = time.time()

    prefactor = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2

    q_dir = np.array(q_dir)
    q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector

    eps_im = prefactor * RPA_head(cell, kpts, q_dir, mo_en_val, mo_en_con, mo_coeff_val, mo_coeff_con) #(E,)

    eps_0q = kk.kramerskronig_im2re_causal(eps_im[np.newaxis,:], parmt.E_max, parmt.dE)[0] + 1. + 1j*eps_im # index shenanigans due to optimized KK for multiple G-vectors

    np.save(parmt.store + '/epsilon_0q.npy', eps_0q)
    print(time.time() - start_time)


def RPA_noLFE_0q(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en = np.load(dft_path + 'mo_en_i.npy')
    kpts = np.load(parmt.store + '/k-pts_i.npy')

    N_k = kpts.shape[0]
    VCell = dark_objects['V_cell'] 
    cell = dark_objects['cell']

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']
    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
    N_ang_bins = parmt.N_phi*(parmt.N_theta-2) + 2

    mo_en_val = mo_en[:,ivalbot:ivaltop+1] #(k,i)
    mo_en_con = mo_en[:,iconbot:icontop+1] #(k,j)
    mo_coeff_val = mo_coeff[:,:,ivalbot:ivaltop+1] #(k,a,i)
    mo_coeff_con = mo_coeff[:,:,iconbot:icontop+1] #(k,b,j)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff, np.ones((N_k, N_AO, N_AO)), mo_coeff)[0]

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

        # check if q = 0 - if so, then special calculation needs to be done for G = 0. Other G are calculated normally
        if (q == 0.).all():
            first_bins = binning.spherical_to_cartesian(bin_centers[:N_ang_bins]) #bins closest to origin

            prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2

            eps_im_head = prefactor_head * RPA_head(cell, kpts, mo_en_val, mo_en_con, mo_coeff_val, mo_coeff_con, first_bins) #(first_bins, E)

            if G_q.shape[0] > 1: # calculate other G normally
                eps_im_body = RPA_noLFE_q(q, mo_en_con, mo_en_val, mo_coeff_con.conj(), mo_coeff_val, kpts, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q[1:], unique_Ri, einsum_path, working_dir, rank)

                #concatenate for binning
                eps_q_im = np.concatenate((eps_im_head, eps_im_body), axis=0) #(first_bins+G,E)
                G_q = np.concatenate((first_bins, G_q[1:]), axis=0) # G = 0 is replaced by first bins for optical limit
            else:
                eps_q_im = eps_im_head #(first_bins, E)
                G_q = first_bins

        else: # q != [0,0,0]
            eps_q_im = RPA_noLFE_q(q, mo_en_con, mo_en_val, mo_coeff_con.conj(), mo_coeff_val, kpts, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

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

def RPA_noLFE_0q_1d(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    N_AO = len(dark_objects['aos'])
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    
    dft_path = parmt.store + '/DFT/'
    mo_coeff = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en = np.load(dft_path + 'mo_en_i.npy')
    kpts = np.load(parmt.store + '/k-pts_i.npy')

    N_k = kpts.shape[0]
    VCell = dark_objects['V_cell'] 
    cell = dark_objects['cell']

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']
    unique_Ri = load_unique_R()

    dir_1d = np.array(parmt.dir_1d)
    dir_1d = dir_1d/np.linalg.norm(dir_1d)
    dir_sph = binning.cartesian_to_spherical(dir_1d[np.newaxis])[0]

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi, dir=True)

    mo_en_val = mo_en[:,ivalbot:ivaltop+1] #(k,i)
    mo_en_con = mo_en[:,iconbot:icontop+1] #(k,j)
    mo_coeff_val = mo_coeff[:,:,ivalbot:ivaltop+1] #(k,a,i)
    mo_coeff_con = mo_coeff[:,:,iconbot:icontop+1] #(k,b,j)

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff, np.ones((N_k, N_AO, N_AO)), mo_coeff)[0]

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

            # check if q = 0 - if so, then special calculation needs to be done for G = 0. Other G are calculated normally
            if (q == 0.).all():
                first_bins = dir_sph # get correct angles
                first_bins[0] = bin_centers[0] # get correct q magnitude
                first_bins = binning.spherical_to_cartesian(first_bins[np.newaxis]) #(1,3) 

                prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2

                eps_im_head = prefactor_head * RPA_head(cell, kpts, mo_en_val, mo_en_con, mo_coeff_val, mo_coeff_con, first_bins) #(1,E)

                if G_q.shape[0] > 1: # calculate other G normally
                    eps_im_body = RPA_noLFE_q(q, mo_en_con, mo_en_val, mo_coeff_con.conj(), mo_coeff_val, kpts, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q[1:], unique_Ri, einsum_path, working_dir, rank)

                    #concatenate for binning
                    eps_q_im = np.concatenate((eps_im_head, eps_im_body), axis=0) #(G,E)
                    G_q = np.concatenate((first_bins, G_q[1:]), axis=0) # G = 0 is replaced by first bin for optical limit
                else:
                    eps_q_im = eps_im_head #(1,E)
                    G_q = first_bins

            else: # q != [0,0,0]
                eps_q_im = RPA_noLFE_q(q, mo_en_con, mo_en_val, mo_coeff_con.conj(), mo_coeff_val, kpts, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank)

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

    #shutil.rmtree(working_dir) # delete working directory

    return 1j*tot_bin_eps_im, tot_bin_weights, bin_centers

def group_q(unique_q):
    q_keys = np.list(unique_q.keys())
    q_pairs = {}
    for i, q in enumerate(q_keys):
        qn = [a if a==0 else -1*a for a in q] # -q
        if qn in q_keys and not q == [0.,0.,0.]:
            q_pairs[i] = [q, qn]
            q_keys.remove(qn) # remove -q
        else:
            q_pairs[i] = [q]
    return q_pairs

"""
def RPA_LFE_0q(dark_objects, q_dir, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    # This is still a work in process
    N_AO = len(dark_objects['aos'])
    valbot, valtop, conbot, contop = np.load(parmt.store + '/bands.npy')

    dft_path = parmt.store + '/DFT/'
    mo_coeff = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en = np.load(dft_path + 'mo_en_i.npy')
    kpts = np.load(parmt.store + '/k-pts_i.npy')

    N_k = kpts.shape[0]
    VCell = dark_objects['V_cell'] 
    cell = dark_objects['cell']

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']
    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
    N_ang_bins = parmt.N_phi*(parmt.N_theta-2) + 2

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff, np.ones((N_k, N_AO, N_AO)), mo_coeff)[0]

    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eta_qG')

    # Group together q and -q
    q_pairs = group_q(unique_q)
    N_q_inv = len(q_pairs)

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q_inv

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}. Total number of q after inversion symmetry: {N_q_inv}')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    q_keys = list(q_pairs.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)

    N_E = int(parmt.E_max/parmt.dE*2 + 1) # number of energies to calculate eps at. This in includes -E_max <= E <= E_max
    N_E_p = int(parmt.E_max/parmt.dE + 1) # number of positive energies to calculate eps at. This in includes 0 <= E <= E_max since we can utilize inversion symmetry

    for i_q, q_p in enumerate(q_keys, start=q_start):
        N_q_p = len(q_pairs[q_p])
        q = q_pairs[q_p][0]
        k_pairs = np.array(unique_q[tuple(q)])
        q = np.array(q)

        if N_q_p == 2: # Calculate -q at same time as q
            q_str = f'{np.array2string(q, precision=5)}, {np.array2string(np.array(q_pairs[q_p][1]), precision=5)}'
        else:
            q_str = np.array2string(q, precision=5)

        if rank == 0 or rank == None:
            logger.info(f'\ti_q: {i_q + 1}\n\t\tq = {q_str}')

        start_time_full = time.time()
        
        # Finding relevant G vectors
        G_q = G_vectors[np.linalg.norm(q[None, :]+G_vectors, axis=1) < parmt.q_max + 1.5*parmt.dq]
        N_G = G_q.shape[0]

        if rank == 0 or rank == None:
            logger.info(f'\t\tnumber of G vectors = {len(G_q)},')

        #Creating hdf5 file to store large intermediate array
        eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'w') #eps_delta with be stored to hdf5 for each energy

        E_per_chunk, G_per_chunk = get_hdf5_chunks(N_E, N_G, 2**30) # want to read/write in 1 GB chunks later
        eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, G_per_chunk, G_per_chunk))

        if rank == 0 or rank == None:
            logger.info(f'\t\tAn hdf5 dataset has been created to store intermediate results. The shape is (N_E={N_E}, N_G={N_G}, N_G={N_G}) saved in chunks of ({E_per_chunk}, {G_per_chunk}, {G_per_chunk}).')

        eps_delta_h5.close()

        # check if q = 0 - if so, then special calculation needs to be done for G = 0. Other G are calculated normally
        if (q == 0.).all():
            q_dir = np.array(q_dir)
            q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector

            prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2

            eps_im_head = prefactor_head * RPA_head(cell, kpts, q_dir, mo_en_val, mo_en_con, mo_coeff_val, mo_coeff_con) #(E,)



        #if N_q_p == 2: # build eps(-q) using inversion symmetry and then bin as well
"""

def RPA_LFE_0q(dark_objects, q_dir, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop):
    # This is still a work in process
    N_AO = len(dark_objects['aos'])
    valbot, valtop, conbot, contop = np.load(parmt.store + '/bands.npy')

    dft_path = parmt.store + '/DFT/'
    mo_coeff = np.load(dft_path + 'mo_coeff_i.npy')
    mo_en = np.load(dft_path + 'mo_en_i.npy')
    kpts = np.load(parmt.store + '/k-pts_i.npy')

    N_k = kpts.shape[0]
    VCell = dark_objects['V_cell'] 
    cell = dark_objects['cell']

    q_cuts = dark_objects['R_cutoff_q_points']
    G_vectors = dark_objects['G_vectors']
    unique_q = dark_objects['unique_q']
    N_q = len(unique_q)
    primgauss_arr, AO_arr, coeff_arr = dark_objects['block_arrays']
    unique_Ri = load_unique_R()

    # Generating bins
    bin_centers = binning.gen_bin_centers(parmt.q_max, parmt.q_min, parmt.dq, parmt.N_theta, parmt.N_phi)
    N_ang_bins = parmt.N_phi*(parmt.N_theta-2) + 2

    # Generate optimal path for 3D overlaps calculation
    einsum_path = np.einsum_path('kbj,kba,kai->kij', mo_coeff, np.ones((N_k, N_AO, N_AO)), mo_coeff)[0]

    # Make working directory
    if rank == None:
        working_dir = parmt.store + '/working_dir'
    else: #MPI
        working_dir = parmt.store + f'/working_dir_rank{rank}'
    makedir(working_dir)
    makedir(working_dir + '/eta_qG')

    if q_start == None:
        q_start = 0
    if q_stop == None:
        q_stop = N_q

    if rank == 0 or rank == None:
        logger.info(f'Total number of q: {N_q}.')
    if rank == 0: #MPI
        logger.info(f'{q_stop - q_start} q will be calculated per node. Only rank 0 will be logged.')

    q_keys = list(unique_q.keys())[q_start:q_stop]

    tot_bin_eps_im = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights = np.zeros(bin_centers.shape[0]+N_ang_bins)

    tot_bin_eps_re = np.zeros((bin_centers.shape[0]+N_ang_bins, int(2*parmt.E_max/parmt.dE)+1))
    tot_bin_weights_re = np.zeros(bin_centers.shape[0]+N_ang_bins)

    N_E = int(parmt.E_max/parmt.dE*2 + 1) # number of energies to calculate eps at. This in includes -E_max <= E <= E_max
    N_E_p = int(parmt.E_max/parmt.dE + 1) # number of positive energies to calculate eps at. This in includes 0 <= E <= E_max since we can utilize inversion symmetry

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
        E_per_chunk, G_per_chunk = get_hdf5_chunks(N_E, N_G, 2**30) # want to read/write in 1 GB chunks later
        eps_delta_h5.create_dataset('eps_delta', (N_E, N_G, N_G), dtype='complex', chunks=(E_per_chunk, G_per_chunk, G_per_chunk))

        if rank == 0 or rank == None:
            logger.info(f'\t\tAn hdf5 dataset has been created to store intermediate results. The shape is (N_E={N_E}, N_G={N_G}, N_G={N_G}) saved in chunks of ({E_per_chunk}, {G_per_chunk}, {G_per_chunk}).')

        eps_delta_h5.close()

        # check if q = 0 - if so, then special calculation needs to be done for G = 0. Other G are calculated normally
        if (q == 0.).all():
            q_dir = np.array(q_dir)
            q_dir = q_dir/np.linalg.norm(q_dir) # normalize to get unit vector

            prefactor_head = 8.*(np.pi**2)*(alpha**4)*me/(VCell*N_k*parmt.dE) * (alpha*me)**2
            prefactor_wings = 8.*(np.pi**2)*(alpha**3)*me/(VCell*N_k)/parmt.dE * alpha*me
            prefactor_body = 8.*(np.pi**2)*(alpha**2)*me/(VCell*N_k)/parmt.dE

            # Calculating head for 0 <= E <= E_max
            eps_im_head = prefactor_head * RPA_head(cell, kpts[k_pairs[:,1]], q_dir, mo_en[k_pairs[:,0],valbot:valtop+1], mo_en[k_pairs[:,1],conbot:contop+1], mo_coeff[k_pairs[:,0],:,valbot:valtop+1], mo_coeff[k_pairs[:,1],:,conbot:contop+1]) #(E,)

            # Head for -E_max <= E <= E_max
            eps_delta_head = np.concatenate((-1*np.flip(eps_im_head)[:-1], eps_im_head)).astype('complex') #(E,)

            # For E >= 0, initial states are occupied and final states are unoccupied
            if rank == 0 or rank == None:
                logger.info(f'\t\t\tStarting calculation of delta part of polarizability for q = [0,0,0] and 0 < E <= {parmt.E_max} eV')
            start_time = time.time()

           # Wings
            eps_delta_wings_p = prefactor_wings * RPA_wings(G_q[1:], cell, kpts, q_dir, mo_en[k_pairs[:,0],valbot:valtop+1], mo_en[k_pairs[:,1],conbot:contop+1], mo_coeff[k_pairs[:,0],:,valbot:valtop+1], mo_coeff[k_pairs[:,1],:,conbot:contop+1], primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, einsum_path) #(E,G) 

            # Body
            im_delE = delta_energy(mo_en[k_pairs[:,0],valbot:valtop+1], mo_en[k_pairs[:,1],conbot:contop+1]) #(k,2,a,b) # Calculating the delta function in energy
            RPA_body_LFE(G_q[1:], kpts[k_pairs[:,1]], mo_coeff[k_pairs[:,0],:,valbot:valtop+1],  mo_coeff[k_pairs[:,1],:,conbot:contop+1].conj(), im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor_body, neg_E=False, N_G_skip=1)

            if rank == 0 or rank == None:
                logger.info(f'\t\t\tFinished calculation of delta part of polarizability for q = [0,0,0] and 0 < E <= {parmt.E_max} eV. Time taken = {(time.time() - start_time):.2f} s')

            # For E < 0, initial states are unoccupied and final states are occupied
            if rank == 0 or rank == None:
                logger.info(f'\t\t\tStarting calculation of delta part of polarizability for q = [0,0,0] and -{parmt.E_max} <= E <= 0 eV')
            start_time = time.time()

            eps_delta_wings_n = -1 * prefactor_wings * RPA_wings(G_q[1:], cell, kpts, q_dir, mo_en[k_pairs[:,0],conbot:contop+1], mo_en[k_pairs[:,1],valbot:valtop+1], mo_coeff[k_pairs[:,0],:,conbot:contop+1], mo_coeff[k_pairs[:,1],:,valbot:valtop+1], primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, einsum_path)

            # Body
            im_delE = delta_energy(mo_en[k_pairs[:,0],conbot:contop+1], mo_en[k_pairs[:,1],valbot:valtop+1]) #(k,2,a,b) # Calculating the delta function in energy
            RPA_body_LFE(G_q[1:], kpts[k_pairs[:,1]], mo_coeff[k_pairs[:,0],:,conbot:contop+1],  mo_coeff[k_pairs[:,1],:,valbot:valtop+1].conj(), im_delE, primgauss_arr, AO_arr, coeff_arr, q_cuts, unique_Ri, einsum_path, working_dir, rank, prefactor_body, neg_E=True, N_G_skip=1)

            # Concatenate wings along energy and write to hdf5 along with head
            eps_delta_wings = np.concatenate((np.flip(eps_delta_wings_n, axis=0)[:-1], eps_delta_wings_p), axis=0) #(E,G)

            eps_delta_h5 = h5py.File(working_dir + '/eps_delta.h5', 'r+')
            eps_delta = eps_delta_h5['eps_delta']

            eps_delta[:,0,0] = eps_delta_head

            eps_delta[:,0,1:] = eps_delta_wings.conj()
            eps_delta[:,1:,0] = eps_delta_wings

            eps_delta_h5.close()

            if rank == 0 or rank == None:
                logger.info(f'\t\t\tFinished calculation of delta part of polarizability for q = [0,0,0] and -{parmt.E_max} <= E <= 0 eV. Time taken = {(time.time() - start_time):.2f} s')

        else: # normal LFE calculation with k_f = k_i
            RPA_LFE_q(q, mo_en, mo_en, mo_coeff.conj(), mo_coeff, kpts, k_pairs, primgauss_arr, AO_arr, coeff_arr, q_cuts, VCell, G_q, unique_Ri, einsum_path, working_dir, rank) # calculate and store delta part of polarizability (E,G,G')

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
'''