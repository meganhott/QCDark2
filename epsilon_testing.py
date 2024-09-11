from dielectric_functions import *
import numpy as np
import input_parameters as parmt
from routines import *

"""
#testing epsilon function (without dynamic R_cutoff implementation)
cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)

tot_bin_eps, tot_bin_weights = initialize_RPA_dielectric(dark_objects, test=True)

binned_eps = get_binned_epsilon(tot_bin_eps, tot_bin_weights)

np.save(parmt.store+'/binned_eps3.npy', binned_eps)
"""

#for running test calculation of epsilon on cluster
cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
dark_objects['R_cutoffs'] = routines.primgauss_1D_overlaps(dark_objects)
dark_objects['R_cutoff_q_points'] = routines.store_Rids(dark_objects)

tot_bin_eps, tot_bin_weights = initialize_RPA_dielectric(dark_objects)

#np.save('/gpfs/scratch/mhott/tot_bin_eps.npy', tot_bin_eps)
#np.save('/gpfs/scratch/mhott/tot_bin_weights.npy', tot_bin_weights)

binned_eps = get_binned_epsilon(tot_bin_eps, tot_bin_weights)
np.save('/gpfs/scratch/mhott/binned_eps.npy', binned_eps)


"""
with mp.get_context('fork').Pool(mp.cpu_count()) as p:  #parallelization over G
    chi = p.map(partial(RPA_dielectric, q=q, k_f=k_f_q, mo_coeff_i=mo_coeff_i_q, mo_coeff_f_conj=mo_coeff_f_q_conj, re_delE=re_delE, im_delE=im_delE, R_id=R_id, unique_Ri=unique_Ri, blocks=blocks, N_AO=N_AO), G_q)
    chi = np.array(chi) 
np.save('/gpfs/scratch/mhott/unbinned_chi.npy', chi)

tot_bin_chi, tot_bin_weights = bin.bin_eps_q(q, G_q, chi, bin_centers, tot_bin_chi, tot_bin_weights)
np.save('/gpfs/scratch/mhott/binned_chi.npy', tot_bin_chi)
"""

#eta_qG = get_3D_overlaps_blocks(q, G, k_f_q, blocks, N_AO, R_id, unique_Ri, mo_coeff_i_q, mo_coeff_f_q_conj) #625ms

#chi = RPA_susceptibility(eta_qG, eta_qG, re_delE, im_delE)
#eps = 1 - (4*np.pi)**2 / (np.dot(q,q)+np.dot(G,G)) * RPA_susceptibility(eta_qG, eta_qG, re_delE, im_delE) #7.8ms

#eps = RPA_dielectric(G, q, k_f_q, mo_coeff_i_q, mo_coeff_f_q_conj, re_delE, im_delE, R_id, unique_Ri, blocks, N_AO) #for one G (non-lfe)
