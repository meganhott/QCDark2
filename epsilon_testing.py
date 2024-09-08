from dielectric_functions import *
import numpy as np
import input_parameters as parmt
from routines import *

#testing epsilon function (without dynamic R_cutoff implementation)
cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)

tot_bin_eps, tot_bin_weights = initialize_RPA_dielectric(dark_objects, test=True)

binned_eps = get_binned_epsilon(tot_bin_eps, tot_bin_weights)


"""
#timings done on laptop
cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
q = list(dark_objects['unique_q'].keys())[0]

ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
dft_path = parmt.store + '/DFT/'
mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[:,:,ivalbot:ivaltop+1]
mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')[:,:,iconbot:icontop+1]
mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop+1]
mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop+1]
k_f = np.load(parmt.store + '/k-pts_f.npy')

N_AO = len(dark_objects['aos'])
blocks = dark_objects['blocks']

R_vectors = dark_objects['R_vectors']
R_id, unique_Ri = load_unique_R(R_vectors)

E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE)

unique_q = dark_objects['unique_q']
k_pairs = np.array(unique_q[q])

q = np.array(q)
G = np.array([0.0,0.0,0.0]) #only testing for one G, G'

#generate parameters for q
mo_en_i_q = mo_en_i[k_pairs[:,0]] #(k_pair,i)
mo_en_f_q = mo_en_f[k_pairs[:,1]] #(k_pair,j)
mo_coeff_i_q = mo_coeff_i[k_pairs[:,0]] #(k_pair,a,i)
mo_coeff_f_q_conj = mo_coeff_f[k_pairs[:,1]].conj() #(k_pair,b,j)
k_f_q = k_f[k_pairs[:,1]]

#Compute energy differences for denominator of Re(chi), and Im(chi)
re_delE, im_delE = get_energy_diff(mo_en_i_q, mo_en_f_q, E) #4.8ms

eta_qG = get_3D_overlaps_blocks(q, G, k_f_q, blocks, N_AO, R_id, unique_Ri, mo_coeff_i_q, mo_coeff_f_q_conj) #625ms

#chi = RPA_susceptibility(eta_qG, eta_qG, re_delE, im_delE)
eps = 1 - (4*np.pi)**2 / (np.dot(q,q)+np.dot(G,G)) * RPA_susceptibility(eta_qG, eta_qG, re_delE, im_delE) #7.8ms

#eps = RPA_dielectric(G, q, k_f_q, mo_coeff_i_q, mo_coeff_f_q_conj, re_delE, im_delE, R_id, unique_Ri, blocks, N_AO) #for one G (non-lfe)
"""