from dielectric_functions import *
import input_parameters as parmt
import numpy as np
import multiprocessing as mp
from functools import partial
from RPA_dielectric_testing_helper import *

num_G = 1000

@routines.time_wrapper
def initialize_RPA_dielectric_test(dark_objects):
    """
    Loads DFT parameters, selects epsilon routine (LFE vs non-LFE), and calculates binned RPA dielectric function, epsilon(q,E).
    """

    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    dft_path = parmt.store + '/DFT/'
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[:,ivalbot:ivaltop+1,:]
    mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')[:,iconbot:icontop+1,:]
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop+1]
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop+1]
    k_f = np.load(parmt.store + '/k-pts_f.npy')

    unique_q = {list(dark_objects['unique_q'].keys())[0]: dark_objects['unique_q'].get(list(dark_objects['unique_q'].keys())[0])} #testing for one q vector 
    G_vectors = dark_objects['G_vectors'][:num_G,:]
    R_vectors = dark_objects['R_vectors']

    R_id, unique_Ri = routines.load_unique_R(R_vectors)

    #initialize bins
    
    if parmt.include_lfe:
        RPA_eps = routines.RPA_dielectric_lfe
    else: #currently only implemented for non-LFE
        RPA_eps = routines.RPA_dielectric

    for q in unique_q.keys():
        k_pairs = np.array(unique_q[q])

        #Only include G_vectors such that |q + G| < q_max
        q = np.array(q)
        G_q = G_vectors[np.linalg.norm(q+G_vectors, axis=1) < parmt.q_max]

        #generate parameters for q
        mo_en_i_q = mo_en_i[k_pairs[:,0]] #(k_pair,i)
        mo_en_f_q = mo_en_f[k_pairs[:,1]] #(k_pair,j)
        mo_coeff_i_q = mo_coeff_i[k_pairs[:,0]] #(k_pair,i,a)
        mo_coeff_f_q = mo_coeff_f[k_pairs[:,1]] #(k_pair,j,b)
        k_f_q = k_f[k_pairs[:,1]]

        #Compute epsilon for all G_q
        with mp.get_context('fork').Pool(mp.cpu_count()) as p:  #parallelization over G
            eps = p.map(partial(RPA_eps, q=q, mo_en_i=mo_en_i_q, mo_en_f=mo_en_f_q, k_f=k_f_q, mo_coeff_i=mo_coeff_i_q, mo_coeff_f=mo_coeff_f_q, R_id=R_id, unique_Ri=unique_Ri), G_q)
        eps = np.array(eps) #(G, E)
        #Implement binning with eps, G_q as input
        routines.logging.info('epsilon_GG(q, E) calculated for {0} G vectors and {1} energies for 1BZ q vector {2}. eps is {3:.3f} MB in memory.'.format(eps.shape[0], eps.shape[1], tuple(q), routines.sys.getsizeof(eps)/10**6))




def dielectric_function_no_lfe(dark_objects: dict):

    pass

def dielectric_function_lfe(dark_objects: dict):
    pass

def RPA_dielectric_function(dark_objects: dict):
    if parmt.include_lfe:
        dielectric_function_lfe(dark_objects)
    else:
        dielectric_function_no_lfe(dark_objects)
    return

cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)

initialize_RPA_dielectric_test(dark_objects)