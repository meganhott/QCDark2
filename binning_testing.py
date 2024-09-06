from binning import *
from dielectric_functions import *

cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
q = np.array(list(dark_objects['unique_q'].keys())[0])
G_q = dark_objects['G_vectors'][np.linalg.norm(q+dark_objects['G_vectors'], axis=1) < parmt.q_max]

bin_centers = gen_bin_centers()
#need to add extra #angular bins "junk" entries to binned epsilon and weights for numba optimization
tot_bin_eps = np.zeros((bin_centers.shape[0]+(parmt.N_phi*(parmt.N_theta-2)+2), int(parmt.E_max/parmt.dE)+1), dtype='complex')
tot_bin_weights = np.zeros(bin_centers.shape[0]+(parmt.N_phi*(parmt.N_theta-2)+2))

eps_q = np.ones((G_q.shape[0], int(parmt.E_max/parmt.dE)+1)) + 1j*np.zeros((G_q.shape[0], int(parmt.E_max/parmt.dE)+1))

#First call to binning function will be slow
tot_bin_eps, tot_bin_weights = bin_eps_q(q, G_q, eps_q, bin_centers, tot_bin_eps, tot_bin_weights)

#All calls to binning function after first one will be faster due to njit optimization 
tot_bin_eps, tot_bin_weights = bin_eps_q(q, G_q, eps_q, bin_centers, tot_bin_eps, tot_bin_weights)

#print(bin_centers[np.where(tot_bin_weights != 0)[0]])


