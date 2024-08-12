from binning import *
from dielectric_functions import *

num_G = 3

cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
q = np.array(list(dark_objects['unique_q'].keys())[0])
G_q = dark_objects['G_vectors'][np.linalg.norm(q+dark_objects['G_vectors'], axis=1) < parmt.q_max][:num_G]

bin_centers = gen_bin_centers()
tot_bin_eps = np.zeros((bin_centers.shape[0], int(parmt.E_max/parmt.dE)+1))
tot_bin_weights = np.zeros(bin_centers.shape[0])

eps_q = np.ones((G_q.shape[0], int(parmt.E_max/parmt.dE)+1))

tot_bin_eps, tot_bin_weights = bin_eps_q(q, G_q, eps_q, bin_centers, tot_bin_eps, tot_bin_weights)

print(bin_centers[np.where(tot_bin_weights != 0)[0]])


