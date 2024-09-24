import numpy as np
from dielectric_functions import *
import dark_objects_routines as do_routines
import epsilon_routines as eps_routines
import input_parameters as parmt

new_dft = False

cell, dark_objects = initialize_cell()
if new_dft: #For new DFT
    dark_objects = electronic_structure(cell, dark_objects)
    dark_objects = dielectric_RPA(dark_objects)
else: #For existing DFT results
    dark_objects['unique_q'] = do_routines.get_1BZ_q_points(cell)
    dark_objects['R_cutoffs'] = do_routines.primgauss_1D_overlaps(dark_objects)
    dark_objects['R_cutoff_q_points'] = do_routines.store_R_ids(dark_objects)

tot_bin_eps, tot_bin_weights = eps_routines.get_RPA_dielectric(dark_objects)
binned_eps = eps_routines.get_binned_epsilon(tot_bin_eps, tot_bin_weights)
#np.save(parmt.store+'/binned_eps.npy', binned_eps)