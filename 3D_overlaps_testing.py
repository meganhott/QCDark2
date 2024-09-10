from dielectric_functions import *
import input_parameters as parmt
import numpy as np
from routines import time_wrapper
import matplotlib.pyplot as plt

@time_wrapper
def get_3D_overlaps_blocks(qG, k2: np.ndarray, blocks: dict, unique_Ri: list[np.ndarray], n: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, q_cuts: np.ndarray) -> np.ndarray:
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): #435us total
        ints.append(np.load('test_resources/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,None,:,:] * np.exp(-1.j*unique_Ri[d][:,None]*k2[None,:,d])[:,:,None,None])
    ovlp = np.empty((k2.shape[0], n, n), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,:,:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)
            tot = np.ones((R_id.shape[1], k2.shape[0], p2.shape[1], p1.shape[1]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][:,:,p2[d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)
            for i in d1:
                for j in d2:
                    ovlp[:,i,j] = (tot@d1[i])@d2[j]
    return np.einsum('kia,kij,kjb->kab', mo_coeff_f_conj, ovlp, mo_coeff_i, optimize = True)

cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
#already have all 1D overlaps generated: do not need to run function

#testing for one q vector
q = list(dark_objects['unique_q'].keys())[0]
k_pairs = np.array(dark_objects['unique_q'][q])

#testing for selected G vectors
num_G = 1
G_vectors = dark_objects['G_vectors'][:num_G,:]

routines.get_band_indices()

ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')

dft_path = parmt.store + '/DFT/'
mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[k_pairs[:,0]]#[:,:,ivalbot:ivaltop+1]
mo_coeff_f_conj = np.load(dft_path + 'mo_coeff_f.npy')[k_pairs[:,1]].conj()#[:,:,iconbot:icontop+1]
mo_en_i = np.load(dft_path + 'mo_en_i.npy')
mo_en_f = np.load(dft_path + 'mo_en_f.npy')
k2 = np.load(parmt.store + '/k-pts_f.npy')[k_pairs[:,1]]
k1 = np.load(parmt.store + '/k-pts_i.npy')[k_pairs[:,0]]
R_id, unique_Ri = routines.load_unique_R(dark_objects['R_vectors'])

blocks = dark_objects['blocks']
aos = dark_objects['aos']
n = len(aos)

qG = np.array(q)
#ovlp_alt = get_3D_overlaps_alternate(qG, k2, dark_objects['aos'], R_id, unique_Ri)
#ovlp_blk = get_3D_overlaps_blocks(qG, k2, dark_objects['blocks'], unique_Ri, n, mo_coeff_i, mo_coeff_f_conj)

from rpa_dielectric_cython import get_3D_overlaps_blocks as cython_3d_overlaps

