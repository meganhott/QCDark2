from dielectric_functions import *
import input_parameters as parmt
import numpy as np
from routines import time_wrapper

num_G = 100

def get_3D_overlaps_numerical(qG, ki, kf, mo_coeff_i, mo_coeff_f):
    coords, weights = routines.pbcdft.gen_grid.gen_becke_grids(cell)
    operator = np.exp(1.j * np.einsum('wx, x -> w', coords, qG))
    ao1 = routines.pbcdft.numint.eval_ao(cell, coords, kpt = ki)
    ao2 = routines.pbcdft.numint.eval_ao(cell, coords, kpt = kf)
    ao_ovlp = np.einsum('w, wi, w, wj -> ij', weights, ao2.conj(), operator, ao1)
    #print(ao_ovlp.shape)
    #return np.einsum('ai,ij,jb->ab', mo_coeff_f.T.conj(), ao_ovlp, mo_coeff_i)
    return ao_ovlp

def get_3D_overlaps_alternate(qG, k2, aos, R_id, unique_Ri):
    new_ovlp = np.zeros((8, 38, 38), dtype = np.complex128)
    for k in range(len(k2)):
        for i in range(38):
            for j in range(38):
                aoi, aoj = aos[i], aos[j]
                tot = np.ones((R_id.shape[1], len(aoj.coef), len(aoi.coef)), dtype = np.complex128)
                for d in range(3):
                    integral = np.load('test_resources/primgauss_1d_integrals_0/dim_{}/{:.5f}.npy'.format(d, qG[d]))
                    vals = integral[:, aoj.prim_indices[d]][:,:,aoi.prim_indices[d]]
                    phs = np.exp(-1.j*k2[k,d]*unique_Ri[d])
                    vals = phs[:,None,None]*vals
                    tot *= vals[R_id[d]]
                tot = tot.sum(axis = 0)
                tot *= aoj.coef[:,None]*aoi.coef[None,:]*aoj.norm[:,None]*aoi.norm[None,:]
                new_ovlp[k, i,j] = np.sum(tot)
    return new_ovlp

def get_3D_overlaps_alt(qG, k2, aos, R_id, unique_Ri):
    ints = []
    for d in range(3):
        ints.append(np.load('test_resources/primgauss_1d_integrals_0/dim_{}/{:.5f}.npy'.format(d, qG[d]))[None,R_id[d],:,:]*np.exp(-1.j*k2[:,None,d]*unique_Ri[d][None,:])[:,R_id[d],None,None])
    n = len(aos)
    ovlp = np.zeros((k2.shape[0], n, n), dtype = np.complex128)
    for i in range(n):
        aoi = aos[i]
        for j in range(n):
            aoj = aos[j]
            tot = np.ones((k2.shape[0], R_id.shape[1], len(aoj.coef), len(aoi.coef)), dtype = np.complex128)
            for d in range(3):
                tot *= ints[d][:,:,aoj.prim_indices[d]][:,:,:,aoi.prim_indices[d]]
            tot = tot.sum(axis = 1)
            tot *= aoj.coef[None,:,None]*aoi.coef[None,None,:]*aoj.norm[None,:,None]*aoi.norm[None,None,:]
            tot = tot.sum(axis = 1)
            tot = tot.sum(axis = 1)
            ovlp[:,i,j] = tot
    return ovlp

@time_wrapper
def all_3D_overlaps(f):
    for G in G_vectors:
        eta_qG = f[0](np.array(q)+G, k2[k_pairs[:,1]], mo_coeff_i[k_pairs[:,0]], mo_coeff_f[k_pairs[:,1]], R_id, unique_Ri, f[1])


cell, dark_objects = initialize_cell()
dark_objects['unique_q'] = routines.get_1BZ_q_points(cell)
#already have all 1D overlaps generated: do not need to run function

#testing for one q vector
q = list(dark_objects['unique_q'].keys())[0]
k_pairs = np.array(dark_objects['unique_q'][q])

#testing for selected G vectors
G_vectors = dark_objects['G_vectors'][:num_G,:]

routines.get_band_indices()

ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
dft_path = parmt.store + '/DFT/'
mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')[k_pairs[:,0]]
mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')[k_pairs[:,1]]
mo_en_i = np.load(dft_path + 'mo_en_i.npy')
mo_en_f = np.load(dft_path + 'mo_en_f.npy')
k2 = np.load(parmt.store + '/k-pts_f.npy')[k_pairs[:,1]]
k1 = np.load(parmt.store + '/k-pts_i.npy')[k_pairs[:,0]]
R_id, unique_Ri = routines.load_unique_R(dark_objects['R_vectors'])

qG = np.array(q)

#num_ovlp = []
#for i in range(len(k_pairs)):
#    num_ovlp.append(get_3D_overlaps_numerical(qG, k1[i], k2[i], mo_coeff_i[i], mo_coeff_f[i]))

#td_ovlp = get_3D_overlaps_tensordot(qG, k2, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, None)
#es_ovlp = get_3D_overlaps_einsum(qG, k2, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, ('optimal', 'optimal'))
aos = dark_objects['aos']
alt_ovlp1 = get_3D_overlaps_alternate(qG, k2, aos, R_id, unique_Ri)
alt_ovlp2 = get_3D_overlaps_alt(qG, k2, aos, R_id, unique_Ri)
"""
for f in zip([get_3D_overlaps_einsum, get_3D_overlaps_tensordot], [('optimal','optimal'),None]):
    all_3D_overlaps(f)
"""
"""
for G in G_vectors:
    eta_qG = routines.get_3D_overlaps(np.array(q)+G, k2[k_pairs[:,1]], mo_coeff_i[k_pairs[:,0]], mo_coeff_f[k_pairs[:,1]], R_id, unique_Ri)
"""

#print(eta_qG.shape)
#np.save('/gpfs/scratch/mhott/eta_q.npy',eta_q)