from dielectric_functions import *
import input_parameters as parmt
import numpy as np
from routines import time_wrapper

num_G = 100

def get_3D_overlaps_einsum(qG, k_f, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, opt):
    """
    Work in progress
    To Do:
    - Optimize!
    - Look into ao_coefficients (Megan pointed out that we should multiply them after the product.)

    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Inputs:
        qG:             np.ndarray of shape (3,): q + G for one q vector in 1BZ and one G vector
        k_f:            np.ndarray of shape (N_kpairs, 3):
        mo_coeff_i:     np.ndarray of shape (N_kpairs, N_val_bands, N_AO)
        mo_coeff_f:     np.ndarray of shape (N_kpairs, N_con_bands, N_AO)
        R_id:           np.ndarray of shape (3, N_R_vectors)
        unique_Ri:      np.ndarray of shape (3, N_R_unique
    Outputs:
        eta_qG:         np.ndarray of shape (N_kpairs, N_val_bands, N_con_bands): all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    Ri_coef_sum = np.zeros((R_id.shape[0], R_id.shape[1], mo_coeff_i.shape[0], mo_coeff_i.shape[2], mo_coeff_f.shape[2]), dtype = np.complex128) #(3,R_vec,k_pairs,a,b)
    for dim in range(3):
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        q_1d_integrals = np.load(dir+'{:.5f}.npy'.format(qG[dim]))
        phase = np.exp(-1j*np.tensordot(unique_Ri[dim],k_f[:,dim], axes=0)) #(Ru, k_pair)
        coef_sum = np.einsum('Rk,Rab->Rkab', phase, q_1d_integrals, optimize=opt[0]) 
        Ri_coef_sum[dim] = coef_sum[R_id[dim]]
    eta_qG = np.sum(np.prod(Ri_coef_sum, axis=0),axis=0) #(k_pair,a,b)
    # Now we should do molecular orbital coefficients.
    #eta_qG = np.einsum('kab,kia,kjb->kij', eta_qG, mo_coeff_i, mo_coeff_f.conj(),optimize=opt[1])
    #logging.info('All {} 3D overlaps generated for q+G vector {}. eta_qG is {:.3f} MB in memory.'.format(np.prod(eta_qG.shape), list(map(lambda qG :str(qG),qG.round(5))), sys.getsizeof(eta_qG)/10**6))
    return eta_qG

def get_3D_overlaps_tensordot(qG, k_f, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, opt):
    Ri_coef_sum = np.zeros((R_id.shape[0], R_id.shape[1], mo_coeff_i.shape[0], mo_coeff_i.shape[2], mo_coeff_f.shape[2]), dtype = np.complex128) #(3,R_vec,k_pairs,a,b)
    for dim in range(3):
        dir = parmt.store + '/primgauss_1d_integrals/dim_{}/'.format(dim)
        q_1d_integrals = np.load(dir+'{:.5f}.npy'.format(qG[dim]))
        phase = np.exp(-1j*np.tensordot(unique_Ri[dim],k_f[:,dim], axes=0)) #(Ru, k_pair)
        coef_sum = np.empty(Ri_coef_sum.shape[1:],dtype='complex')
        for i in range(unique_Ri[dim].shape[0]):
            coef_sum[i] = np.tensordot(phase[i], q_1d_integrals[i], axes=0) #outer product (k),(ab) -> (kab)
        Ri_coef_sum[dim] = coef_sum[R_id[dim]]
    eta_qG = np.sum(np.prod(Ri_coef_sum, axis=0),axis=0) #(k_pair,a,b)
    # Now we should do molecular orbital coefficients.
    #eta_qG1 = np.empty((k_f.shape[0],mo_coeff_i.shape[1],mo_coeff_f.shape[1]), dtype = np.complex128)
    #for k in range(k_f.shape[0]):
    #    eta_qG1[k] = np.tensordot(np.tensordot(mo_coeff_i[k],eta_qG[k], axes=(1,0)), mo_coeff_f.conj()[k], axes=(1,1)) #Inner products over a and b
    return eta_qG

def get_3D_overlaps_numerical(qG, ki, kf, mo_coeff_i, mo_coeff_f):
    coords, weights = routines.pbcdft.gen_grid.gen_becke_grids(cell)
    operator = np.exp(1.j * np.einsum('wx, x -> w', coords, qG))
    print(operator)
    ao1 = routines.pbcdft.numint.eval_ao(cell, coords, kpt = ki)
    ao2 = routines.pbcdft.numint.eval_ao(cell, coords, kpt = kf)
    ao_ovlp = np.einsum('w, wi, w, wj -> ij', weights, ao2.conj(), operator, ao1)
    #print(ao_ovlp.shape)
    #return np.einsum('ai,ij,jb->ab', mo_coeff_f.T.conj(), ao_ovlp, mo_coeff_i)
    return ao_ovlp

@time_wrapper
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

num_ovlp = []
for i in range(len(k_pairs)):
    num_ovlp.append(get_3D_overlaps_numerical(qG, k1[i], k2[i], mo_coeff_i[i], mo_coeff_f[i]))

td_ovlp = get_3D_overlaps_tensordot(qG, k2, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, None)
es_ovlp = get_3D_overlaps_einsum(qG, k2, mo_coeff_i, mo_coeff_f, R_id, unique_Ri, ('optimal', 'optimal'))
aos = dark_objects['aos']
alt_ovlp = get_3D_overlaps_alternate(qG, k2, aos, R_id, unique_Ri)
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