from dielectric_functions import *
import input_parameters as parmt
import numpy as np
from routines import time_wrapper
import matplotlib.pyplot as plt
from numba import jit, njit

@time_wrapper
def get_3D_overlaps_numerical(qG, ao1, ao2):
    coords = np.load(parmt.store + '/coords.npy')
    coords, weights = coords[:,:3], coords[:,3]
    operator = np.exp(1.j * np.einsum('wx, x -> w', coords, qG))
    ao_ovlp = np.einsum('w, kwi, w, kwj -> kij', weights, ao2.conj(), operator, ao1, optimize = 'optimal')
    #print(ao_ovlp.shape)
    #return np.einsum('ai,ij,jb->ab', mo_coeff_f.T.conj(), ao_ovlp, mo_coeff_i)
    return np.array(ao_ovlp)

@time_wrapper
def get_3D_overlaps(qG, k2, aos, R_id, unique_Ri):
    """
    - Keep ints in terms of unique_R instead of R_vectors - significant (~20x) speedup (60s to 3.3s)
    - Use @/numpy array operations instead of broadcasting - small (~2%) speedup (~0.05s)
    - Broadcasting method is still fastest for ints/phase
    """
    ints = []
    for d in range(3): #435us total
        ints.append(np.load('test_resources/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,None,:,:] * np.exp(-1.j*unique_Ri[d][:,None]*k2[None,:,d])[:,:,None,None]) #145us
    n = len(aos)
    ovlp = np.empty((k2.shape[0], n, n), dtype = np.complex128) #0.25us
    for i in range(n): #16s total - overestimate since some AO only have one primgauss
        aoi = aos[i]
        for j in range(n): #11.2ms per (i,j)
            aoj = aos[j]
            tot = np.ones((R_id.shape[1], k2.shape[0], len(aoj.coef), len(aoi.coef)), dtype = np.complex128) #614us
            for d in range(3):
                ints_ij = ints[d][:,:,aoj.prim_indices[d]][:,:,:,aoi.prim_indices[d]] #123us
                tot *= ints_ij[R_id[d]] #2.75ms
            tot = tot.sum(axis = 0) #sum over R #642us
            tot = (tot @ (aoi.coef)) @ (aoj.coef) #multiply by coefficients and sum over m,n #1.58ms
            ovlp[:,i,j] = tot
    return ovlp

@time_wrapper
def get_3D_overlaps_blocks(qG, k2, blocks, R_id, unique_Ri, n):
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
    return ovlp

@time_wrapper
@jit
def get_3D_overlaps_jit(integrals, k2, ao_prim_indices, ao_coeff, R_id, unique_Ri):
    ints = []
    for d in range(3):
        ints.append(integrals[d][None,R_id[d],:,:]*np.exp(-1.j*k2[:,None,d]*unique_Ri[d][None,:])[:,R_id[d],None,None])
    n = len(ao_coeff)
    ovlp = np.zeros((k2.shape[0], n, n), dtype = np.complex128)
    for i in range(n):
        for j in range(n):
            tot = np.ones((k2.shape[0], R_id.shape[1], len(ao_coeff[j]), len(ao_coeff[i])), dtype = np.complex128)
            for d in range(3):
                tot *= ints[d][:,:,ao_prim_indices[j][d]][:,:,:,ao_prim_indices[i][d]]
            tot = tot.sum(axis = 1)
            tot *= ao_coeff[j][None,:,None]*ao_coeff[i][None,None,:]
            tot = tot.sum(axis = 1)
            tot = tot.sum(axis = 1)
            ovlp[:,i,j] = tot
    return ovlp


def get_3D_overlaps2(qG, k2, ao_coeff, ao_bool, R_id, unique_Ri):
    """
    Doesn't work: requires too much memory (650 GB) for last line in dim loop

    make two AO arrays: 
    Boolean such that primitive gaussians are picked out #(38, 62) - used to get correct 1d integrals
    coefficient matrix with just primgauss shells (1s, 2s, 1d, 2d, etc.) - doesn't depend on dimension #(38, 42)

    Or keep coeff matrix (38,62) but have all coefficients filled in for shell with multiple possible l
    Boolean: (3,38,62)
    Coeff: (38,62)
    """
    primgauss_integrals = np.empty((3,R_id.shape[1],k2.shape[0],ao_coeff.shape[1],ao_coeff.shape[1]), dtype='complex') #(3,Rvec,k,m,n)
    for d in range(3):
        integrals = np.load('test_resources/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d])) #(R,m,n)
        phs = np.exp(-1.j*np.tensordot(unique_Ri[d], k2[:,d], axes=0)) #(R,k)
        x = np.einsum('Rk,Rmn->Rkmn', phs, integrals)
        primgauss_integrals[d] = np.einsum('Rkmn,am,bn->Rkambn', x[R_id[d]], ao_bool[d], ao_bool[d]) #sets unused integrals to 0 for each AO
    primgauss_integrals = np.prod(primgauss_integrals, axis=0) #(R,k,a,m,b,n)
    eta_q = np.einsum('Rkambn,am,bn->kab', primgauss_integrals, ao_coeff, ao_coeff)
    return eta_q

def load_1D_integrals(qG):
    integrals = []
    for d in range(3):
        integrals.append(np.load('test_resources/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))) #(R,m,n)
    return integrals

@time_wrapper
@njit
def get_3D_overlaps2_njit(integrals, k2, ao_coeff, ao_bool, R_id, unique_Ri):
    """
    njit version of overlaps2
    - Loop over a and b to reduce memory required
    - All inputs are arrays so we can use njit
    - Super slow, even after first run
    """
    eta_qG = np.zeros((k2.shape[0], ao_coeff.shape[0], ao_coeff.shape[0]), dtype = np.complex128)

    intphase = []
    for d in range(3):
        phase = np.exp(-1.j*unique_Ri[d][:,None]*k2[None,:,d]) #(R,k)
        intphase.append(phase[:,:,None,None]*integrals[d][:,None]) #(d,R,k,m,n)

    for a, a_coeff in enumerate(ao_coeff):
        for b, b_coeff in enumerate(ao_coeff):
            tot = np.ones((R_id.shape[1], k2.shape[0], ao_coeff.shape[1], ao_coeff.shape[1]), dtype = np.complex128) #(Rvec,k,62,62)
            for d in range(3):
                x = intphase[d]*ao_bool[d,a,None,None,:,None]*ao_bool[d,b,None,None,None,:] #(R,k,m,n) #sets unused integrals to 0
                #x = phase[:,None,None]*x #(R,m,n)
                tot *= x[R_id[d]] #(Rvec,k,m,n) #multiply over dim
            tot = np.sum(tot, axis=0) #sum over Rvec #(k,m,n)
            tot *= a_coeff[None,:,None]*b_coeff[None,None,:]
            tot = np.sum(tot, axis=1) #sum over m
            tot = np.sum(tot, axis=1) #sum over n
            eta_qG[:,a,b] = tot
    return eta_qG


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
num_G = 1
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

blocks = dark_objects['blocks']
aos = dark_objects['aos']
n = len(aos)

qG = np.array(q)
#ovlp_alt = get_3D_overlaps_alternate(qG, k2, dark_objects['aos'], R_id, unique_Ri)
ovlp = get_3D_overlaps(qG, k2, dark_objects['aos'], R_id, unique_Ri)
ovlp_blk = get_3D_overlaps_blocks(qG, k2, dark_objects['blocks'], R_id, unique_Ri, n)

coords, weights = routines.pbcdft.gen_grid.gen_becke_grids(cell)
np.save(parmt.store + '/coords', np.transpose(np.append(coords.T, weights[None,:], axis = 0)))
ao1, ao2 = [], []
for k in range(len(k_pairs)):
    ao1.append(routines.pbcdft.numint.eval_ao(cell, coords, kpt = k1[k]))
    ao2.append(routines.pbcdft.numint.eval_ao(cell, coords, kpt = k2[k]))
ao1, ao2 = np.array(ao1), np.array(ao2)
num_ovlp = get_3D_overlaps_numerical(qG, ao1, ao2)

movlp1 = np.einsum('kia,kij,kjb->kab', mo_coeff_f.conj()[:,:,iconbot:icontop+1], ovlp, mo_coeff_i[:,:,ivalbot:ivaltop+1])
movlp2 = np.einsum('kia,kij,kjb->kab', mo_coeff_f.conj()[:,:,iconbot:icontop+1], ovlp_blk, mo_coeff_i[:,:,ivalbot:ivaltop+1])
