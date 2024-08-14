# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

ctypedef np.complex128_t complex128_t

cdef complex128_t I = 1j

def get_3D_overlaps_blocks(np.ndarray[double, ndim=1] qG, 
                           np.ndarray[double, ndim=2] k2, 
                           dict blocks,
                           np.ndarray[short, ndim=2] R_id, 
                           list unique_Ri,
                           int n, 
                           np.ndarray[complex128_t, ndim=3] mo_coeff_i, 
                           np.ndarray[complex128_t, ndim=3] mo_coeff_f_conj):
    
    cdef:
        int d, i, j
        list ints = []
        np.ndarray[complex128_t, ndim=3] ovlp
        np.ndarray[complex128_t, ndim=4] tot
        np.ndarray[complex128_t, ndim=3] summed
        tuple p1, p2
        dict d1, d2
        np.ndarray[double, ndim=1] d1_array, d2_array
    
    for d in range(3):
        ints.append(np.load(f'test_resources/primgauss_1d_integrals/dim_{d}/{qG[d]:.5f}.npy')[:,None,:,:] * 
                    np.exp(-I * unique_Ri[d][:,None] * k2[None,:,d])[:,:,None,None])
    
    ovlp = np.empty((k2.shape[0], n, n), dtype=np.complex128)
    
    for p1 in blocks:
        d1 = blocks[p1]
        
        ints_i = [ints[d][:,:,:,p1[d]] for d in range(3)]
        
        for p2 in blocks:
            d2 = blocks[p2]
            
            tot = np.ones((R_id.shape[1], k2.shape[0], len(p2[0]), len(p1[0])), dtype=np.complex128)
            
            for d in range(3):
                tot *= ints_i[d][:,:,p2[d]][R_id[d]]
            
            summed = tot.sum(axis=0)
            
            for i in d1:
                d1_array = np.asarray(d1[i], dtype=np.float64)
                for j in d2:
                    d2_array = np.asarray(d2[j], dtype=np.float64)
                    ovlp[:,i,j] = (summed @ d1_array) @ d2_array
    
    return np.einsum('kia,kij,kjb->kab', mo_coeff_f_conj, ovlp, mo_coeff_i, optimize=True)