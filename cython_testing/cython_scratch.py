from cython_testing import ovlp_cython_np, ovlp_cython_mv, convert_mv
import numpy as np
import time
import sys
sys.path.append('../../dielectric_pyscf')
from dielectric_pyscf.dielectric_functions import initialize_cell

from line_profiler import LineProfiler

import numba as nb

def load():
    primgauss_arr = np.load('primgauss_arr.npy')
    AO_arr = np.load('AO_arr.npy')
    coeff_arr = np.load('coeff_arr.npy')
    ints_x = np.copy(np.transpose(np.load('ints_x.npy'), (1,2,0)), order='C')
    ints_y = np.copy(np.transpose(np.load('ints_y.npy'), (1,2,0)), order='C')
    ints_z = np.copy(np.transpose(np.load('ints_z.npy'), (1,2,0)), order='C')
    R_id = np.load('R_id.npy').T
    return primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id


def ovlp_original2(blocks, N_AO, ints, R_id):
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)
            """
            tot = np.ones((p2.shape[1], p1.shape[1], R_id.shape[0]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][p2[d]]
                tot *= ints_ij[:,:,R_id[:,d]]
            """

            tot = ints_i[0][p2[0]][:,:,R_id[:,0]]*ints_i[1][p2[1]][:,:,R_id[:,1]]*ints_i[2][p2[2]][:,:,R_id[:,2]]
            tot = tot.sum(axis = 2)
            for i in d1:
                for j in d2:
                    ovlp[i,j] = (tot@d1[i])@d2[j]
    return ovlp

def ovlp_original1(blocks, N_AO, ints, R_id):
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 
    for p1 in blocks:
        d1 = blocks[p1]
        p1 = np.array(p1)
        ints_i = []
        for d in range(3):
            ints_i.append(ints[d][:,p1[d]])
        for p2 in blocks:
            d2 = blocks[p2]
            p2 = np.array(p2)
            tot = np.ones((p2.shape[1], p1.shape[1], R_id.shape[0]), dtype = np.complex128)
            for d in range(3):
                ints_ij = ints_i[d][p2[d]]
                tot *= ints_ij[:,:,R_id[:,d]]
            tot = tot.sum(axis = 2)
            for i in d1:
                for j in d2:
                    ovlp[i,j] = (tot@d1[i])@d2[j]
    return ovlp

"""
short[:,:,::1] primgauss_arr, 
cnp.uint8_t[:,::1] AO_arr, 
double complex[:,::1] coeff_arr, 
double complex[:,:,::1] ints_x, 
double complex[:,:,::1]  ints_y, 
double complex[:,:,::1]  ints_z, 
long[:,::1] R_id)
"""

@nb.njit(nb.complex128[:,::1]( nb.boolean[:,:,::1], nb.boolean[:,::1], nb.complex128[:,::1], nb.complex128[:,:,::1], nb.complex128[:,:,::1], nb.complex128[:,:,::1], nb.int64[:,::1] ))
def ovlp_numba(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id):

    N_blocks = primgauss_arr.shape[0] #68
    #N_primgauss = primgauss_arr.shape[1] #86
    N_AO = coeff_arr.shape[0] #78
    #N_maxcoeff = coeff_arr.shape[1] #13

    ovlp = np.empty((N_AO, N_AO), dtype=np.complex128)

    for block1 in range(N_blocks):
        pgauss1 = primgauss_arr[block1] #(pgauss,3) boolean
        N_pg1 = np.count_nonzero(pgauss1[:,0])

        AO1 = AO_arr[block1] #(N_AO) boolean
        ints_i_x = ints_x[:,pgauss1[:,0],:]
        ints_i_y = ints_y[:,pgauss1[:,1],:]
        ints_i_z = ints_z[:,pgauss1[:,2],:]

        for block2 in range(N_blocks):
            pgauss2 = primgauss_arr[block2] #(pgauss,3) boolean
            N_pg2 = np.count_nonzero(pgauss2[:,0])
            AO2 = AO_arr[block2] #(N_AO) boolean

            tot = np.sum(ints_i_x[pgauss2[:,0]][:,:,R_id[0]] * ints_i_y[pgauss2[:,1]][:,:,R_id[1]] * ints_i_z[pgauss2[:,2]][:,:,R_id[2]], axis=2)
            #tot_s = np.sum(tot, axis=2)

            for i in np.nonzero(AO1)[0]: #AO for block1
                for j in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot@coeff1)@coeff2
    return ovlp

if __name__ == '__main__':
    primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id = load()
    
    cell, dark_objects = initialize_cell()
    blocks = dark_objects['blocks']
    start_time = time.time()
    ovlp = ovlp_original1(blocks, 78, np.array([ints_x, ints_y, ints_z]), R_id)
    print(f'original1: {time.time() - start_time}')

    start_time = time.time()
    ovlp = ovlp_original2(blocks, 78, np.array([ints_x, ints_y, ints_z]), R_id)
    print(f'original2: {time.time() - start_time}')

    R_id = np.copy(np.transpose(R_id, (1,0)), order='C')
    start_time = time.time()
    ovlp = ovlp_numba(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id)
    print(f'numba 1: {time.time() - start_time}')

    start_time = time.time()
    ovlp = ovlp_numba(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id)
    print(f'numba 2: {time.time() - start_time}')

    """
    lp = LineProfiler()
    lp_wrapper = lp(ovlp_original2)
    lp_wrapper(blocks, 78, np.array([ints_x, ints_y, ints_z]), R_id)
    lp.print_stats()
    """

    exit()
    start_time = time.time()
    ovlp = ovlp_cython_np(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id)
    print(f'numpy: {time.time() - start_time}')
    

    primgauss_arr = primgauss_arr.astype(np.int16) #convert to short int
    primgauss_arr_mv, AO_arr_mv, coeff_arr_mv, ints_x_mv, ints_y_mv, ints_z_mv, R_id_mv = convert_mv(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id)
    start_time = time.time()
    ovlp = ovlp_cython_mv(primgauss_arr_mv, AO_arr_mv, coeff_arr_mv, ints_x_mv, ints_y_mv, ints_z_mv, R_id_mv)
    print(f'memoryview: {time.time() - start_time}')

