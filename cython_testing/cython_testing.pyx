import numpy as np
cimport numpy as cnp
#from libcpp cimport bool as bool_t
cnp.import_array()

cimport cython

@cython.boundscheck(False) #does not check if indices are within bounds
@cython.wraparound(False) #does not check for negative indexing
#using memoryviews instead of numpy arrays
cpdef double complex[:,::1] ovlp_cython_mv(   short[:,:,::1] primgauss_arr, 
                                            cnp.uint8_t[:,::1] AO_arr, 
                                            double complex[:,::1] coeff_arr, 
                                            double complex[:,:,::1] ints_x, 
                                            double complex[:,:,::1]  ints_y, 
                                            double complex[:,:,::1]  ints_z, 
                                            long[:,::1] R_id): #nogil?

    cdef size_t N_blocks = primgauss_arr.shape[0] #68
    cdef size_t N_primgauss = primgauss_arr.shape[1] #86
    cdef size_t N_AO = coeff_arr.shape[0] #78
    cdef size_t N_maxcoeff = coeff_arr.shape[1] #13
    cdef size_t N_R = R_id.shape[0]

    cdef double complex [:,::1] ovlp = np.empty((N_AO, N_AO), dtype=np.complex128, order='C')

    #cdef bint [:,:] pgauss1, pgauss2
    #cdef bint [:] AO1, AO2
    cdef double complex [:,::1] tot
    cdef double complex [::1] coeff1, coeff2, ints_x_ij, ints_y_ij, ints_z_ij

    cdef double complex a

    cdef long [::1] pg1_x, pg1_y, pg1_z, pg2_x, pg2_y, pg2_z

    cdef size_t block1, block2, N_pg1, N_pg2, i, j, n, pg1, pg2, r #indices of arrays should be int (size_t) type

    cdef long[::1] R_id_r = np.zeros(3, dtype=np.int64, order='C')

    for block1 in range(N_blocks):
        #pgauss1 = primgauss_arr[block1] #(pgauss,3) boolean
        #N_pg1 = np.count_nonzero(pgauss1[:,0]) #number of primitive gaussians in block1
        #N_pg1 = 0
        #for n in range(N_primgauss):
        #    N_pg1 += primgauss_arr[block1, n, 0]

        
        #N_pg1 = sum(pgauss1[:,0]) #number of primitive gaussians in block1

        #rewriting np.nonzero
        #determine number of primgauss in block
        """
        N_pg1 = 0
        for p in range(N_primgauss):
            N_pg1 += primgauss_arr[block1, p, 0]
        print(N_pg1)
        """
        #make three arrays of length N_pg1

        #

        pg1_x = np.nonzero(primgauss_arr[block1, :, 0])[0]
        pg1_y = np.nonzero(primgauss_arr[block1, :, 1])[0]
        pg1_z = np.nonzero(primgauss_arr[block1, :, 2])[0]
        N_pg1 = pg1_x.shape[0]

        #ints_i_x = np.empty((N_R, ), dtype=np.complex128)

        #AO1 = AO_arr[block1] #(N_AO) boolean
        #ints_i_x = ints_x[:,:,pgauss1[:,0]]
        #ints_i_y = ints_y[:,:,pgauss1[:,1]]
        #ints_i_z = ints_z[:,:,pgauss1[:,2]]

        for block2 in range(N_blocks):
            pg2_x = np.nonzero(primgauss_arr[block1, :, 0])[0]
            pg2_y = np.nonzero(primgauss_arr[block1, :, 1])[0]
            pg2_z = np.nonzero(primgauss_arr[block1, :, 2])[0]
            N_pg2 = pg2_x.shape[0]

            #pgauss2 = primgauss_arr[block2] #(pgauss,3) boolean
            #N_pg2 = np.count_nonzero(pgauss2[:,0]) #number of primitive gaussians in block2
            #AO2 = AO_arr[block2] #(N_AO) boolean

            tot = np.empty((N_pg2, N_pg1), dtype=np.complex128, order='C') #make matrix memoryview so numpy is not called? https://stackoverflow.com/questions/21005656/cython-create-memoryview-without-numpy-array
            
            with nogil:
                for pg2 in range(N_pg2):
                    for pg1 in range(N_pg1):
                        a = 0
                        ints_x_ij = ints_x[pg2_x[pg2], pg1_x[pg1]]
                        ints_y_ij = ints_x[pg2_x[pg2], pg1_x[pg1]]
                        ints_z_ij = ints_x[pg2_x[pg2], pg1_x[pg1]]
                        for r in range(N_R):
                            a += ints_x_ij[R_id[r,0]]*ints_y_ij[R_id[r,1]]*ints_z_ij[R_id[r,2]]
                            #a += ints_x[pg2_x[pg2], pg1_x[pg1], R_id[r, 0]]*ints_y[pg2_y[pg2], pg1_y[pg1], R_id[r, 1]]*ints_z[pg2_z[pg2], pg1_z[pg1], R_id[r, 2]]
                            #tot[r,pg2,pg1] = ints_x[R_id[0, r], pg2_x[pg2], pg1_x[pg1]] * ints_y[R_id[1, r], pg2_y[pg2], pg1_y[pg1]] * ints_z[R_id[2, r], pg2_z[pg2], pg1_z[pg1]]
                        #tot[pg2, pg1] = 1 #this is fast but not when setting equal to a. Is tot still numpy array somehow? implement memoryview creation above?
                        tot[pg2, pg1] = a
            

            """
            tot = np.ones((R_id.shape[1], N_pg2, N_pg1), dtype=np.complex128)
            for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
                ints_ij = ints_i[:,pgauss2[:,d]]
                tot *= ints_ij[R_id[d]]
            
            tot_s = np.sum(tot, axis=0)
            

            for i in np.nonzero(AO_arr[block1])[0]: #AO for block1
                for j in np.nonzero(AO_arr[block2])[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot@coeff1)@coeff2
            """

    return tot #ovlp

cpdef get_pg(short[:] arr, size_t N_primgauss, size_t N_pg):
    cdef int[:] pg = np.empty(N_pg, dtype=np.int16)
    cdef size_t i, pg_i
    pg_i = 0 #nonzero primgauss index 
    for i in range(N_primgauss):
        if arr[i] == 1:
            pg[pg_i] = i #write to next nonzero primgauss
            pg_i += 1 #increment nonzero primgauss index
    


#cnp.ndarray[cnp.npy_bool, ndim=3] primgauss_arr
def convert_mv(cnp.ndarray[cnp.int16_t, ndim=3] primgauss_arr, 
                cnp.ndarray[cnp.npy_bool, ndim=2] AO_arr, 
                cnp.ndarray[cnp.complex128_t, ndim=2] coeff_arr, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_x, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_y, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_z, 
                cnp.ndarray[cnp.int64_t, ndim=2] R_id):

    cdef short [:,:,:] primgauss_arr_mv = primgauss_arr
    cdef cnp.uint8_t [:,:] AO_arr_mv = AO_arr
    cdef double complex [:,:] coeff_arr_mv = coeff_arr
    cdef double complex [:,:,:] ints_x_mv = ints_x
    cdef double complex [:,:,:] ints_y_mv = ints_y
    cdef double complex [:,:,:] ints_z_mv = ints_z
    cdef long [:,:] R_id_mv = R_id

    return primgauss_arr_mv, AO_arr_mv, coeff_arr_mv, ints_x_mv, ints_y_mv, ints_z_mv, R_id_mv



def ovlp_cython_np(cnp.ndarray[cnp.npy_bool, ndim=3] primgauss_arr, 
                cnp.ndarray[cnp.npy_bool, ndim=2] AO_arr, 
                cnp.ndarray[cnp.complex128_t, ndim=2] coeff_arr, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_x, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_y, 
                cnp.ndarray[cnp.complex128_t, ndim=3] ints_z, 
                cnp.ndarray[cnp.int64_t, ndim=2] R_id):
    #assert a.type == np.bool and c.type == np.complex128 and p.type == np.bool

    cdef int N_blocks = primgauss_arr.shape[0] #68
    cdef int N_primgauss = primgauss_arr.shape[1] #86
    cdef int N_AO = coeff_arr.shape[0] #78
    cdef int N_maxcoeff = coeff_arr.shape[1] #13

    cdef cnp.ndarray[cnp.complex128_t, ndim=2] ovlp = np.empty((N_AO, N_AO), dtype=np.complex128)

    cdef cnp.ndarray[cnp.npy_bool, ndim=2] pgauss1, pgauss2
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] AO1, AO2
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] ints_i_x, ints_i_y, ints_i_z
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] tot
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] tot_s
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] coeff1, coeff2

    cdef int block1, block2, N_pg1, N_pg2, i, j #indices of arrays should be int type

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

            tot = ints_i_x[pgauss2[:,0]][:,:,R_id[:,0]] * ints_i_y[pgauss2[:,1]][:,:,R_id[:,1]] * ints_i_z[pgauss2[:,2]][:,:,R_id[:,2]]
            """
            tot = np.ones((R_id.shape[1], N_pg2, N_pg1), dtype=np.complex128)
            for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
                ints_ij = ints_i[:,pgauss2[:,d]]
                tot *= ints_ij[R_id[d]]
            """
            tot_s = np.sum(tot, axis=2)

            for i in np.nonzero(AO1)[0]: #AO for block1
                for j in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot_s@coeff1)@coeff2
    return ovlp


    """
    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 

    for block1 in range(N_blocks):
        pgauss1 = primgauss_arr[block1] #(pgauss,3) boolean
        N_pg1 = np.count_nonzero(pgauss1[:,0])

        AO1 = AO_arr[block1] #(N_AO) boolean
        ints_i_x = ints_x[:,:,pgauss1[:,0]]
        ints_i_y = ints_y[:,:,pgauss1[:,1]]
        ints_i_z = ints_z[:,:,pgauss1[:,2]]

        for block2 in range(N_blocks):
            pgauss2 = primgauss_arr[block2] #(pgauss,3) boolean
            N_pg2 = np.count_nonzero(pgauss2[:,0])
            AO2 = AO_arr[block2] #(N_AO) boolean

            tot = np.ones((R_id.shape[1], N_pg2, N_pg1), dtype='complex')
            for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
                ints_ij = ints_i[:,pgauss2[:,d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)

            for i in np.nonzero(AO1)[0]: #AO for block1
                for j in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot@coeff1)@coeff2

    return ovlp
    """




'''
def get_3D_overlaps_k_arrays(qG: np.ndarray, k_f: np.ndarray, primgauss_arr, AO_arr, coeff_arr, N_AO: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray, unique_Ri:list[np.ndarray], q_cuts: np.ndarray, path) -> np.ndarray:
    R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG)) - 1))
    ints = []
    for d in range(3): 
        ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
    ovlp = ovlp_sum_full_arrays(primgauss_arr, AO_arr, coeff_arr, ints[0], ints[1], ints[2], R_id) #numba optimized function

    return np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0] #This is faster with added k index for some reason??

@njit(nb.types.Array(nb.types.complex128, 2, 'C')(nb.types.Array(nb.types.boolean, 3, 'C'), nb.types.Array(nb.types.boolean, 2, 'C'), nb.types.Array(nb.types.complex128, 2, 'C'), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.complex128, 3, 'C'), nb.types.Array(nb.types.int64, 2, 'F')))
def ovlp_sum_full_arrays(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id):

    N_blocks = 68
    N_primgauss = 85
    N_AO = 78
    N_maxcoeff = 13

    ovlp = np.empty((N_AO, N_AO), dtype = np.complex128) 

    for block1 in range(N_blocks):
        pgauss1 = primgauss_arr[block1] #(pgauss,3) boolean
        N_pg1 = np.count_nonzero(pgauss1[:,0])

        AO1 = AO_arr[block1] #(N_AO) boolean
        ints_i_x = ints_x[:,:,pgauss1[:,0]]
        ints_i_y = ints_y[:,:,pgauss1[:,1]]
        ints_i_z = ints_z[:,:,pgauss1[:,2]]

        for block2 in range(N_blocks):
            pgauss2 = primgauss_arr[block2] #(pgauss,3) boolean
            N_pg2 = np.count_nonzero(pgauss2[:,0])
            AO2 = AO_arr[block2] #(N_AO) boolean

            tot = np.ones((R_id.shape[1], N_pg2, N_pg1), dtype='complex')
            for d, ints_i in enumerate([ints_i_x, ints_i_y, ints_i_z]):
                ints_ij = ints_i[:,pgauss2[:,d]]
                tot *= ints_ij[R_id[d]]
            tot = tot.sum(axis = 0)

            for i in np.nonzero(AO1)[0]: #AO for block1
                for j in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot@coeff1)@coeff2

    return ovlp
'''