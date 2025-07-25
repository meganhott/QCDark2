import numpy as np
import numba as nb

import dielectric_pyscf.input_parameters as parmt

#Constants
alpha = 1/137
me = 0.51099895000e6 #eV

def get_eps_im_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path, im_delE, working_dir):
    """
    For non-LFE code
    """
    N_E = int(parmt.E_max/parmt.dE + 1)

    #Calculating 3D overlaps
    eta_qG = get_3D_overlaps_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path) #(a,b,G)
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,:] #(a,b,G)

    eps_im = np.zeros((N_E, qG.shape[0]), dtype='float')

    a_ind, b_ind = np.nonzero(im_delE[i_k,0] < N_E)
    eta_qG_ab = eta_qG[a_ind, b_ind] #(a,b,G) -> (ab,G) #only keeps a,b pairs relevent to delta calculation
    eta_qG_sq = (eta_qG_ab*eta_qG_ab.conj()).real #(ab,G)
    im_delE_ab = im_delE[i_k, :, a_ind, b_ind] #(2, ab)

    for i in range(eta_qG_sq.shape[0]):
        ind, rem = im_delE_ab[i]
        eps_im[int(ind)] += rem*eta_qG_sq[i] #already checked ind < nE
        if ind < N_E - 1:
            eps_im[int(ind+1)] += (1. - rem)*eta_qG_sq[i]
    np.save(working_dir + f'/eps_im/eps_im_k{i_k}.npy', np.transpose(eps_im, (1,0)).copy())
    #return eps_im


def get_3D_overlaps_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path, working_dir=None):
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Possible improvements:
    - change how ints are stored in the first place so we don't have to transpose them in this function

    Inputs:
        i_k (int): k index
        k_f ((3,) np.ndarray): 
            a single k' corresponding to q
        primgauss_arr ((N_blocks, N_primgauss, 3) boolean np.ndarray):
            Primitive gaussians are included in each block
        AO_arr ((N_blocks, N_AO) boolean np.ndarray):
            Atomic orbitals are included in each block
        coeff_arr ((N_AO, N_max_primgauss) complex np.ndarray):
            Primgauss coefficients for each AO
        mo_coeff_i ((N_AO, N_valbands) np.ndarray)
        mo_coeff_f_conj ((N_AO, N_conbands) np.ndarray)
        unique_Ri (list): list of [R_unique_x, R_unique_y, R_unique_z]
        q_cuts ((N_q,) np.ndarray): 
            |q+G| at which R_cutoff changes
    Outputs:
        eta_qG ((N_val_bands (i), N_cond_bands (j)) np.ndarray): 
            all 3D overlaps <jk'|exp(i(q+G)r)|ik>
    """
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[1], mo_coeff_f_conj.shape[1]), dtype='complex128')
    for i, qG_i in enumerate(qG):
        R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG_i)) - 1))
        ints = []
        for d in range(3): 
            ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG_i[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None])
            ints[-1] = np.copy(np.transpose(ints[-1], (1,2,0)), order='C')
        ovlp = ovlp_sum(primgauss_arr, AO_arr, coeff_arr, ints[0], ints[1], ints[2], R_id) #numba optimized function
        eta_qG[i] = np.einsum('kbj,kba,kai->kij', mo_coeff_f_conj[None,:,:], ovlp[None,:,:], mo_coeff_i[None,:,:], optimize = path)[0]
        #eta_qG[i] = get_3D_overlaps_k_arrays2(qG_i, k_f[0], primgauss_arr, AO_arr, coeff_arr, mo_coeff_i[0], mo_coeff_f_conj[0], unique_Ri, q_cuts, einsum_path)
    eta_qG = np.transpose(eta_qG, axes=(1,2,0)) #(G,i,j) -> (i,j,G)
    if working_dir is not None:
        np.save(working_dir + f'/eta_qG/eta_qG_k{i_k}.npy', eta_qG)
    return eta_qG

@nb.njit(nb.complex128[:,::1]( nb.boolean[:,:,::1], nb.boolean[:,::1], nb.complex128[:,::1], nb.complex128[:,:,::1], nb.complex128[:,:,::1], nb.complex128[:,:,::1], nb.int64[:,::1] ))
def ovlp_sum(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id):

    N_blocks = primgauss_arr.shape[0] #68
    N_AO = coeff_arr.shape[0] #78

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

            tot = np.sum(ints_i_x[pgauss2[:,0]][:,:,R_id[:,0]] * ints_i_y[pgauss2[:,1]][:,:,R_id[:,1]] * ints_i_z[pgauss2[:,2]][:,:,R_id[:,2]], axis=2)

            for i in np.nonzero(AO1)[0]: #AO for block1
                for j in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[i,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[j,:N_pg2] #(pgauss2)
                    ovlp[i,j] = (tot@coeff1)@coeff2
    return ovlp

#General outer product for eta_qG_sq (kij, G, G') = eta_qG(kij, G)*eta_qG(kij, G').conj() to replace much slower np.einsum('ag,ah -> agh')
@nb.njit(nb.complex128[:,:,::1](nb.complex128[:,::1], nb.complex128[:,::1], nb.complex128[:,:,::1], nb.float64), parallel=True)
def gen_outer(A, B, C, prefactor):
    for i in nb.prange(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[1]):
                C[i,j,k] = A[i,j]*B[i,k]
    return prefactor*C