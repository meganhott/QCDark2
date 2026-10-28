import numpy as np
import numba as nb

import qcdark2.dielectric_pyscf.input_parameters as parmt

#Constants
alpha = 1/137
me = 0.51099895000e6 #eV

def delta_G(im_delE_k, ovlp1, ovlp2, N_E):
    """
    Returns spectral part of dielectric function (up to a prefactor) for head, wings, and noLFE body for one k

    Inputs:
        im_delE_k (2,i,j) delta function information
        ovlp1, ovlp2 (i,j,G) overlap factors
        N_E: int: number of energies
    Outputs:
        eps_delta (E,G)

    """
    N_G = ovlp1.shape[2]
    eps_delta = np.zeros((N_E, N_G), dtype='complex')

    i_ind, j_ind = np.nonzero(im_delE_k[0] < N_E)
    im_delE_ij = im_delE_k[:, i_ind, j_ind] #(2, ij)

    ovlp1_ij = ovlp1[i_ind, j_ind] #(i,j,G) -> (ij,G) #only keeps i,j pairs relevent to delta calculation
    ovlp2_ij = ovlp2[i_ind, j_ind]
    ovlp = ovlp1_ij * ovlp2_ij #(ij,G)

    for i in range(ovlp.shape[0]):
        ind, rem = im_delE_ij[:,i]
        eps_delta[int(ind)] += rem * ovlp[i] #already checked ind < nE
        if ind < N_E - 1:
            eps_delta[int(ind+1)] += (1. - rem) * ovlp[i]

    return eps_delta #(E,G)

def delta_GG(im_delE_ind, im_delE_rem, eta_qG, E_i, E_f, N_G, prefactor, eps_delta_n_min_1):
    """
    Returns spectral part of dielectric function for LFE body for one E_i:E_f chunk

    Inputs:
        im_delE_ind (k,i,j) delta function energy indices
        im_delE_rem (k,i,j) delta function remainders
        eta_qG (k,i,j,G) overlap factors
        E_i: int: initial energy index of chunk
        E_f: int: final energy index of chunk
        N_G: int: number of G-vectors

    Outputs:
        eps_delta_chunk (E_i:E_f,G)
    """
    eps_delta_chunk = np.empty((E_f-E_i, N_G, N_G), dtype='complex')

    for i in range(E_f - E_i):
        n_E = E_i + i
        k_ind, i_ind, j_ind = np.where(im_delE_ind == n_E)
        eta_qG_kij = eta_qG[k_ind, i_ind, j_ind] #(k,i,j,G) -> (kij,G) #only keeps k,i,j elements relevent to delta calculation
        rem_kij = im_delE_rem[k_ind, i_ind, j_ind]

        eta_qG_kij_sq = np.empty((k_ind.shape[0], eta_qG_kij.shape[1], eta_qG_kij.shape[1]), dtype='complex')
        eta_qG_kij_sq = gen_outer(eta_qG_kij.conj(), eta_qG_kij, eta_qG_kij_sq, prefactor) #numba optimized function

        eps_delta_n = np.tensordot(rem_kij, eta_qG_kij_sq, axes=(0,0)) 
        eps_delta_chunk[i] = eps_delta_n + eps_delta_n_min_1

        eps_delta_n_min_1 = np.tensordot(1 - rem_kij, eta_qG_kij_sq, axes=(0,0)) # (1 - rem)*eta_sq

    return eps_delta_chunk, eps_delta_n_min_1

def get_eps_im_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path, im_delE, working_dir):
    """
    Returns imaginary part of dielectric function (up to a prefactor) for body of non-LFE
    """
    N_E = int(parmt.E_max/parmt.dE + 1)

    #Calculating 3D overlaps
    eta_qG = get_3D_overlaps_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path) #(i,j,G)
    eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,:] #(i,j,G)

    eps_im = np.real(delta_G(im_delE[i_k], eta_qG, eta_qG.conj(), N_E)) #(E,G)

    np.save(working_dir + f'/eps_im/eps_im_k{i_k}.npy', np.transpose(eps_im, (1,0)).copy()) #(G,E)
    #return eps_im

def get_eps_im_k_head(i_k, ovlp, im_delE):
    """
    Returns imaginary part of head (G = G' = 0, q -> 0 limit) of dielectric function (up to a prefactor) 

    Inputs:
        i_k int
        ovlp (3,i,j)
        im_delE (k,2,i,j)
        q_dir (bins,3)
    Output:
        eps_im_t (3,3,E): optical limit tensor
    """
    N_E = int(parmt.E_max/parmt.dE + 1)

    ovlp_t = np.einsum('xij, yij -> ijxy', ovlp, ovlp.conj()).reshape((ovlp.shape[1], ovlp.shape[2], 9)) #(i,j,3,3) -> (i,j,9)

    # treat each xy component as G vector in delta_G function
    eps_im_t = np.real(delta_G(im_delE[i_k], ovlp_t, np.ones_like(ovlp_t), N_E)).reshape((N_E,3,3))

    return eps_im_t #(E,3,3)

def get_eps_delta_k_wings(i_k, k_f, ovlp, mo_coeff_i, mo_coeff_f_conj, G, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path, im_delE):
    """
    Returns spectral part of wings (G = 0, G' != 0, q -> 0 limit) of dielectric function (up to a prefactor) 

    Inputs:
        i_k int
        ovlp (3,i,j)
        im_delE (k,2,i,j)
    """
    N_E = int(parmt.E_max/parmt.dE + 1)

    #Calculating 3D overlaps
    eta_qG = get_3D_overlaps_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, G, primgauss_arr, AO_arr, coeff_arr, unique_Ri, q_cuts, path) #(i,j,G)
    eta_qG = eta_qG / np.linalg.norm(G, axis=1)[None,None,:] #(i,j,G)

    eps_delta = np.zeros((3, N_E, G.shape[0]), dtype='complex')
    for i in range(3): # for each direction of nabla overlaps
        eps_delta[i] = delta_G(im_delE[i_k], np.repeat(ovlp[i,:,:,None], eta_qG.shape[2], axis=2), eta_qG, N_E) #(E,G)

    return eps_delta #(3,E,G)

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
            ints[-1] = np.copy(np.transpose(ints[-1], (1,2,0)), order='C') #(a,b,R)
        ovlp = ovlp_sum(primgauss_arr, AO_arr, coeff_arr, ints[0], ints[1], ints[2], R_id) #numba optimized function #(b,a)
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

    for block1 in range(N_blocks): #b
        pgauss1 = primgauss_arr[block1] #(pgauss,3) boolean
        N_pg1 = np.count_nonzero(pgauss1[:,0])

        AO1 = AO_arr[block1] #(N_AO) boolean
        ints_i_x = ints_x[:,pgauss1[:,0],:]
        ints_i_y = ints_y[:,pgauss1[:,1],:]
        ints_i_z = ints_z[:,pgauss1[:,2],:]

        for block2 in range(N_blocks): #a
            pgauss2 = primgauss_arr[block2] #(pgauss,3) boolean
            N_pg2 = np.count_nonzero(pgauss2[:,0])
            AO2 = AO_arr[block2] #(N_AO) boolean

            tot = np.sum(ints_i_x[pgauss2[:,0]][:,:,R_id[:,0]] * ints_i_y[pgauss2[:,1]][:,:,R_id[:,1]] * ints_i_z[pgauss2[:,2]][:,:,R_id[:,2]], axis=2)

            for b in np.nonzero(AO1)[0]: #AO for block1
                for a in np.nonzero(AO2)[0]: #AO for block2
                    coeff1 = coeff_arr[b,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[a,:N_pg2] #(pgauss2)
                    ovlp[b,a] = (tot@coeff1)@coeff2
    return ovlp

#General outer product for eta_qG_sq (kij, G, G') = eta_qG(kij, G)*eta_qG(kij, G').conj() to replace much slower np.einsum('ag,ah -> agh')
@nb.njit(nb.complex128[:,:,::1](nb.complex128[:,::1], nb.complex128[:,::1], nb.complex128[:,:,::1], nb.float64), parallel=True)
def gen_outer(A, B, C, prefactor):
    for i in nb.prange(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[1]):
                C[i,j,k] = A[i,j]*B[i,k]
    return prefactor*C

def block_inversion_diag(A, B, C, D_inv):
    """
    Computes the diagonal components of the [[A, B],[C, D]] matrix using blockwise inversion.

    Inputs:
        A: (N_E,) (head)
        B: (N_E, N_G) (wing)
        C: (N_E, N_G) (wing)
        D: (N_E, N_G, N_G) (body)
    Outputs:
        M_inv_head_diag: (N_E,)
        M_inv_body_diag: (N_E, N_G)
    """
    MD_schur_inv = 1 / (A - np.einsum('eg,egh,eh -> e', B, D_inv, C))

    N_E = A.shape[0]
    N_G = B.shape[1] + 1

    G = np.einsum('eh,e,eg -> ehg', C, MD_schur_inv, B)
    M_inv_diag = 1/np.diagonal(D_inv + (D_inv @ G @ D_inv), axis1=1, axis2=2)
    return 1/MD_schur_inv, M_inv_diag


## ----- Current-current overlaps ----- ##

def get_Pi_vv_im_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, primgauss_vv_coeff, unique_Ri, q_cuts, path, im_delE, working_dir):
    """
    Returns imaginary part of current-current self-energy Pi_{v^i v_j} (up to a prefactor) for body of noLFE
    """
    N_E = int(parmt.E_max/parmt.dE + 1)

    #Calculating 3D overlaps
    eta_qG_v = get_3D_overlaps_vv_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, primgauss_vv_coeff, unique_Ri, q_cuts, path) #(4,i,j,G)
    # Do I want to still divide by |q+G|?
    #eta_qG = eta_qG / np.linalg.norm(qG, axis=1)[None,None,:] #(i,j,G)

    # Pi_vv[0] = Pi_11 (normal overlaps)
    # Pi_vv[1-7] = Pi_vx,vx, Pi_vy,vy, Pi_vz,vz, Pi_vx,vy, Pi_vx,vz, Pi_vy,vz 
    # (Pi_vi,vj = Pi_vj,vi*) 
    Pi_vv = np.zeros((7, N_E, eta_qG_v.shape[3]), dtype='complex')
    Pi_vv[0] = delta_G(im_delE[i_k], eta_qG_v[0].conj(), eta_qG_v[0], N_E) #(E,G), no derivative (Pi_11)
    for i in range(1, 4):
        Pi_vv[i] = delta_G(im_delE[i_k], eta_qG_v[i].conj(), eta_qG_v[i], N_E) # Pi_vi,vi
    Pi_vv[4] = delta_G(im_delE[i_k], eta_qG_v[1].conj(), eta_qG_v[2], N_E) # Pi_vx,vy
    Pi_vv[5] = delta_G(im_delE[i_k], eta_qG_v[1].conj(), eta_qG_v[3], N_E) # Pi_vx,vz
    Pi_vv[6] = delta_G(im_delE[i_k], eta_qG_v[2].conj(), eta_qG_v[3], N_E) # Pi_vy,vz

    np.save(working_dir + f'/eps_im/Pi_vv_k{i_k}.npy', np.transpose(Pi_vv, (0,2,1)).copy()) #(7,G,E)
    #return eps_im

def get_3D_overlaps_vv_k(i_k, k_f, mo_coeff_i, mo_coeff_f_conj, qG, primgauss_arr, AO_arr, coeff_arr, primgauss_vv_coeff, primgauss_vv_gaussians, unique_Ri, q_cuts, path, working_dir=None):
    """
    Calculates all 3D overlaps eta = <jk'|exp(i(q+G)r)|ik> and eta_v_a = <jk'|exp(i(q+G)r) del_a |ik> for cartesian directions 'a', for a given 1BZ q-vector and given G-vector using stored 1D overlaps

    Inputs:
        i_k (int): k index
        k_f ((3,) np.ndarray): 
            a single k' corresponding to q
        primgauss_arr (3(l), N_blocks, N_primgauss, 3(d)) boolean np.ndarray:
            Primitive gaussians are included in each block
        AO_arr ((N_blocks, N_AO) boolean np.ndarray):
            Atomic orbitals are included in each block
        coeff_arr ((N_AO, N_max_primgauss) complex np.ndarray):
            Primgauss coefficients for each AO
        primgauss_vv_coeff (3, N_primgauss)
        primgauss_vv_gaussians (3, N_primgauss_full)
        mo_coeff_i ((N_AO, N_valbands) np.ndarray)
        mo_coeff_f_conj ((N_AO, N_conbands) np.ndarray)
        unique_Ri (list): list of [R_unique_x, R_unique_y, R_unique_z]
        q_cuts ((N_q,) np.ndarray): 
            |q+G| at which R_cutoff changes
    Outputs:
        eta_qG ((N_val_bands (i), N_cond_bands (j)) np.ndarray): 
            all 3D overlaps <jk'|exp(i(q+G)r)|ik>
        eta_qG_v (3, N_val_bands (i), N_cond_bands (j))
            all 3D overlaps <jk'|exp(i(q+G)r) del_a |ik>
    """
    eta_qG = np.empty((qG.shape[0], mo_coeff_i.shape[1], mo_coeff_f_conj.shape[1]), dtype='complex128') # (G,i,j)
    eta_qG_v = np.empty((qG.shape[0], 4, mo_coeff_i.shape[1], mo_coeff_f_conj.shape[1]), dtype='complex128') # (G,deriv,i,j)
    for i, qG_i in enumerate(qG):
        R_id = np.load(parmt.store + '/R_ids/{}.npy'.format(np.sum(q_cuts < np.linalg.norm(qG_i)) - 1))
        ints = []
        for d in range(3): 
            ints.append(np.load(parmt.store + '/primgauss_1d_integrals/dim_{}/{:.5f}.npy'.format(d, qG_i[d]))[:,:,:] * np.exp(-1.j*unique_Ri[d][:]*k_f[None,d])[:,None,None]) #(R,a,b)
            ints[-1] = np.copy(np.transpose(np.array([ints[-1][:,primgauss_vv_gaussians[i]][:,:,primgauss_vv_gaussians[0]]*primgauss_vv_coeff[i][None,:,None] for i in range(3)]), (0,2,3,1)), order='C') # ints[d] = (3(l),a,b,R)
        ovlp = ovlp_sum_v(primgauss_arr, AO_arr, coeff_arr, ints[0], ints[1], ints[2], R_id, qG_i.astype('complex')) # (4(deriv),b,a) # ints have already been multiplied by primgauss coeff

        eta_qG_v[i] = np.einsum('dbj,dba,dai->dij', mo_coeff_f_conj[None,:,:], ovlp[:,:,:], mo_coeff_i[None,:,:], optimize = path)
    eta_qG_v = np.transpose(eta_qG_v, axes=(0,2,3,1)) #(deriv,G,i,j) -> (deriv,i,j,G)
    if working_dir is not None:
        np.save(working_dir + f'/eta_qG/eta_qG_k{i_k}.npy', eta_qG_v)
    return eta_qG_v # (4(normal+deriv),i,j,G)

@nb.njit(nb.complex128[:,:,::1]( nb.boolean[:,:,:,::1], nb.boolean[:,::1], nb.complex128[:,::1], nb.complex128[:,:,::1], nb.complex128[:,:,:,::1], nb.complex128[:,:,:,::1], nb.int64[:,::1], nb.complex128[::1] ))
def ovlp_sum_v(primgauss_arr, AO_arr, coeff_arr, ints_x, ints_y, ints_z, R_id, qG):

    N_blocks = primgauss_arr.shape[1]
    N_AO = coeff_arr.shape[0]

    ovlp = np.empty((3, N_AO, N_AO), dtype=np.complex128)

    for block1 in range(N_blocks): # block1: prim_b
        pgauss1 = primgauss_arr[:,block1] #(3(l),pgauss,3(d)) boolean
        N_pg1 = np.count_nonzero(pgauss1[:,:,0])

        AO1 = AO_arr[block1] #(N_AO) boolean
        ints_i_x = ints_x[:,:,pgauss1[0,:,0]] #(3(l),prim_a,prim_b,R)
        ints_i_y = ints_y[:,:,pgauss1[0,:,1]] #(3(l),prim_a,prim_b,R)
        ints_i_z = ints_z[:,:,pgauss1[0,:,2]] #(3(l),prim_a,prim_b,R)


        #ints_i_x = np.array([ints_x[i,:,pgauss1[i,:,0],:]] for i in range(3)) #(3(l),prim_b,prim_a,R)
        #ints_i_y = np.array([ints_y[i,:,pgauss1[i,:,1],:]] for i in range(3))
        #ints_i_z = np.array([ints_z[i,:,pgauss1[i,:,2],:]] for i in range(3))

        for block2 in range(N_blocks): # block2: prim_b
            pgauss2 = primgauss_arr[0,block2] #(3(l),pgauss,3(d)) boolean # b always normal Gaussian
            N_pg2 = np.count_nonzero(pgauss2[:,:,0])

            AO2 = AO_arr[block2] #(N_AO) boolean
            ints_i_x2 = np.array([ints_i_x[i,pgauss2[i,:,0]] for i in range(3)])[:,:,:,R_id[:,0]] #(l,a,b,R)
            ints_i_y2 = np.array([ints_i_y[i,pgauss2[i,:,1]] for i in range(3)])[:,:,:,R_id[:,1]]
            ints_i_z2 = np.array([ints_i_z[i,pgauss2[i,:,2]] for i in range(3)])[:,:,:,R_id[:,2]]


            #ints_i_x2 = ints_i_x[:,pgauss2[:,0]][:,:,R_id[:,0]] #(3(l),prim_b,prim_a,R)
            #ints_i_y2 = ints_i_y[:,pgauss2[:,1]][:,:,R_id[:,1]]
            #ints_i_z2 = ints_i_z[:,pgauss2[:,1]][:,:,R_id[:,1]]

            # first axis corresponds to derivative
            tot = np.empty((4, ints_i_x2.shape[1], ints_i_x2.shape[2]), dtype=np.complex128) #first axis: no derivative, d_x, d_y, d_z
            tot[0] = np.sum(ints_i_x2[0] * ints_i_y2[0] * ints_i_z2[0], axis=2) # no derivative, (prim_b,prim_a), summed over R
            tot[1] = -1j/2/me * np.sum((1j*qG[0]*ints_i_x2[0] + 2*ints_i_x2[1] + 2*ints_i_x2[2]) * ints_i_y2[0] * ints_i_z2[0], axis=2) # d_x, (prim_a,prim_b)
            tot[2] = -1j/2/me * np.sum((1j*qG[1]*ints_i_y2[0] + 2*ints_i_y2[1] + 2*ints_i_y2[2]) * ints_i_x2[0] * ints_i_z2[0], axis=2) # d_y, (prim_a,prim_b)
            tot[3] = -1j/2/me * np.sum((1j*qG[2]*ints_i_z2[0] + 2*ints_i_z2[1] + 2*ints_i_z2[2]) * ints_i_x2[0] * ints_i_y2[0], axis=2) # d_z, (prim_a,prim_b)

            for b in np.nonzero(AO1)[0]: #AO for block1 (b)
                for a in np.nonzero(AO2)[0]: #AO for block2 (a)
                    coeff1 = coeff_arr[b,:N_pg1] #(pgauss1)
                    coeff2 = coeff_arr[a,:N_pg2] #(pgauss2)
                    ovlp[:,b,a] = (tot@coeff1)@coeff2 # check (4(deriv),b,a)
                    
    return ovlp