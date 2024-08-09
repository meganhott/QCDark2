import numpy as np
import input_parameters as parmt

def get_3D_overlaps_blocks(qG, k2: np.ndarray, blocks: dict, R_id: np.ndarray, unique_Ri: list[np.ndarray], n: int, mo_coeff_i: np.ndarray, mo_coeff_f_conj: np.ndarray) -> np.ndarray:
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

def epsilon_no_LFE_G(qG0: float, qG1: float, qG2: float, k2: np.ndarray, blocks: dict, R_id: np.ndarray, unique_Ri: list[np.ndarray], n: int, k_pairs: np.ndarray, bands: list[int], VCell: float) -> np.ndarray:
    mo_coeff_i = np.load(parmt.store + '/DFT/mo_coeff_i.npy')[k_pairs[:,0]][:,:,bands[0]:bands[1]+1]
    mo_coeff_f_conj = np.load(parmt.store + '/DFT/mo_coeff_i.npy')[k_pairs[:,0]][:,:,bands[2]:bands[3]+1].conj()
    qG = np.array([qG0, qG1, qG2])
    coeff = 4*np.pi*alpha/np.sum(qG**2)
    trans_prob = np.abs(get_3D_overlaps_blocks(qG, k2, blocks, R_id, unique_Ri, n, mo_coeff_i, mo_coeff_f_conj))**2
    mo_en_i = np.load(parmt.store + 'DFT/mo_en_i.npy')[k_pairs[:,0]][:,bands[0]:bands[1]+1]
    mo_en_f = np.load(parmt.store + 'DFT/mo_en_i.npy')[k_pairs[:,0]][:,bands[2]:bands[3]+1]
    omega = np.arange(0, parmt.E_max+0.5*parmt.dE, parmt.dE)
    en_diff = omega[:,None,None,None] - mo_en_f[None,:,None,:] + mo_en_i[None,:,:,None]
    chi0_real = 2./VCell/k_pairs.shape[0]*np.einsum('okij, kij -> o', 1/en_diff, trans_prob, optimize = True)
    eps_real = np.ones(omega.shape) - coeff*chi0_real
    pass

