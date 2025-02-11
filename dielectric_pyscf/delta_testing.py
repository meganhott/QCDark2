import numpy as np
import numba as nb
import time
import dielectric_pyscf.input_parameters as parmt
from dielectric_pyscf.epsilon_routines import delta_energy


def delta_old(i_k, eta_qG_sq, im_delE, eps_delta):
    nE = 501
    for a in range(14):
        for b in range(64):
            ind, rem = im_delE[i_k, a, b]
            if ind < nE:
                eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b]
                if ind < nE - 1:
                    eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
    return eps_delta


def delta_nonzero(i_k, eta_qG_sq, im_delE, eps_delta):
    nE = 501
    a_ind, b_ind = np.nonzero(im_delE[i_k,0] < nE)
    print(a_ind.shape)
    exit()
    for i in range(a_ind.shape[0]):
        a = a_ind[i]
        b = b_ind[i]
        ind, rem = im_delE[i_k, :, a, b]

        eps_delta[:,:,int(ind)] += rem*eta_qG_sq[a, b] #already checked ind < nE
        if ind < nE - 1:
            eps_delta[:,:,int(ind+1)] += (1. - rem)*eta_qG_sq[a, b]
    return eps_delta

#@nb.njit #faster without numba
def delta_ab(eta_qG_sq, im_delE_ab, eps_delta):
    N_E = 501
    for i in range(eta_qG_sq.shape[0]):
            ind, rem = im_delE_ab[i]
            eps_delta[int(ind)] += rem*eta_qG_sq[i] #already checked ind < nE
            if ind < N_E - 1:
                eps_delta[int(ind+1)] += (1. - rem)*eta_qG_sq[i]
    return(eps_delta)
    

def main():
    dft_path = parmt.store + '/DFT/'
    ivalbot, ivaltop, iconbot, icontop = np.load(parmt.store + '/bands.npy')
    mo_en_i = np.load(dft_path + 'mo_en_i.npy')[:,ivalbot:ivaltop+1]
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')[:,iconbot:icontop+1]

    im_delE_old = delta_energy(mo_en_i, mo_en_f) #(k,a,b,2)
    im_delE = np.copy(np.transpose(im_delE_old, (0,3,1,2))) #(k,2,a,b)

    N_G = 500
    N_E = 501
    eta_qG = np.ones((14, 64, N_G), dtype='complex128')
    #eta_qG_sq = np.ones((14, 64, N_G, N_G), dtype='complex128')
    eps_delta_old = np.zeros((N_G, N_G, 501), dtype='complex')
    eps_delta = np.zeros((501, N_G, N_G), dtype='complex')

    for i in range(3):
        i_k = 0
        start_time = time.time()
        eta_qG_sq = np.einsum('ijg, ijh -> ijgh', eta_qG, eta_qG.conj())
        print(f'old eta_qG_sq initialization: {time.time() - start_time}, size = {eta_qG_sq.nbytes/2**30} GB')

        start_time = time.time()
        eps_delta0 = delta_old(i_k, eta_qG_sq, im_delE_old, eps_delta_old)
        print(f'old delta calculation: {time.time() - start_time}')

        start_time = time.time()
        a_ind, b_ind = np.nonzero(im_delE[i_k,0] < N_E)
        eta_qG_ab = eta_qG[a_ind, b_ind] #(a,b,G) -> (ab,G) #only keeps a,b pairs relevent to delta calculation
        im_delE_ab = im_delE[i_k, :, a_ind, b_ind] #(2, ab)
        eta_qG_sq = np.einsum('ag, ah -> agh', eta_qG_ab, eta_qG_ab.conj()) #(ab,G,G')
        print(f'new eta_qG_sq initialization: {time.time() - start_time}, size = {eta_qG_sq.nbytes/2**30} GB')

        start_time = time.time()
        eps_delta1 = delta_ab(eta_qG_sq, im_delE_ab, eps_delta)
        print(f'new delta calculation: {time.time() - start_time}', '\n')

        #print((np.transpose(eps_delta0, (2,0,1)) == eps_delta1).all())

    

if __name__ == '__main__':
    main()

