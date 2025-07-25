import numpy as np
import numba as nb

@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64, nb.float64))
def kramerskronig_im2re(im, E_max, dE):
    """
    Computes the Kramers-Kronig transformation of the input imaginary part to obtain the real part. Requires Im[f(-E_max <= E <= E_max)].

    Input:
        im: np.ndarray (N_G, N_E)
    Output:
        re: np.ndarray (N_G, N_E), Note that Re(eps) = eps_re + 1 
    """
    E = np.arange(-E_max, E_max+dE, dE)
    re = np.empty_like(im)
    N_G, N_E = im.shape
    for nE in range(N_E):
        E_pv = np.delete(E, nE) #removes Ei = En for principal value
        En = E[nE]
        for nG in range(N_G):
            im_pv = np.concatenate((im[nG, :nE], im[nG, nE+1:]))

            s = 0
            for ns in range(N_E-1):
                s += im_pv[ns]/(E_pv[ns] - En)

            re[nG,nE] = 1/np.pi*dE*(s - 0.5*(im_pv[0]/(E_pv[0]-En) + im_pv[-1]/(E_pv[-1]-En))) #trapezoid rule
    return re

@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64, nb.float64))
def kramerskronig_re2im(re, E_max, dE):
    """
    Computes the Kramers-Kronig transformation of the input real part to obtain the imaginary part. Requires Re[f(-E_max <= E <= E_max)].

    Input:
        re: np.ndarray (N_G, N_E)
    Output:
        im: np.ndarray (N_G, N_E)
    """
    E = np.arange(-E_max, E_max+dE, dE)
    im = np.empty_like(re)
    N_G, N_E = re.shape
    for nE in range(N_E):
        E_pv = np.delete(E, nE) #removes Ei = En for principal value
        En = E[nE]
        for nG in range(N_G):
            re_pv = np.concatenate((re[nG, :nE], re[nG, nE+1:]))

            s = 0
            for ns in range(N_E-1):
                s += re_pv[ns]/(E_pv[ns] - En)

            im[nG,nE] = -1/np.pi*dE*(s - 0.5*(re_pv[0]/(E_pv[0]-En) + re_pv[-1]/(E_pv[-1]-En))) #trapezoid rule
    return im

@nb.njit(nb.complex128[:,:,::1](nb.complex128[:,:,::1], nb.float64, nb.float64), parallel=True)
def kramerskronig_lfe(eps_delta, E_max, dE):
    """
    Calculates principal value part of Re(eps) and Im(eps) for LFEs.

    Input:
        eps_delta: np.ndarray of shape (N_E, N_G, N_G): Dirac delta part of dielectric function
    Output: 
        epa_lfe: np.ndarray of shape (N_E, N_G, N_G): i*Delta part of dielectric function plus its Fourier transform
    """
    E = np.arange(-E_max, E_max+dE, dE).astype('complex')
    eps_lfe = np.empty_like(eps_delta)
    N_E, N_G, N_Gp = eps_delta.shape

    pi = complex(np.pi)
    dE = complex(dE)

    for nE in nb.prange(N_E):
        E_pv = np.delete(E, nE) #removes Ei = En for principal value
        En = complex(E[nE])
        for nG in range(N_G):
            for nGp in range(N_Gp):
                eps_delta_pv = np.concatenate((eps_delta[:nE, nG, nGp], eps_delta[nE+1:, nG, nGp]))

                s = complex(0)
                for ns in range(N_E - 1):
                    s += eps_delta_pv[ns]/(E_pv[ns] - En)

                eps_lfe[nE,nG,nGp] = 1j*eps_delta[nE,nG,nGp] + 1/pi*dE*(s - 0.5*(eps_delta_pv[0]/(E_pv[0]-En) + eps_delta_pv[-1]/(E_pv[-1]-En)))
    return eps_lfe #still need to add identity to this

@nb.njit()
def kramerskronig_im2re_causal(im, E_max, dE):
    """
    Computes the Kramers-Kronig transformation of the input imaginary part to obtain the real part. Assumes f(-E) = f(E)* and requires Im[f(0 <= E <= E_max)].

    Input:
        im: np.ndarray (N_E, N_G)
    Output:
        re: np.ndarray (N_E, N_G), Note that Re(eps) = eps_re + 1 
    """
    E = np.arange(0, E_max+dE, dE)
    re = np.empty_like(im)
    N_G, N_E = im.shape
    for nE in range(N_E):
        E_pv = np.delete(E, nE) #removes Ei = En for principal value
        En = E[nE]
        for nG in range(N_G):
            im_pv = np.concatenate((im[nG, :nE], im[nG, nE+1:]))

            s = 0
            for ns in range(N_E-1):
                s += E_pv[ns]*im_pv[ns]/(E_pv[ns]**2 - En**2)

            re[nG,nE] = 2/np.pi*dE*(s - 0.5*(E_pv[0]*im_pv[0]/(E_pv[0]**2-En**2) + E_pv[-1]*im_pv[-1]/(E_pv[-1]**2-En**2))) #trapezoid rule
    return re

@nb.njit()
def kramerskronig_re2im_causal(re, E_max, dE):
    """
    Computes the Kramers-Kronig transformation of the input real part to obtain the imaginary part. Assumes f(-E) = f(E)* and requires Re[f(0 <= E <= E_max)].

    Input:
        re: np.ndarray (N_E, N_G)
    Output:
        im: np.ndarray (N_E, N_G)
    """
    E = np.arange(0, E_max+dE, dE)
    im = np.empty_like(re)
    N_G, N_E = re.shape
    for nE in range(N_E):
        E_pv = np.delete(E, nE) #removes Ei = En for principal value
        En = E[nE]
        for nG in range(N_G):
            re_pv = np.concatenate((re[nG, :nE], re[nG, nE+1:]))

            s = 0
            for ns in range(N_E-1):
                s += re_pv[ns]/(E_pv[ns]**2 - En**2)

            im[nG,nE] = -2/np.pi*dE*En*(s - 0.5*(re_pv[0]/(E_pv[0]**2-En**2) + re_pv[-1]/(E_pv[-1]**2-En**2))) #trapezoid rule
    return im

def kramerskronig_lfe_causal(eps_delta):
    """
    THIS FUNCTION SHOULD NOT BE USED FOR EPS_LFE - IT IS HERE FOR TESTING PURPOSES ONLY.
    The general KK transformation should be used for eps_LFE

    Calculates principal value part of Re(eps) and Im(eps) for LFEs. This function is parallelized over G-vectors (over first G-vector of eps_delta).

    Input:
        eps_delta: np.ndarray of shape (N_G, N_G, N_E): Dirac delta part of dielectric function

    Output: 
        epa_pv: np.ndarray of shape (N_G, N_G, N_E): Principal value part of dielectric function
    
    """
    import multiprocessing as mp
    
    eps_delta_re = np.real(eps_delta)
    eps_delta_im = np.imag(eps_delta)
    with mp.get_context('fork').Pool(mp.cpu_count()) as p:
        eps_pv_re = p.map(kramerskronig_im2re_causal, eps_delta_re)
        eps_pv_im = p.map(kramerskronig_re2im_causal, eps_delta_im)
    eps_pv = np.array(eps_pv_re) - 1j*np.array(eps_pv_im)
    return eps_pv