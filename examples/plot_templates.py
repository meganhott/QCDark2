import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib

plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def elf(eps):
    return np.imag(eps)/((np.imag(eps))**2+np.real(eps)**2)

def plot_epsilon(epsilon=None, eps_r=None, q=None, E=None, suptitle=None, th=1e-4):
    """
    Can either specify df object (epsilon) or dielectric function as np array (eps_r) along with q and E arrays
    """
    if epsilon is not None:
        eps_r = epsilon.eps
        elf_eps = epsilon.elf()
        q = epsilon.q
        E = epsilon.E
    elif all([eps_r is not None, q is not None, E is not None]):
        elf_eps = elf(eps_r)
        q = q
        E = E
    else:
        raise KeyError('Either epsilon must be input as a df object, or eps_r, q, and E must be specified as numpy arrays.')

    c_re = 'coolwarm_r'
    c_im = 'gnuplot2_r'
    re_min = np.min(np.real(eps_r[np.invert(np.isnan(eps_r))]))
    re_max = np.max(np.real(eps_r[np.invert(np.isnan(eps_r))]))
    im_max = np.max(np.imag(eps_r[np.invert(np.isnan(eps_r))]))
    elf_max = np.max(elf_eps[np.invert(np.isnan(elf_eps))])
    re_max = max(-1*re_min, re_max)

    fig, ax = plt.subplots(1,3, figsize=(9, 3.5), layout='constrained')

    im0 = ax[0].pcolormesh(E, q, np.real(eps_r), cmap=c_re, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im1 = ax[1].pcolormesh(E, q, np.imag(eps_r), cmap=c_im, norm=matplotlib.colors.LogNorm(th))
    im2 = ax[2].pcolormesh(E, q, elf_eps, cmap=c_im, norm=matplotlib.colors.LogNorm(th))

    ax[0].set_title(r'$\mathrm{Re}[\epsilon(\omega, q)$]')
    ax[1].set_title(r'$\mathrm{Im}[\epsilon(\omega, q)$]')
    ax[2].set_title(r'$\mathrm{Im}[-1/(\omega, q)]$')

    if suptitle is not None:
        fig.suptitle(suptitle)

    for i in [0,1,2]:
        ax[i].set_xlabel(r'$\omega$ [eV]')
        ax[i].set_ylabel(r'q [$\alpha m_e$]')

    plt.colorbar(im0, ax=ax[0], location='bottom')
    plt.colorbar(im1, ax=ax[1], location='bottom')
    plt.colorbar(im2, ax=ax[2], location='bottom')

    return ax

def plot_dynamic_structure_factor(epsilon, suptitle=None, th=1e-4):
    S = epsilon.S()
    q = epsilon.q
    E = epsilon.E

    fig, ax = plt.subplots(1, 1, figsize=(4,3.5), layout='constrained')

    im0 = ax.pcolormesh(E, q, S, cmap='gnuplot2_r', norm=matplotlib.colors.LogNorm(th))
    plt.colorbar(im0, ax=ax, location='right')

    ax.set_title(r'$S(\omega, q)$')
    ax.set_xlabel(r'$\omega$ [eV]')
    ax.set_ylabel(r'$q$ [$\alpha m_e$]')

    if suptitle is not None:
        fig.suptitle(suptitle)

    return ax

def plot_dielectric_function_q(epsilon=None, eps_r=None, E=None, suptitle=None, ax=None, label=None, color=None, q=0, dq=None):
    if epsilon is not None:
        eps_r = epsilon.eps
        E = epsilon.E
        dq = epsilon.q[1] - epsilon.q[0]
        q_i = int(q/dq) # index of momentum
    elif all([eps_r is not None, E is not None]):
        if dq is None:
            if q == 0:
                q_i = 0
            else:
                raise KeyError('For nonzero q, you also need to specify dq') 
        else:
            q_i = int(q/dq) # index of momentum
    else:
        raise KeyError('Either epsilon must be input as a df object, or eps_r, q, and E must be specified as numpy arrays.')

    try: # add lines to existing plot or make new plot?
        if ax is None:
            fig, ax = plt.subplots(1,3, figsize=(9,3), layout='constrained')
    except:
        pass

    ax[0].plot(E, np.real(eps_r[q_i]), label=label, color=color)
    ax[0].axhline(0, color='k', linestyle='--')
    ax[0].set_title(r'$\mathrm{Re}[\epsilon(\omega, q = \:$' + f'{round(q, 2)}' +  r'$)]$')

    ax[1].plot(E, np.imag(eps_r[q_i]), label=label, color=color)
    ax[1].axhline(0, color='k', linestyle='--')
    ax[1].set_title(r'$\mathrm{Im}[\epsilon(\omega, q =\: $' + f'{round(q, 2)}' +  r'$)]$')
    if label is not None:
        ax[1].legend(fontsize=8)

    ax[2].plot(E, elf(eps_r)[q_i], label=label, color=color)
    ax[2].set_title(r'$\mathrm{Im}[-1/\epsilon(\omega, q =\: $' + f'{round(q, 2)}' +  r'$)]$')

    if suptitle is not None:
        fig.suptitle(suptitle)

    for i in [0,1,2]:
        ax[i].set_xlabel(r'$\omega$ [eV]')
    
    return ax

def plot_recoil_spectrum(R_Q, ax=None, label=None, color=None, linewidth=None, suptitle=None, Q_max=None):
    """
    Plots the electron recoil spectrum, R_Q
    """

    try:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4,3.5), layout='constrained')
    except:
        pass

    Q = np.arange(1, R_Q.shape[0]+1)
    if Q_max is not None:
        Q = Q[:Q_max]
    else:
        Q_max = Q.shape[0]-1

    for i, Q_i in enumerate(Q):
        if i == 0:
            im = ax.plot([Q_i - 0.45, Q_i + 0.45], [R_Q[i], R_Q[i]], color=color, linewidth=linewidth, label=label)
        else:
            ax.plot([Q_i - 0.45, Q_i + 0.45], [R_Q[i], R_Q[i]], color=im[0].get_color(), linewidth=linewidth)

        ax.set_yscale('log')
        ax.set_xlabel(r'$Q$ $[e^{-}]$')
        ax.set_ylabel(r'$\Delta R_Q$ [Events/kg/year]')
        ax.set_xticks(np.arange(1,Q_max+1))
        ax.set_xlim([0.5, Q_max+0.5])
        if label is not None:
            ax.legend()

    if suptitle is not None:
        fig.suptitle(suptitle)

    return ax