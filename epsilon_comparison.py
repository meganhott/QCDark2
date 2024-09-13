import numpy as np
import input_parameters as parmt
import matplotlib.pyplot as plt
import matplotlib.colors
from routines import epsilon_r
from binning import *

q = np.arange(0.01, parmt.q_max, parmt.dq) #alpha me
E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE) #eV

#RPA
binned_eps = np.load('test_resources/epsilon_1q_30E/binned_eps.npy')
bin_centers = gen_bin_centers()
eps = epsilon_r(bin_centers, binned_eps)

#Lindhard model
VCell = 5.209e-9
nValence = 8
MCell = 52322355000.0
mElectron = 5.1099894e5
alpha = 1.0/137.03599908

def Lindhard(om, q, fp): #energy, q, 
    q = q*alpha*mElectron #convert q from ame to eV
    def plog(x):
        return np.log(np.abs(x)) + 1j*np.angle(x)
    ne = nValence/VCell
    kF = (3*np.pi**2*ne)**(1./3.)
    omp = np.sqrt(4*np.pi*alpha*ne/mElectron)
    vF = kF/mElectron
    Gp = fp*omp
    Qp = q/(2*kF) + (om + 1j*Gp)/(q*vF)
    Qm = q/(2*kF) - (om + 1j*Gp)/(q*vF)
    factor1 = 3*(omp**2)/(q**2)/(vF**2)
    factor2 = 0.5 + kF/(4*q)*(1-Qm**2)*plog((Qm+1)/(Qm-1)) + kF/(4*q)*(1-Qp**2)*plog((Qp+1)/(Qp-1))
    return 1 + factor1*factor2
VCell = 5.209e-9
nValence = 8
MCell = 52322355000.0
mElectron = 5.1099894e5
alpha = 1.0/137.03599908

def Lindhard(om, q, fp): #energy, q, 
    q = q*alpha*mElectron #convert q from ame to eV
    def plog(x):
        return np.log(np.abs(x)) + 1j*np.angle(x)
    ne = nValence/VCell
    kF = (3*np.pi**2*ne)**(1./3.)
    omp = np.sqrt(4*np.pi*alpha*ne/mElectron)
    vF = kF/mElectron
    Gp = fp*omp
    Qp = q/(2*kF) + (om + 1j*Gp)/(q*vF)
    Qm = q/(2*kF) - (om + 1j*Gp)/(q*vF)
    factor1 = 3*(omp**2)/(q**2)/(vF**2)
    factor2 = 0.5 + kF/(4*q)*(1-Qm**2)*plog((Qm+1)/(Qm-1)) + kF/(4*q)*(1-Qp**2)*plog((Qp+1)/(Qp-1))
    return 1 + factor1*factor2

E_mesh, q_mesh = np.meshgrid(E, q)
eps_l = Lindhard(E_mesh, q_mesh, 0.1)

c = 'coolwarm_r'
th = 1e-4
re_min = np.min(np.concatenate([np.real(eps[np.invert(np.isnan(eps))]), np.real(eps_l[np.invert(np.isnan(eps_l))])]))
re_max = np.max(np.concatenate([np.real(eps[np.invert(np.isnan(eps))]), np.real(eps_l[np.invert(np.isnan(eps_l))])]))
im_min = np.min(np.concatenate([np.imag(eps[np.invert(np.isnan(eps))]), np.imag(eps_l[np.invert(np.isnan(eps_l))])]))
im_max = np.max(np.concatenate([np.imag(eps[np.invert(np.isnan(eps))]), np.imag(eps_l[np.invert(np.isnan(eps_l))])]))
re_max = max(-1*re_min, re_max)

fig, ax = plt.subplots(2,2)

im0 = ax[(0,0)].pcolormesh(E, q, np.real(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
im1 = ax[(0,1)].pcolormesh(E, q, np.imag(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-im_max, vmax=im_max,))
im2 = ax[(1,0)].pcolormesh(E, q, np.real(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
im3 = ax[(1,1)].pcolormesh(E, q, np.imag(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-im_max, vmax=im_max,))
"""
im0 = ax[(0,0)].imshow(np.real(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max), origin='lower')
im1 = ax[(0,1)].imshow(np.imag(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-im_max, vmax=im_max,), origin='lower')
im2 = ax[(1,0)].imshow(np.real(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max), origin='lower')
im3 = ax[(1,1)].imshow(np.imag(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-im_max, vmax=im_max,), origin='lower')
"""
fig.colorbar(im0, ax=ax[(0,0)])
fig.colorbar(im1, ax=ax[(0,1)])
ax[(0,0)].set_title(r'Lindhard Re($\epsilon$)')
ax[(0,1)].set_title(r'Lindhard Im($\epsilon$)')
ax[(1,1)].axvline(x=1.12, ymin=0, ymax=1, color='k', linestyle='--')
fig.colorbar(im2, ax=ax[(1,0)])
fig.colorbar(im3, ax=ax[(1,1)])
ax[(1,0)].set_title(r'RPA Re($\epsilon$)')
ax[(1,1)].set_title(r'RPA Im($\epsilon$)')

for i in [0,1]:
    for j in [0,1]:
        ax[(i,j)].set_xlabel('E (eV)')
        ax[(i,j)].set_ylabel(r'q ($\alpha m_e$)')
plt.tight_layout()
plt.show()