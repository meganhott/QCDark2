import numpy as np
import input_parameters as parmt
import matplotlib.pyplot as plt
import matplotlib.colors
from routines import epsilon_r
from binning import *

#q = np.arange(0.01, 25.00, 0.02) #alpha me
q = np.arange(0.01, 1, parmt.dq) #alpha me
#E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE) #eV
E = np.arange(0, 15+parmt.dE, parmt.dE) #eV

#RPA
binned_eps = np.load('test_resources/epsilon_1q_15E/binned_eps.npy')
bin_centers = gen_bin_centers(q_max=1.0)
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

E_mesh, q_mesh = np.meshgrid(E, q)
eps_l = Lindhard(E_mesh, q_mesh, 0.1)

c = 'coolwarm_r'
th = 1e-6
re_min = np.min(np.array([np.real(eps), np.real(eps_l)]))
re_max = np.max(np.array([np.real(eps), np.real(eps_l)]))
im_min = np.min(np.array([np.imag(eps), np.imag(eps_l)]))
im_max = np.max(np.array([np.imag(eps), np.imag(eps_l)]))

fig, ax = plt.subplots(2,2)
im0 = ax[(0,0)].pcolormesh(E, q, np.real(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=re_min, vmax=re_max))
im1 = ax[(0,1)].pcolormesh(E, q, np.imag(eps_l), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=im_min, vmax=im_max,))
fig.colorbar(im0, ax=ax[(0,0)])
fig.colorbar(im1, ax=ax[(0,1)])
ax[(0,0)].set_title(r'Lindhard Re($\epsilon$)')
ax[(0,1)].set_title(r'Lindhard Im($\epsilon$)')

im2 = ax[(1,0)].pcolormesh(E, q, np.real(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=re_min, vmax=re_max))
im3 = ax[(1,1)].pcolormesh(E, q, np.imag(eps), cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=im_min, vmax=im_max,))
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

"""
#constants
m_e = 0.51099895000e6 #eV
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter

#model parameters for Si
e_0 = 11.3
tau = 1.563
omega_p = 16.6 #eV
q_TF = 4.13e3 #eV

eps_an = 1 + (1/(e_0-1) + tau*(q[:,None]*hbarc/bohr2m/q_TF)**2 + (q[:,None]*hbarc/bohr2m)**4/(4*m_e**2*omega_p**2) - (E[None,:]/omega_p)**2)**(-1) #(q,E)
chi_an = -(1/(e_0-1) + tau*(q[:,None]*hbarc/bohr2m/q_TF)**2 + (q[:,None]*hbarc/bohr2m)**4/(4*m_e**2*omega_p**2) - (E[None,:]/omega_p)**2)**(-1)

chi_an[chi_an > 100] = 100 #for plot
chi_an[chi_an < -100] = -100

binned_chi = np.load('test_resources/binned_chi1.npy')
bin_centers = gen_bin_centers()
chi = epsilon_r(bin_centers, binned_chi)
chi_r = np.real(chi) #analytic approx only computes real part
chi_r[chi_r > 100] = 100 #for plot
chi_r[chi_r < -100] = -100
chi_i = np.imag(chi)
chi_i[chi_i > 100] = 100 #for plot
chi_i[chi_i < -100] = -100

fig, ax = plt.subplots(1,2)
im0 = ax[0].imshow(chi_an, cmap='gnuplot2_r', origin='lower')
im1 = ax[1].imshow(chi_r, cmap='gnuplot2_r', origin='lower')
fig.colorbar(im0, ax=ax[0], orientation='horizontal')
fig.colorbar(im1, ax=ax[1], orientation='horizontal')

fig1, ax1 = plt.subplots(1,2)
im10 = ax1[0].imshow(chi_r, cmap='gnuplot2_r', origin='lower')
im11 = ax1[1].imshow(chi_i, cmap='gnuplot2_r', origin='lower')
fig1.colorbar(im10, ax=ax1[0], orientation='horizontal')
fig1.colorbar(im11, ax=ax1[1], orientation='horizontal')

ax1[0].set_title('Re(chi)')
ax1[1].set_title('Im(chi)')
for i in [0,1]:
    ax1[i].set_xlabel('E (0-15eV)')
    ax1[i].set_ylabel('q (0-1ame)')

    ax[i].set_xlabel('E (0-15eV)')
    ax[i].set_ylabel('q (0-1ame)')

plt.show()
"""