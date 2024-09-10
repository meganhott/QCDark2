import numpy as np
import input_parameters as parmt
import matplotlib.pyplot as plt
from routines import epsilon_r
from binning import *

#constants
m_e = 0.51099895000e6 #eV
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter

#model parameters for Si
e_0 = 11.3
tau = 1.563
omega_p = 16.6 #eV
q_TF = 4.13e3 #eV

#q = np.arange(0.01, 25.00, 0.02) #alpha me
q = np.arange(0.01, parmt.q_max, parmt.dq) #alpha me
#E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE) #eV
E = np.arange(0, parmt.E_max+parmt.dE, parmt.dE) #eV

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