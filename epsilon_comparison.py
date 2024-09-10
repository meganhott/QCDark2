import numpy as np
import input_parameters as parmt
import matplotlib.pyplot as plt

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

chi_an[chi_an > 20] = 20 #for plot
chi_an[chi_an < -20] = -20

plt.imshow(chi_an, cmap='gnuplot2_r', origin='lower')
plt.colorbar()
plt.show()