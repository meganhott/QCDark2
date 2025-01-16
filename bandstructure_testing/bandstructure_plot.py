import numpy as np
import matplotlib.pyplot as plt
import pyscf.pbc.gto as pbcgto

from dft_testing_functions import *
import dielectric_pyscf.materials.input_paramters_Ge as parmt_Ge

bg = 0.67 #experimental bandgap

cell = pbcgto.M(
    a = np.asarray(parmt_Ge.lattice_vectors),
    atom = parmt_Ge.atomloc,
    basis = parmt_Ge.mybasis,
    cart = True,
    verbose = parmt_Ge.pyscf_outlev,
    output = parmt_Ge.pyscf_outfile,
    ecp = parmt_Ge.effective_core_potential,
    precision = parmt_Ge.precision,
    pseudo = parmt_Ge.pseudo
    )
cell.build()


band_kpts, kpath, sp_points, sp_labels = initialize_kpts_fcc(initialize_cell()[0])
energy = np.load('bandstructure_testing/Ge_cc-pvdz_8k_pbe_p-12.npy') #(k, band)
energy = energy*27.2113862 #Hartrees to eV

energy = energy - energy[:,:32].max() #setting top of valence band to 0 eV
gap = min(energy[:,32]) - max(energy[:,31])
energy[:,32:] = energy[:,32:] - gap + bg

for i in range(energy.shape[1]):
    plt.plot(kpath, energy[:,i], 'k')

#QE
bands_qe = np.loadtxt('bandstructure_testing/Ge_bands.dat.gnu')

k = np.unique(bands_qe[:, 0])
bands_qe = np.reshape(bands_qe[:, 1], (-1, len(k)))
bands_qe = bands_qe - max(bands_qe[13]) #setting top of valence to 0eV
gap = min(bands_qe[14]) - max(bands_qe[13])
bands_qe[14:] = bands_qe[14:] - gap + bg #scissor correction

for band in range(len(bands_qe)):
    plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'b')


for h in sp_points:
    plt.axvline(h, color='k', alpha=0.5)
plt.xticks(ticks=sp_points, labels=sp_labels)
plt.xlim([min(kpath), max(kpath)])

plt.ylim([-15,15])


plt.show()