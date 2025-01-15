import numpy as np
import matplotlib.pyplot as plt
import pyscf.pbc.gto as pbcgto

from dft_testing_functions import *
import dielectric_pyscf.materials.input_paramters_Ge as parmt_Ge

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

for i in range(energy.shape[1]):
    plt.plot(kpath, energy[:,i], 'k')

for h in sp_points:
    plt.axvline(h, color='k', alpha=0.5)
plt.xticks(ticks=sp_points, labels=sp_labels)
plt.xlim([min(kpath), max(kpath)])

plt.ylim([-15,15])


plt.show()