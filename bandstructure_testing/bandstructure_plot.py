import numpy as np
import matplotlib.pyplot as plt
import pyscf.pbc.gto as pbcgto

from dft_testing_functions import *
import dielectric_pyscf.materials.input_paramters_Ge as parmt_Ge
import dielectric_pyscf.materials.input_paramters_GaAs as parmt_GaAs
import dielectric_pyscf.materials.input_paramters_Si as parmt_Si

def plot_Ge():
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

    #pyscf 8k
    """
    energy = np.load('/gpfs/scratch/mhott/dielectric_pyscf/bandstructure_testing/bands_data/Ge_cc-pvdz_8k_pbe_p-12.npy') #(k, band)
    energy = energy*27.2113862 #Hartrees to eV

    energy = energy - energy[:,:32].max() #setting top of valence band to 0 eV
    gap = min(energy[:,32]) - max(energy[:,31])
    energy[:,32:] = energy[:,32:] - gap + bg

    for i in range(energy.shape[1]):
        if i==0:
            plt.plot(kpath, energy[:,i], 'k', label='pyscf 8k PBE cc-pvdz')
        else:
            plt.plot(kpath, energy[:,i], 'k')
    """

    #pyscf 10k
    energy = np.load('/gpfs/scratch/mhott/dielectric_pyscf/bandstructure_testing/bands_data/Ge_cc-pvdz_pbe_10k_bands.npy') #(k, band)
    energy = energy*27.2113862 #Hartrees to eV

    energy = energy - energy[:,:32].max() #setting top of valence band to 0 eV
    gap = min(energy[:,32]) - max(energy[:,31])
    energy[:,32:] = energy[:,32:] - gap + bg

    for i in range(energy.shape[1]):
        if i==0:
            plt.plot(kpath, energy[:,i], 'b', label='pyscf 10k PBE cc-pvdz')
        else:
            plt.plot(kpath, energy[:,i], 'b')

    """
    #QE
    bands_qe = np.loadtxt('/gpfs/home/mhott/q-e/Ge/Ge_bands_pbe_standard.dat.gnu')

    k = np.unique(bands_qe[:, 0])
    bands_qe = np.reshape(bands_qe[:, 1], (-1, len(k)))
    bands_qe = bands_qe - max(bands_qe[13]) #setting top of valence to 0eV
    gap = min(bands_qe[14]) - max(bands_qe[13])
    bands_qe[14:] = bands_qe[14:] - gap + bg #scissor correction

    for band in range(len(bands_qe)):
        if band==0:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'b', label='QE standard PBE')
        else:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'b')
    """

    #QE KE 70Ry
    """
    bands_qe = np.loadtxt('/gpfs/home/mhott/q-e/Ge/Ge_bands_pbe_stringent.dat.gnu')

    k = np.unique(bands_qe[:, 0])
    bands_qe = np.reshape(bands_qe[:, 1], (-1, len(k)))

    bands_qe = bands_qe - max(bands_qe[21]) #setting top of valence to 0eV
    gap = min(bands_qe[22]) - max(bands_qe[21])
    bands_qe[22:] = bands_qe[22:] - gap + bg #scissor correction

    for band in range(len(bands_qe)):
        if band==0:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r', label='QE PBE 70')
        else:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r')
    """

    #QE KE 100Ry
    bands_qe = np.loadtxt('/gpfs/home/mhott/q-e/Ge/Ge_bands.dat.gnu')

    k = np.unique(bands_qe[:, 0])
    bands_qe = np.reshape(bands_qe[:, 1], (-1, len(k)))

    bands_qe = bands_qe - max(bands_qe[21]) #setting top of valence to 0eV
    gap = min(bands_qe[22]) - max(bands_qe[21])
    bands_qe[22:] = bands_qe[22:] - gap + bg #scissor correction

    for band in range(len(bands_qe)):
        if band==0:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r', label='QE PBE 100')
        else:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r')


    for h in sp_points:
        plt.axvline(h, color='k', alpha=0.5)
    plt.xticks(ticks=sp_points, labels=sp_labels)
    plt.xlim([min(kpath), max(kpath)])

    plt.ylim([-20,30])
    plt.title('Germanium')
    plt.legend()

    plt.savefig('bandstructure_testing/bands_plots/Ge_plot')
    #plt.show()

def plot_GaAs():
    bg = parmt_GaAs.scissor_bandgap #experimental bandgap

    cell = pbcgto.M(
        a = np.asarray(parmt_Si.lattice_vectors),
        atom = parmt_Si.atomloc,
        basis = parmt_Si.mybasis,
        cart = True,
        verbose = parmt_Si.pyscf_outlev,
        output = parmt_Si.pyscf_outfile,
        ecp = parmt_Si.effective_core_potential,
        precision = parmt_Si.precision,
        pseudo = parmt_Si.pseudo
        )
    cell.build()

    #pyscf 10k
    val = 32
    band_kpts, kpath, sp_points, sp_labels = initialize_kpts_fcc(initialize_cell()[0])
    energy = np.load('/gpfs/scratch/mhott/dielectric_pyscf/bandstructure_testing/bands_data/GaAs_cc-pvdz_pbe_10k_bands.npy') #(k, band)
    energy = energy*27.2113862 #Hartrees to eV

    energy = energy - energy[:,:val].max() #setting top of valence band to 0 eV
    gap = min(energy[:,val]) - max(energy[:,(val-1)])
    energy[:,val:] = energy[:,val:] - gap + bg

    for i in range(energy.shape[1]):
        if i==0:
            plt.plot(kpath, energy[:,i], 'b', label='pyscf 10k PBE cc-pvdz')
        else:
            plt.plot(kpath, energy[:,i], 'b')

    #QE KE 100Ry
    bands_qe = np.loadtxt('/gpfs/home/mhott/q-e/GaAs/GaAs_bands.dat.gnu')
    val = 22
    k = np.unique(bands_qe[:, 0])
    bands_qe = np.reshape(bands_qe[:, 1], (-1, len(k)))

    bands_qe = bands_qe - max(bands_qe[val-1]) #setting top of valence to 0eV
    gap = min(bands_qe[val]) - max(bands_qe[val-1])
    bands_qe[val:] = bands_qe[val:] - gap + bg #scissor correction

    for band in range(len(bands_qe)):
        if band==0:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r', label='QE PBE 100')
        else:
            plt.plot(k * kpath[-1]/k[-1], bands_qe[band, :], 'r')

    for h in sp_points:
        plt.axvline(h, color='k', alpha=0.5)
    plt.xticks(ticks=sp_points, labels=sp_labels)
    plt.xlim([min(kpath), max(kpath)])

    plt.ylim([-30,30])
    plt.title('GaAs')
    plt.legend()

    plt.savefig('bandstructure_testing/bands_plots/GaAs_plot')
    #plt.show()

def plot_Si():
    """
    cell = pbcgto.M(
        a = np.asarray(parmt_GaAs.lattice_vectors),
        atom = parmt_GaAs.atomloc,
        basis = parmt_GaAs.mybasis,
        cart = True,
        verbose = parmt_GaAs.pyscf_outlev,
        output = parmt_GaAs.pyscf_outfile,
        ecp = parmt_GaAs.effective_core_potential,
        precision = parmt_GaAs.precision,
        pseudo = parmt_GaAs.pseudo
        )
    cell.build()
    """

    #cc-pvtz
    val = 14
    band_kpts, kpath, sp_points, sp_labels = initialize_kpts_fcc(initialize_cell()[0])

    colors = ['b', 'g', 'r']
    for j, b in enumerate(['cc-pvtz', 'cc-pvqz', 'cc-pwcvtz']):
        energy = np.load(f'/gpfs/scratch/mhott/dielectric_pyscf/bandstructure_testing/bands_data/Si_{b}_pbe_8k_bands.npy') #(k, band)
        energy = energy*27.2113862 #Hartrees to eV

        energy = energy - energy[:,:val].max() #setting top of valence band to 0 eV
        gap = min(energy[:,val]) - max(energy[:,(val-1)])
        #energy[:,val:] = energy[:,val:] - gap + bg

        for i in range(energy.shape[1]):
            if i==0:
                plt.plot(kpath, energy[:,i], color=colors[j], label=b)
            else:
                plt.plot(kpath, energy[:,i], color=colors[j])

    for h in sp_points:
        plt.axvline(h, color='k', alpha=0.5)
    plt.xticks(ticks=sp_points, labels=sp_labels)
    plt.xlim([min(kpath), max(kpath)])

    plt.ylim([-15,50])
    plt.title('Si')
    plt.legend()

    plt.savefig('bandstructure_testing/bands_plots/Si_plot')


if __name__ == '__main__':
    #plot_Ge()
    #plot_GaAs()
    plot_Si()