import numpy as np
import matplotlib.pyplot as plt
from ase.dft.kpoints import sc_special_points, get_bandpath

from dielectric_functions import initialize_cell
from dft_routines import save_dft, KS_electronic_structure, KS_non_self_consistent_field
import input_parameters as parmt

def initialize_kpts_fcc(cell):
    points = sc_special_points('fcc')
    G = points['G']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    path = get_bandpath([L,G,X,W,K,G], cell.a, npoints=100)
    band_kpts = path.kpts
    kpath, sp_points, sp_labels = path.get_linear_kpoint_axis()
    band_kpts = cell.get_abs_kpts(band_kpts)
    return band_kpts, kpath, sp_points, sp_labels

def run_scf():
    cell = initialize_cell()[0]
    new_dft, dft_params = save_dft()
    if new_dft:
        return KS_electronic_structure(cell, dft_params)
    else:
        return None

def get_bands(kmf, filename):
    cell = initialize_cell()[0]
    band_kpts, kpath, sp_points, sp_labels = initialize_kpts_fcc(cell)
    energy = np.array(kmf.get_bands(band_kpts)[0])
    np.save(filename + '_bands', energy)
    return energy

def plot_bands(filename):
    energy = np.load(filename) #(k, band)
    for i in range(energy.shape[1]):
        plt.plot(energy[:,i], 'k')
    plt.show()

def main(filename):
    kmf = run_scf()
    energy = get_bands(filename)

if __name__ == '__main__':
    main(parmt.system_name)