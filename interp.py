import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.interpolate import RBFInterpolator

import input_parameters as parmt
from epsilon_routines import epsilon_r, kramerskronig
from binning import gen_bin_centers, spherical_to_cartesian, cartesian_to_spherical

#Testing interpolation for bins missing Im(eps) data
def main():
    q = np.arange(0.01, 1, parmt.dq) #alpha me
    E = np.arange(0, 50+parmt.dE, parmt.dE) #eV

    binned_eps = 10/4/np.pi*np.imag(np.load('test_resources/binned_eps/cc-pvtz_pbe_6_bg1.12_1q_50E.npy'))
    bin_centers_sph = gen_bin_centers(q_max=1)
    bin_centers = spherical_to_cartesian(bin_centers_sph) #need cartesian coordinates for interpolator
    eps_r_im = epsilon_r(bin_centers_sph, binned_eps, eps_dtype='float')
    eps_r_re = kramerskronig(eps_r_im)

    N_ang_bins = (parmt.N_phi*(parmt.N_theta-2)+2)
    binned_eps = binned_eps[:-N_ang_bins] #get rid of extra bins and only take imaginary part

    #nan indices:
    nan_loc = np.where(np.isnan(binned_eps[:,0]))[0] #if data is missing for bin, then all energies will be nan so only need to check one of them
    interp_loc = np.where(np.invert(np.isnan(binned_eps[:,0])))[0]

    nan_bins = bin_centers[nan_loc]

    interp_bins = bin_centers[interp_loc]
    interp_eps = binned_eps[interp_loc]

    interp1 = RBFInterpolator(interp_bins, interp_eps, kernel='thin_plate_spline')(nan_bins)
    binned_eps[nan_loc] = interp1 #replace nans with interpolated data
    eps_int1_r_im = epsilon_r(bin_centers_sph, binned_eps, eps_dtype='float')
    eps_int1_r_re = kramerskronig(eps_int1_r_im)

    interp2 = RBFInterpolator(interp_bins, interp_eps, kernel='cubic')(nan_bins)
    binned_eps[nan_loc] = interp2 #replace nans with interpolated data
    eps_int2_r_im = epsilon_r(bin_centers_sph, binned_eps, eps_dtype='float')
    eps_int2_r_re = kramerskronig(eps_int2_r_im)

    interp3 = RBFInterpolator(interp_bins, interp_eps, kernel='gaussian', epsilon=10)(nan_bins)
    binned_eps[nan_loc] = interp3 #replace nans with interpolated data
    eps_int3_r_im = epsilon_r(bin_centers_sph, binned_eps, eps_dtype='float')
    eps_int3_r_re = kramerskronig(eps_int3_r_im)

    interp4 = RBFInterpolator(interp_bins, interp_eps, kernel='linear')(nan_bins)
    binned_eps[nan_loc] = interp4 #replace nans with interpolated data
    eps_int4_r_im = epsilon_r(bin_centers_sph, binned_eps, eps_dtype='float')
    eps_int4_r_re = kramerskronig(eps_int4_r_im)

    n = 5
    fig, ax = plt.subplots(2, n, figsize=(4*n,8))

    c = 'coolwarm_r'
    th = 1e-4
    re_min = np.min(np.real(eps_r_re[np.invert(np.isnan(eps_r_re))]))
    re_max = np.max(np.real(eps_r_re[np.invert(np.isnan(eps_r_re))]))
    re_max = max(-1*re_min, re_max)

    im0 = ax[(0,0)].pcolormesh(E, q, eps_r_im, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im1 = ax[(1,0)].pcolormesh(E, q, eps_r_re, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im2 = ax[(0,1)].pcolormesh(E, q, eps_int1_r_im, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im3 = ax[(1,1)].pcolormesh(E, q, eps_int1_r_re, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im4 = ax[(0,2)].pcolormesh(E, q, eps_int2_r_im, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im5 = ax[(1,2)].pcolormesh(E, q, eps_int2_r_re, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im6 = ax[(0,3)].pcolormesh(E, q, eps_int3_r_im, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im7 = ax[(1,3)].pcolormesh(E, q, eps_int3_r_re, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im6 = ax[(0,4)].pcolormesh(E, q, eps_int4_r_im, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))
    im7 = ax[(1,4)].pcolormesh(E, q, eps_int4_r_re, cmap=c, norm=matplotlib.colors.SymLogNorm(th, vmin=-re_max, vmax=re_max))

    ax[(0,0)].set_title(r'Im($\epsilon$)')
    ax[(1,0)].set_title(r'Re($\epsilon$)')
    ax[(0,1)].set_title(r'thin plate spline')
    ax[(0,2)].set_title(r'cubic')
    ax[(0,3)].set_title(r'gaussian, epsilon=0.1')
    ax[(0,4)].set_title(r'linear')

    for i in [0,1]:
        for j in range(n):
            ax[(i,j)].set_xlabel('E (eV)')
            ax[(i,j)].set_ylabel(r'q ($\alpha m_e$)')
    plt.tight_layout()
    plt.show()

def interpolation_test():
    bin_centers_sph = gen_bin_centers(q_max=1)
    bin_centers = spherical_to_cartesian(bin_centers_sph) #need cartesian coordinates for interpolator

    test = bin_centers[:,0] #test function is f(x,y,z) = x
    input_coords = bin_centers[(bin_centers[:,0] < 0.6) | (bin_centers[:,0] > 0.8)]
    test = test[(bin_centers[:,0] < 0.6) | (bin_centers[:,0] > 0.8)] #remove some data

    plt.plot(input_coords[:,0], test, 'o', label=r'Input: $f(x,y,z)=x$')

    x_missing = np.linspace(0.6, 0.8, 5)
    coords_missing = np.stack([x_missing, np.zeros_like(x_missing), np.zeros_like(x_missing)], axis=1)
    interp = RBFInterpolator(input_coords, test, kernel='thin_plate_spline')(coords_missing)
    #get singular matrix error if two input coordinates are the same

    plt.plot(x_missing, interp, 'o', label='Interpolated')
    plt.title('Interpolation test')
    plt.xlabel('x')
    plt.legend()
    plt.show()