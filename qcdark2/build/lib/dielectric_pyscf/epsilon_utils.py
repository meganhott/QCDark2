import numpy as np
from scipy.interpolate import LinearNDInterpolator, make_interp_spline

from dielectric_pyscf.binning import spherical_to_cartesian

# Post-processing functions for the dielectric function.

def epsilon_r(bin_centers, binned_eps, eps_dtype='complex'):
    """
    Calculate angular averaged dielectric function eps(|q|, E) from binned epsilon.
    """
    N_theta = np.unique(bin_centers[:,1]).shape[0]
    N_phi = np.unique(bin_centers[:,2]).shape[0]

    N_ang_bins = (N_phi*(N_theta-2)+2)
    r = np.unique(bin_centers[:,0])
    eps_r = np.zeros((r.shape[0], binned_eps.shape[1]), dtype=eps_dtype)
    for i, r_i in enumerate(r):
        eps_ri = binned_eps[i*N_ang_bins:(i+1)*N_ang_bins]
        eps_r[i] = np.nansum(eps_ri, axis=0) / (N_ang_bins - np.sum(np.isnan(eps_ri).astype(int), axis=0)) #treats nans as 0, want to average over all non-nan entries
    return eps_r

def interp_eps(bin_centers_sph, binned_eps):
    """
    Interpolate missing bins of epsilon. Can be used to interpolate Im(eps) before computing Re(eps) with Kramers-Kronig, or to interpolate Re(eps) and Im(eps) in the LFE case.

    bin_centers input should be in spherical coordinates
    """
    if bin_centers_sph.ndim == 2: # 3D coordinates
        bin_centers = spherical_to_cartesian(bin_centers_sph)
    else: # 1D coordinates: bins_centers only have q magnitude
        bin_centers = bin_centers_sph

    nan_loc = np.where(np.isnan(binned_eps[:,0]))[0] #Find indices of missing bins
    nan_bins = bin_centers[nan_loc]

    interp_loc = np.where(np.invert(np.isnan(binned_eps[:,0])))[0] #Use remaining bins for interpolation input
    interp_bins = bin_centers[interp_loc]
    interp_eps = binned_eps[interp_loc]

    binned_eps_interp = binned_eps.copy()

    if bin_centers_sph.ndim == 2: # 3D
        interp = LinearNDInterpolator(interp_bins, interp_eps)(nan_bins)
    else: # 1D
        interp = make_interp_spline(interp_bins, interp_eps, k=1)(nan_bins) # 1D linear interpolation
    binned_eps_interp[nan_loc] = interp #replace nans with interpolated data

    return binned_eps_interp