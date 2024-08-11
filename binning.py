import numpy as np
import input_parameters as parmt
import multiprocessing as mp
from functools import partial
import logging
from routines import time_wrapper

n = 10 #rounding precision

def spherical_to_cartesian(sph: np.ndarray, unique=False) -> np.ndarray:
    """
    Convert all vectors given in spherical polar coordinates to corresponding vectors in cartesian coordinates,
    and remove non-unique vectors. They can come up when theta = 0 or pi, because phi becomes degenerate. 
    Inputs:
        sph:    np.ndarray of shape (N, 3)
                arr[i] = [r, theta, phi]
    Outputs:
        cart:   np.ndarray of shape (N, 3)
                cart[i] = [x, y, z]
    """
    z = sph[:,0]*np.cos(sph[:,1])
    r = sph[:,0]*np.sin(sph[:,1])
    x = r*np.cos(sph[:,2])
    y = r*np.sin(sph[:,2])
    cart = np.round(np.transpose([x, y, z]), 10)
    if unique:
        cart = np.unique(cart, axis=0)
    return cart

def cartesian_to_spherical(cart: np.ndarray, unique=False) -> np.ndarray:
    """
    Converts all vectors given in cartesian coordinates to corresponding vectors in spherical polar coordinates,
    and remove non-unique vectors.
    Inputs:
        cart:   np.ndarray of shape (N, 3)
                cart[i] = [x, y, z]
    Outputs:
        sph:    np.ndarray of shape (N, 3)
                sph[i] = [r, theta, phi]
                0 <= theta <= pi, -pi <= phi < pi 
    """
    r = np.sqrt(cart[:,0]**2 + cart[:,1]**2 + cart[:,2]**2)
    theta = np.arccos(cart[:,2]/r)
    phi = np.arctan2(cart[:,1],cart[:,0])
    for i,th in enumerate(theta):
        if th == 0 or round(th, 9) == round(np.pi, 9):
            phi[i] = 0
        if np.isnan(th): #when r = 0
            theta[i] = 0
            phi[i] = 0
        if round(phi[i],9) == round(np.pi, 9): 
            phi[i] = -np.pi
    sph = np.round(np.transpose([r, theta, phi]), n)
    if unique:
        sph = np.unique(sph, axis=0)
    return sph

def construct_theta_bins() -> np.ndarray:
    """
    Construct bin edges in theta, such that the z and -z axis are endpoints, and each bin center
    follows the following requirements:
        bin_center[0], bin_center[-1] = 0, pi
        {integral _bin_edge[i] ^bin_edge[i+1]} d cos(theta) = c * f(bin_center[i]),
                where f(i) =    1       if i = 0 or i = -2 and
                                N_phi   otherwise
                and c is a constant
    Then, to do an integral in solid angle, we have
        {integral _0 ^pi} d -cos(theta) {integral _0 ^2*pi} d phi f(theta, phi) = sum(f(theta, phi), theta, phi) * (4*pi)/f(theta, phi).shape[0]
    Inputs:
        N_theta:    int
        N_phi:      int
    Returns:
        bin_center:   np.ndarray
    """
    i = np.arange(parmt.N_theta - 2) + .5
    N = 2 + (parmt.N_theta - 2)*parmt.N_phi # Note: we begin counting from 0 in our bin_edges, so to conserve shape we must subtract 1. 
    b_i = np.append([1], 1 - 2/N*(1 + i*parmt.N_phi))
    b_i = np.append(b_i, [-1])
    return np.arccos(b_i)

def construct_all_solid_angles() -> np.ndarray:
    """
    Construct points in theta and phi, such that the integral over solid angles is well-approximated by a trapezoidal rule integration law in theta.
    """
    if type(parmt.N_theta) != int or type(parmt.N_phi) != int:
        logging.info('Raising exception, in input_parameters.py, N_theta and N_phi must be of the type int.')
        raise Exception('In input_parameters.py, N_theta and N_phi must be of the type int.')
    if not parmt.N_theta%2:
        logging.info('! WARNING: Given N_theta = {} is even and will not contain points in the x-y plane.\nThe accuracy of the dielectric function will remain unaffected.'.format(parmt.N_theta))
    theta_bins = construct_theta_bins()
    phi_bins = np.arange(-0.5, 0.5, 1./parmt.N_phi)*2*np.pi #-pi <= phi <pi
    solid_angles = []
    for theta in theta_bins:
        if theta == 0 or round(theta, 9) == round(np.pi, 9):
            solid_angles.append([theta, 0.])
        else:
            for phi in phi_bins:
                solid_angles.append([theta, phi])
    return np.array(solid_angles)

def gen_bin_centers(cartesian=False) -> np.ndarray:
    Omega = construct_all_solid_angles()
    qr = np.arange(parmt.dq*0.5, parmt.q_max + parmt.dq*0.5, parmt.dq)
    qra = []
    for q in qr:
        for O in Omega:
            qra.append([q, O[0], O[1]])
    qra = np.array(qra)
    if cartesian: #Convert from spherical to cartesian
        qra = spherical_to_cartesian(qra)
    return np.round(qra, n)

@time_wrapper
def bin_eps_q(q, G_vectors, eps_q, bin_centers, tot_bin_eps, tot_bin_weights):
    """
    To Do:
    - Need to speed up
        - Should we make bin_centers into dict? Then I don't think we can use multiprocessing to add eps and w to each bin if dict = {(r,phi,theta): (tot_bin_eps, tot_bin_weight)}. We could instead just store the position of each bin center in a dict: dict = {(r,phi,theta): position in array(int)} and add all new eps and weights at the end. Or, since bin centers are generated procedurally, we can calculate which position in bin_centers we need to write to. 
    - Test phi=0/2pi edge cases
    - How should we deal with r_l < 0 and r_g > q_max? Currently they just don't match to any bins
    
    Notes:
    - We need to calculate phi weights before correcting for edge cases. For example, if we need to correct phi_bin=pi to phi_bin=-pi for some phi, then weighting after the correction will cause huge weight since phi_bin - phi ~ 2pi. On the other hand, we need to calculate theta weights after correcting for edge cases near 0 and pi

    Inputs:
        bin_centers: (N_bins,3): (r,theta,phi)
        eps_q: (G,E)
        tot_bin_eps: (N_bins,E): total w*eps for all q+G calculated so far
        tot_bin_weights: (N_bins) weights should be same for all energies 
    Outputs:
        updated tot_bin_eps and tot_bin_weights
    """
    #convert q+G to spherical coords
    qG_sph = cartesian_to_spherical(q + G_vectors)

    #determine closest r bin centers
    r_l = (np.floor(np.round((qG_sph[:,0]/parmt.dq - 0.5), n)) + 0.5)*parmt.dq
    r_g = (np.ceil(np.round((qG_sph[:,0]/parmt.dq - 0.5), n)) + 0.5)*parmt.dq

    w_r_l = 1 - (qG_sph[:,0] - r_l)/parmt.dq
    w_r_g = 1 - (r_g - qG_sph[:,0])/parmt.dq

    #determine closest cos(theta)
    bin_theta = np.unique(bin_centers[:,1]) #unique theta in bins
    theta_l = bin_theta[np.sum((np.round(qG_sph[:,1][:,None] - bin_theta, n) >= 0), axis=1) - 1]
    theta_g = bin_theta[np.sum((np.round(qG_sph[:,1][:,None] - bin_theta, n) > 0), axis=1)]

    #weight according to cos(theta)
    bin_diff = np.cos(theta_l) - np.cos(theta_g)
    w_theta_l = 1 - (np.cos(theta_l) - np.cos(qG_sph[:,1]))/bin_diff
    w_theta_g = 1 - (np.cos(qG_sph[:,1]) - np.cos(theta_g))/bin_diff

    #fix weights for theta_l = 0 and theta_g = pi
    w_theta_l[np.isnan(w_theta_l)] = 1
    w_theta_g[np.isnan(w_theta_g)] = 1
    
    #determine closest phi
    d_phi = 2*np.pi/parmt.N_phi
    phi_l = np.floor(qG_sph[:,2]/d_phi)*d_phi
    phi_g = np.ceil(qG_sph[:,2]/d_phi)*d_phi

    w_phi_l = 1 - (qG_sph[:,2] - phi_l)/d_phi
    w_phi_g = 1 - (phi_g - qG_sph[:,2])/d_phi

    #change phi_l<-pi and phi_g>pi, =pi 
    phi_l[phi_l < -np.pi] = np.pi - d_phi
    phi_g[phi_g == np.pi] = -np.pi
    phi_g[phi_g > np.pi] = -np.pi + d_phi

    all_closest_bins = np.round(np.stack([np.repeat(np.stack([r_l,r_g], axis=1), 4, axis=1), np.tile(np.repeat(np.stack([theta_l,theta_g], axis=1), 2, axis=1), 2), np.tile(np.stack([phi_l,phi_g], axis=1), 4)], axis=2), n) #(G,8,3)
    
    all_weights = np.prod(np.stack([np.repeat(np.stack([w_r_l,w_r_g], axis=1), 4, axis=1), np.tile(np.repeat(np.stack([w_theta_l,w_theta_g], axis=1), 2, axis=1), 2), np.tile(np.stack([w_phi_l,w_phi_g], axis=1), 4)], axis=2), axis=2) #(G,8)

    #match to bins
    #trade-off: 
    # dict: fast lookup but have to loop to add to existing values
    # array: slow lookup but can add everything at once and use multiprocessing

    with mp.get_context('fork').Pool(mp.cpu_count()) as p: #parallelization over bins
        res = p.map(partial(find_bin, eps_q=eps_q, all_closest_bins=all_closest_bins, all_weights=all_weights), bin_centers)
    new_bin_weights = np.array([i[0] for i in res])
    new_bin_eps = np.array([i[1] for i in res])

    tot_bin_weights += np.array(new_bin_weights)
    tot_bin_eps += np.array(new_bin_eps)

    return tot_bin_eps, tot_bin_weights

def find_bin(bin_center, eps_q, all_closest_bins, all_weights):
    """
    bin_center: (3,)
    eps_q: (G,E)
    bins: (G,8,3)
    weights: (G,8)
    """
    ind = np.array(np.where(((all_closest_bins[:,:,0] == bin_center[0]).astype(int) + (all_closest_bins[:,:,1] == bin_center[1]).astype(int) + (all_closest_bins[:,:,2] == bin_center[2]).astype(int)) == 3))

    eps_bin = eps_q[ind[0]]
    w_bin = all_weights[ind[0],ind[1]]

    w = np.sum(w_bin)
    eps_E = np.sum(w_bin[:,None]*eps_bin, axis=0)
    return w, eps_E
