import numpy as np
from numba import njit

n = 10 #rounding precision

def spherical_to_cartesian(sph, unique=False):
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

@njit
def cartesian_to_spherical(cart):
    """
    Converts all vectors given in cartesian coordinates to corresponding vectors in spherical polar coordinates.

    Notes:
    - Huge speedup using numba which is important for binning algorithm (288 ms to 3.8 ms for ~70000 vectors). Can't implement "unique" option with numba but I don't think we were using that anyway

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
    sph = np.round(np.stack((r, theta, phi), axis=1), n)
    return sph

def construct_theta_bins(N_theta, N_phi):
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
    i = np.arange(N_theta - 2) + .5
    N = 2 + (N_theta - 2)*N_phi # Note: we begin counting from 0 in our bin_edges, so to conserve shape we must subtract 1. 
    b_i = np.append([1], 1 - 2/N*(1 + i*N_phi))
    b_i = np.append(b_i, [-1])
    return np.arccos(b_i)

def construct_all_solid_angles(N_theta, N_phi):
    """
    Construct points in theta and phi, such that the integral over solid angles is well-approximated by a trapezoidal rule integration law in theta.
    """
    theta_bins = construct_theta_bins(N_theta, N_phi)
    phi_bins = np.arange(-0.5, 0.5, 1./N_phi)*2*np.pi #-pi <= phi <pi
    solid_angles = []
    for theta in theta_bins:
        if theta == 0 or round(theta, 9) == round(np.pi, 9):
            solid_angles.append([theta, 0.])
        else:
            for phi in phi_bins:
                solid_angles.append([theta, phi])
    return np.array(solid_angles)

def gen_bin_centers(q_max, q_min, dq, N_theta, N_phi, cartesian=False, dir=False):
    qr = np.linspace(q_min + dq*0.5, q_max + dq*0.5, int((q_max - q_min)/dq) + 1)
    if dir:
        return qr # no angular bins needed if direction already selected
    else:
        Omega = construct_all_solid_angles(N_theta, N_phi)
        qra = []
        for q in qr:
            for O in Omega:
                qra.append([q, O[0], O[1]])
        qra = np.array(qra)
        if cartesian: # convert from spherical to cartesian
            qra = spherical_to_cartesian(qra)
        return np.round(qra, n)

@njit
def bin_eps_q(q, G_vectors, eps_q, bin_centers, tot_bin_eps, tot_bin_weights):
    """
    Notes:
    - We need to calculate phi weights before correcting for edge cases. For example, if we need to correct phi_bin=pi to phi_bin=-pi for some phi, then weighting after the correction will cause huge weight since phi_bin - phi ~ 2pi. On the other hand, we need to calculate theta weights after correcting for edge cases near 0 and pi
    - Flattening bins and weights makes find_bins much faster, particularly when making the mask (750us for (G,8) -> 14us for (G*8) flattened)!
    - Takes 27s for all G vectors on my laptop (12 cores)
    - Multiprocessing does not seem to be working well, it's actually faster to run in a for loop (20s). May have to chunk into multiple processes (e.g. each process gets 1000 bin_centers to loop through) and run those in parallel for any speedup
    - Chunking is even slower than normal multiprocessing (~50s)
    - Way faster to loop through all_closest_bins: now ~1.8s. We can't implement multiprocessing for this method since we have multiple contributions being written to each bin.
    - numpy functions to find r, theta, phi bins are not parallelized but are very fast anyway. May be possible to parallelize bin construction for slight speedup, but slowest part of code is still the loop to bin epsilon. We can't parallelize this though since one bin will be written to many times, and past attempts at parallelizing this step were slower than current implementation.
    - Applying numba to the binning loop speeds that step up by a factor of ~2. We need to add extra "junk" rows to our binned epsilon and weights for bins beyond the ones included in bin_centers since we can't use the try-except statement in numba's c-implementation (it results in a seg fault since we're trying to access invalid memory)
    - all_closest_bins_id and all_weights were rewritten so whole function is now comapatible with numba. This reduces total run time per q vector to 0.51s!

    Timings: (on PC for 71137 G vectors)
    Converting to spherical: 4 ms (290 ms without numba)
    Finding r bins: 1.7 ms
    Finding phi bins: 1.5 ms
    Finding theta bins: 7.4 ms
    Constructing all closest bins array: 8.2 ms (27.1 ms before flattening)
    Constructing weights array: 2.6 ms (27.5 ms before flattening)
    Binning epsilon: 422 ms (1.07 s without numba)
    Total: 0.51 s

    Inputs:
        bin_centers: (N_bins,3): (r,theta,phi)
        eps_q: (G,E)
        tot_bin_eps: (N_bins,E): total w*eps for all q+G calculated so far
        tot_bin_weights: (N_bins) weights should be same for all energies 
    Outputs:
        updated tot_bin_eps and tot_bin_weights
    """
    N_theta = np.unique(bin_centers[:,1]).shape[0]
    N_phi = np.unique(bin_centers[:,2]).shape[0]
    q_min = np.min(bin_centers[:,0])
    dq = np.unique(bin_centers[:,0])[1] - np.unique(bin_centers[:,0])[0]

    def find_bin_id(coord_id):
        """
        Input: (G, 8, [r_id,theta_id,phi_id])
        Output: (G, 8) = bin_id
        """
        theta_id = coord_id[:,1]
        theta_factor = (theta_id > 0)*1 + ((theta_id - 1) > 0)*(theta_id - 1)*N_phi
        phi_factor = (1 - (theta_id == 0))*(1 - (theta_id == (N_theta-1)))*coord_id[:,2]
        return coord_id[:,0]*(N_phi*(N_theta-2) + 2) + theta_factor + phi_factor
    
    #convert q+G to spherical coords
    qG_sph = cartesian_to_spherical(q + G_vectors)

    #find bin index and weights simultaneously
    r_n = np.round((qG_sph[:,0] - q_min)/dq - 0.5, n)
    r_l = np.floor(r_n).astype(np.int32)
    r_l[r_l < 0] = 0
    r_g = np.ceil(r_n).astype(np.int32)
    w_r_l = 1 - r_n % 1
    w_r_g = 1 - w_r_l #if r_l=r_g, only one weight=1, otherwise we're double-counting

    d_phi = 2*np.pi/N_phi
    phi_n = np.round((qG_sph[:,2] + np.pi)/d_phi, n)
    phi_l = np.floor(phi_n).astype(np.int32)
    phi_g = np.ceil(phi_n).astype(np.int32)
    w_phi_l = 1 - phi_n % 1
    w_phi_g = 1 - w_phi_l
    phi_g[phi_g == N_phi] = 0 #maps pi to -pi

    #determine closest cos(theta)
    bin_costheta = np.cos(np.unique(bin_centers[:,1])) #unique cos(theta) in bins
    qG_cos = np.cos(qG_sph[:,1])
    theta_l = np.sum(np.round(qG_cos[:,None] - bin_costheta, n) <= 0, axis=1) - 1
    theta_g = N_theta - np.sum(np.round(qG_cos[:,None] - bin_costheta, n) >= 0, axis=1)

    #weight according to cos(theta)
    bin_diff = bin_costheta[theta_l] - bin_costheta[theta_g]
    w_theta_l = 1 - (bin_costheta[theta_l] - qG_cos)/bin_diff
    w_theta_g = 1 - (qG_cos - bin_costheta[theta_g])/bin_diff

    #fix weights for theta_l = theta_g
    w_theta_l[np.round(bin_diff, n) == 0] = 1
    w_theta_g[np.round(bin_diff, n) == 0] = 1

    all_closest_bins_id = find_bin_id(np.stack((np.ravel(np.stack((r_l,r_l,r_l,r_l,r_g,r_g,r_g,r_g), axis=1)),  np.ravel(np.stack((theta_l,theta_l,theta_g,theta_g,theta_l,theta_l,theta_g,theta_g), axis=1)),  np.ravel(np.stack((phi_l,phi_g,phi_l,phi_g,phi_l,phi_g,phi_l,phi_g), axis=1))), axis=1)) #(G*8,) (each group of 8 elements corresponds to one G vector)

    all_weights = np.ravel(np.stack((w_r_l,w_r_l,w_r_l,w_r_l,w_r_g,w_r_g,w_r_g,w_r_g), axis=1)) * np.ravel(np.stack((w_theta_l,w_theta_l,w_theta_g,w_theta_g,w_theta_l,w_theta_l,w_theta_g,w_theta_g), axis=1)) * np.ravel(np.stack((w_phi_l,w_phi_g,w_phi_l,w_phi_g,w_phi_l,w_phi_g,w_phi_l,w_phi_g), axis=1))

    for i, bin_id in enumerate(all_closest_bins_id):
        w = all_weights[i]
        tot_bin_weights[bin_id] += w
        tot_bin_eps[bin_id] += w*eps_q[i//8] #i//8 is G-vector index
 
    return tot_bin_eps, tot_bin_weights

def bin_eps_q_1d(q, G_vectors, eps_q, bin_centers, tot_bin_eps, tot_bin_weights):
    """
    Binning for epsilon calculated along one direction

    Inputs:
        bin_centers: (N_bins, ): (r, )
        eps_q: (G,E)
        tot_bin_eps: (N_bins,E): total w*eps for all q+G calculated so far
        tot_bin_weights: (N_bins) weights should be same for all energies 
    Outputs:
        updated tot_bin_eps and tot_bin_weights
    """
    q_min = np.min(bin_centers)
    dq = np.unique(bin_centers)[1] - np.unique(bin_centers)[0]
    
    #convert q+G to spherical coords
    qG_sph = cartesian_to_spherical(q + G_vectors)

    #find bin index and weights simultaneously
    r_n = np.round((qG_sph[:,0] - q_min)/dq - 0.5, n)
    r_l = np.floor(r_n).astype(np.int32)
    r_l[r_l < 0] = 0
    r_g = np.ceil(r_n).astype(np.int32)
    w_r_l = 1 - r_n % 1
    w_r_g = 1 - w_r_l #if r_l=r_g, only one weight=1, otherwise we're double-counting

    all_closest_bins_id = np.concatenate([[r_l[i], r_g[i]] for i in range(len(r_l))]) #(G*2,) (each group of 2 elements corresponds to one G vector)
    all_weights = np.concatenate([[w_r_l[i], w_r_g[i]] for i in range(len(r_l))])

    for i, bin_id in enumerate(all_closest_bins_id):
        w = all_weights[i]
        tot_bin_weights[bin_id] += w
        tot_bin_eps[bin_id] += w*eps_q[i//2] #i//2 is G-vector index
 
    return tot_bin_eps, tot_bin_weights
