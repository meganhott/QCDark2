from routines import *
import matplotlib.pyplot as plt
import spglib as spg

def spherical_to_cartesian(sph: np.ndarray) -> np.ndarray:
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
    return np.transpose([x, y, z])

def construct_theta_bins(N_theta: int = 7, N_phi: int = 8) -> np.ndarray:
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
    N = 2 + (N_theta - 2)*N_phi                             # Note: we begin counting from 0 in our bin_edges, so to conserve shape we must subtract 1. 
    b_i = np.append([1], 1 - 2/N*(1 + i*N_phi))
    b_i = np.append(b_i, [-1])
    return np.arccos(b_i)

def construct_all_solid_angles(N_theta: int = 7, N_phi: int = 8) -> np.ndarray:
    """
    Construct points in theta and phi, such that the integral over solid angles is well-approximated by a trapezoidal rule integration law in theta.
    """
    if type(N_theta) != int or type(N_phi) != int:
        logging.info('Raising exception, in input_parameters.py, N_theta and N_phi must be of the type int.')
        raise Exception('In input_parameters.py, N_theta and N_phi must be of the type int.')
    if not N_theta%2:
        logging.info('! WARNING: Given N_theta = {} is even and will not contain points in the x-y plane.\nThe accuracy of the dielectric function will remain unaffected.'.format(N_theta))
    theta_bins = construct_theta_bins(N_theta, N_phi)
    phi_bins = np.arange(0, 1., 1./N_phi)*2*np.pi
    solid_angles = []
    for theta in theta_bins:
        if theta == 0 or round(theta, 9) == round(np.pi, 9):
            solid_angles.append([theta, 0.])
        else:
            for phi in phi_bins:
                solid_angles.append([theta, phi])
    return np.array(solid_angles)

def gen_q_vectors(cartesian=True) -> np.ndarray:
    Omega = construct_all_solid_angles(parmt.N_theta, parmt.N_phi)
    qr = np.arange(parmt.dq*0.5, parmt.q_max + parmt.dq*0.5, parmt.dq)
    qra = []
    for q in qr:
        for O in Omega:
            qra.append([q, O[0], O[1]])
    if cartesian: #Convert from spherical to cartesian
        qra = spherical_to_cartesian(np.array(qra))
    return np.array(qra)

def get_1BZ_q_vectors(q:np.ndarray, mod='G'):
    """
    Parallel version of get_eq_1BZ_kpoint for array of cartesion q vectors. 
    To do:
    - Output array of unique q_1BZ vectors: Even when rounded to 5 decimal places to account for floating point errors, there seem to be no repeated q_1BZ vectors generated 
    - Output indices i_q, i_G to map q to q_1BZ[i_q] + G[i_G] where G are all unique G vectors
    - Also output all unique G vectors? - There are repeated G vectors
    - Add error message for mod != G or k
    """
    #add cell input and rounding precision later
    G = cell.reciprocal_vectors()
    D = np.linalg.inv(G.T)

    q_D = np.tensordot(D, q, axes=(1,1)).T + 0.5 #need to add 1/2 since 1BZ is defined for -1/2G < k < 1/2G

    if mod == 'k':
        k_grid = np.array(parmt.ik_grid)
        m_G = (q_D*k_grid // 1)/k_grid #need to do this for floating-point errors
        q_1BZ_D = (q_D*k_grid % 1)/k_grid - 0.5 
    elif mod == 'G':
        m_G = q_D // 1 
        q_1BZ_D = (q_D % 1) - 0.5 #shift back by 1/2 
    else:
        print('error')
    
    q_1BZ = np.round(np.tensordot(G, q_1BZ_D, axes=(0,1)).T, 5)

    """
    Old code for reference
    q_D = np.round(D@q + 0.5, 5) #need to add 1/2 since 1BZ is defined for -1/2G < k < 1/2G
    m_G = q_D // 1 #can return -0. instead of 0. - may be an issue
    k_D = (q_D % 1) - 0.5 #shift back by 1/2
    k = G.T @ k_D
    """
    return q_1BZ, m_G

def get_q_plus_k_vectors(q:np.ndarray, k:np.ndarray):
    """
    For an array of shape (n, 3) of q vectors and another array of shape (m, 3) of k vectors, returns all unique q+k vectors. Currently there seem to be no identical q+k vectors generated when rounded to 5 decimal places.
    """
    qk = get_all_unique_vectors_in_array(np.reshape(q[:,np.newaxis,:]+k[np.newaxis,:,:], (q.shape[0]*k.shape[0],3)), round_to=5)
    return qk            

def get_IBZ_q_vectors(q:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs: (n,3) array of q-vectors in 1BZ
    Outputs: array of q-vectors in IBZ, indices to map 1BZ q to IBZ q

    Test by inputting k grid
        - pyscf has 8 (29) unique IBZ points for 4x4x4 (8x8x8) k-grid, while this results in 11 (50) unique IBZ q

    To Do:
        - Add option to use different point group? - could get from pyscf cell.kpts.groupname
        - This method is probably slow because we loop over all q - can this loop be done using e.g. tensordot instead?
        - Another method would be to select all q in the irreducible wedge defined by the high-symmetry points as q_IBZ and then just have to figure out i_q. Not sure if this will work because we don't have Wigner-Seitz cell and 3D shape of wedge is complicated
    """
    #Getting rotation operations for point group
    rots = spg.get_symmetry_from_database(525)['rotations'] #525 has point group m-3m for fcc/bcc cell
    
    q_IBZ = np.array([[np.nan,np.nan,np.nan]]) #initialize q_IBZ
    i_q = np.empty(q.shape[0], dtype=int) #index to q_IBZ for each q
    for j, qi in enumerate(q):
        #apply all rotations to qi
        qi_rot = np.round(np.tensordot(rots, qi, axes=1),5) #(48,3)
        for i, q_IBZi in enumerate(q_IBZ):
            if any(np.equal(q_IBZi, qi_rot).all(1)): #if any vector in q_IBZ equals transformed q, set i_q to be i
                i_q[j] = i - 1 #-1 since we get rid of first placeholder q_IBZ
                break #go to next q
        else: #otherwise, qi is a new IBZ point
            q_IBZ = np.vstack([q_IBZ, np.round(qi,5)])   
            i_q[j] = q_IBZ.shape[0]-2 #need -2 since we are getting rid of first q_IBZ and indexing starts at 0
    return q_IBZ[1:], i_q

def nearest_kq(q, tolerance, q_far_min=3): #tolerance=1/max(parmt.ik_grid)**2):
    """
    Test function for reducing nscf computations for large q when integrating
    q should be in spherical coordinates
    To Do:
    - What is best tolerance?
    - determine rtol for np.isclose/allclose
    """
    q_far = q[q[:,0] > q_far_min] #Only want to do this approximation for large q
    q_far = spherical_to_cartesian(q_far)
    q_far_modk, m_Gk = get_1BZ_q_vectors(q_far, mod='k')

    #compare g_far_modk to determine which are close enough to treat as equivalent
    q_far_modk_unique = np.array([[np.nan,np.nan,np.nan]]) #initialize
    i_modk_q = np.empty(q_far.shape[0], dtype=int) #index to q_IBZ for each q

    for j, qi in enumerate(q_far_modk):
        for l, qu in enumerate(q_far_modk_unique):
            if np.allclose(qi, qu, atol=tolerance): #what to pick for rtol?
                i_modk_q[j] = l - 1
                break #go to next q_far_modk
        else: #otherwise qi is a new unique point    
            q_far_modk_unique = np.vstack([q_far_modk_unique, np.round(qi,5)])
            i_modk_q[j] = q_far_modk_unique.shape[0]-2

    return q_far, q_far_modk, m_Gk, q_far_modk_unique[1:], i_modk_q

def test_nearest_kq(plot=False):
    cell = build_cell_from_input()
    G = cell.reciprocal_vectors()

    q = gen_q_vectors(cartesian=False)
    q_far, q_far_modk, m_Gk, q_far_modk_unique, i_modk_q = nearest_kq(q, parmt.dq/np.linalg.norm(G[0])) #tolerance is dq
    print(f'q_far: {q_far.shape[0]}, unique q_far mod k: {q_far_modk_unique.shape[0]} for q > 3ame')

    q_far, q_far_modk, m_Gk, q_far_modk_unique, i_modk_q = nearest_kq(q, parmt.dq/np.linalg.norm(G[0]),q_far_min=0)
    print(f'q_far: {q_far.shape[0]}, unique q_far mod k: {q_far_modk_unique.shape[0]} for all q')
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(q_far.T[0],q_far.T[1],q_far.T[2]) #original points

        q_far_approx = q_far_modk_unique[i_modk_q] + np.tensordot(G,m_Gk,(0,1)).T

        ax.scatter(q_far_approx.T[0],q_far_approx.T[1],q_far_approx.T[2]) #approximate points

        #ax.scatter(q_far_modk_unique.T[0],q_far_modk_unique.T[1],q_far_modk_unique.T[2], alpha = 0.9)
        plt.show()
    
    #return q, q_far, q_far_modk_unique, i_modk_q

cell = build_cell_from_input()

def get_1BZ_testing_plots():
    #Make sure dq is large and k-grid is small for these plots!! 
    q = gen_q_vectors() #cartesian q vectors
    k = make_kpts(cell).kpts

    q_1BZ, m_G = get_1BZ_q_vectors(q)
    G = cell.reciprocal_vectors()

    #Plotting q and q_1BZ
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(q.T[0],q.T[1],q.T[2], s=1)
    ax1.set_title('q')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.set_title('q in 1BZ')
    ax2.scatter(q_1BZ.T[0],q_1BZ.T[1],q_1BZ.T[2], s=1)

    #Verifying we can transform back to q from q_1BZ
    q_re = q_1BZ + np.tensordot(G,m_G,(0,1)).T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q.T[0],q.T[1],q.T[2])
    ax.scatter(q_re.T[0],q_re.T[1],q_re.T[2])

    #Testing mod='k'
    q_1BZ_modk, m_G_modk = get_1BZ_q_vectors(q,mod='k')
    q_re = q_1BZ_modk + np.tensordot(G,m_G_modk,(0,1)).T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q.T[0],q.T[1],q.T[2])
    ax.scatter(q_re.T[0],q_re.T[1],q_re.T[2])

    #Plotting q+k and (q+k)_1BZ
    """
    qk = get_q_plus_k_vectors(q,k)
    qk_1BZ = get_1BZ_q_vectors(qk)[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(qk.T[0],qk.T[1],qk.T[2], s=1)
    ax1.set_title('q+k')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.scatter(qk_1BZ.T[0],qk_1BZ.T[1],qk_1BZ.T[2], s=0.5)
    ax2.set_title('q+k in 1BZ')
    """
    plt.show()

def get_IBZ_testing_plots():
    k = make_kpts(cell).kpts
    G = cell.reciprocal_vectors()
    """
    k_IBZ, i_k = get_IBZ_q_vectors(k)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.scatter(k.T[0],k.T[1],k.T[2], alpha=0.2)
    ax1.scatter(k_IBZ.T[0],k_IBZ.T[1],k_IBZ.T[2], alpha=0.8)
    
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    for i,k_IBZi in enumerate(k_IBZ):
        mask = (i_k == i) 
        ki = k[mask]
        ax2.scatter(ki.T[0],ki.T[1],ki.T[2])

        #individual plots for each set of equivalent 1BZ points
        
        fig_i = plt.figure()
        ax_i = fig_i.add_subplot(projection='3d')
        ax_i.scatter(ki.T[0],ki.T[1],ki.T[2])
        ax_i.set_title(f'i = {i}')
        for j in range(G.shape[0]):
            ax_i.plot([0,G[j][0]],[0,G[j][1]],[0,G[j][2]],linewidth=2,color='k')
    
    for i in range(G.shape[0]):
        ax1.plot([0,G[i][0]],[0,G[i][1]],[0,G[i][2]],linewidth=2,color='k')
        #ax2.plot([0,G[i][0]],[0,G[i][1]],[0,G[i][2]],linewidth=2,color='k')
    """

    #Verifying we can transform back to q from q_1BZ
    q = gen_q_vectors()
    q_1BZ, m_G = get_1BZ_q_vectors(q)
    q_IBZ, i_q = get_IBZ_q_vectors(q)
    q_re = q_IBZ[i_q] + np.tensordot(G,m_G,(0,1)).T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q.T[0],q.T[1],q.T[2])
    ax.scatter(q_re.T[0],q_re.T[1],q_re.T[2])

    plt.show()