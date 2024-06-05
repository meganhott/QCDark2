from routines import *
import matplotlib.pyplot as plt

def spherical_to_cartesian(sph: np.ndarray) -> np.ndarray:
    """
    Convert all vectors given in spherical polar coordinates to corresponding vectors in cartesian coordinates.
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
    cart = np.transpose([x, y, z])
    cart = np.unique(np.round(cart, 10), axis = 0)
    srt = np.argsort(np.linalg.norm(cart, axis = 1))
    return cart[srt]

def cartesian_to_spherical(cart: np.ndarray) -> np.ndarray:
    """
    Convert all vectors in cartesian coordinates to spherical coordinates.
    Inputs:
        cart:   np.ndarray of shape (N, 3)
    Returns:
        sph:    np.ndarray of shape (N, 3)
    """
    xy = cart[:,0]**2 + cart[:,1]**2
    r = np.sqrt(cart[:,2]**2 + xy)
    th = np.arctan2(np.sqrt(xy), cart[:,2])
    ph = np.arctan2(cart[:,1], cart[:,0])
    sph = np.transpose([r, th, ph])
    return sph

def gen_q_vectors() -> tuple[np.ndarray, np.ndarray]:
    """
    NOTE: Add symmetrization: gen all relevant q-vectors, filter to unique q-vectors given SC symmetries:
            rotations in x, y and z, and mirror symmetries about x, y and z.
    Generate relevant q-vectors using parameters from input parameters in spherical coordinates
    and construct cartesian coordinates. 
    Inputs:
        None
    Outputs:
        q_grid: np.ndarray of shape (N, 3)
                q[i] = [q_x, q_y, q_z]
    """
    if np.round(1%parmt.d_theta_q, 10) != 0 or np.round(1%parmt.d_phi_q, 10) != 0:
        raise ValueError("Error in input_parameters.py, d_theta_q and d_phi_q must divide 1 exactly.\nGiven: d_theta_q = {:.5f}, d_phi_q = {:.5f}".format(parmt.d_theta_q, parmt.d_phi_q))
    qr = np.arange(parmt.dq*0.5, parmt.q_max + parmt.dq*0.5, parmt.dq)
    q_theta = np.arange(0, np.pi*(1 + 0.5*parmt.d_theta_q), np.pi*parmt.d_theta_q)
    q_phi = np.arange(0, 2*np.pi, np.pi*parmt.d_phi_q)
    q_grid = np.transpose(np.meshgrid(qr, q_theta, q_phi)).reshape((-1, 3))
    q_grid = spherical_to_cartesian(q_grid)
    return q_grid, cartesian_to_spherical(q_grid)

def get_1BZ_q_vectors(q:np.ndarray):
    """
    Parallel version of get_eq_1BZ_kpoint for array of cartesion q vectors. 
    To do:
    - Output array of unique q_1BZ vectors: Even when rounded to 5 decimal places to account for floating point errors, there seem to be no repeated q_1BZ vectors generated 
    - Output indices i_q, i_G to map q to q_1BZ[i_q] + G[i_G] where G are all unique G vectors
    - Also output all unique G vectors? - There are repeated G vectors
    """
    #add cell input and rounding precision later
    G = cell.reciprocal_vectors()
    D = np.linalg.inv(G.T)

    q_D = np.tensordot(D, q, axes=(1,1)).T + 0.5
    m_G = q_D // 1 #need to add 1/2 since 1BZ is defined for -1/2G < k < 1/2G
    k_D = (q_D % 1) - 0.5 #shift back by 1/2 
    k = np.round(np.tensordot(G.T, k_D, axes=(1,1)).T, 5)

    """
    Old code for reference
    q_D = np.round(D@q + 0.5, 5) #need to add 1/2 since 1BZ is defined for -1/2G < k < 1/2G
    m_G = q_D // 1 #can return -0. instead of 0. - may be an issue
    k_D = (q_D % 1) - 0.5 #shift back by 1/2
    k = G.T @ k_D
    """
    return k, m_G

def get_q_plus_k_vectors(q:np.ndarray, k:np.ndarray):
    """
    For an array of shape (n, 3) of q vectors and another array of shape (m, 3) of k vectors, returns all unique q+k vectors. Currently there seem to be no identical q+k vectors generated when rounded to 5 decimal places.
    """
    qk = get_all_unique_vectors_in_array(np.reshape(q[:,np.newaxis,:]+k[np.newaxis,:,:], (q.shape[0]*k.shape[0],3)), round_to=5)
    return qk            

cell = build_cell_from_input()

def testing_plots():
    #Make sure dq is large and k-grid is small for these plots!! 
    q = gen_q_vectors()[0] #cartesian q vectors
    k = make_kpts(cell).kpts

    #Plotting q and q_1BZ
    q_1BZ = get_1BZ_q_vectors(q)[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(q.T[0],q.T[1],q.T[2], s=1)
    ax1.set_title('q')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.set_title('q in 1BZ')
    ax2.scatter(q_1BZ.T[0],q_1BZ.T[1],q_1BZ.T[2], s=1)

    #Plotting q+k and (q+k)_1BZ
    qk = get_q_plus_k_vectors(q,k)
    qk_1BZ = get_1BZ_q_vectors(qk)[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(qk.T[0],qk.T[1],qk.T[2], s=1)
    ax1.set_title('q+k')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.scatter(qk_1BZ.T[0],qk_1BZ.T[1],qk_1BZ.T[2], s=0.5)
    ax2.set_title('q+k in 1BZ')
    plt.show()