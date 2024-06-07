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
    
    q_IBZ = np.array([[0.,0.,0.]]) #can have this to initialize q_IBZ since gamma will always be in q? #unique q in IBZ
    i_q = np.empty(q.shape[0], dtype=int) #index to q_IBZ for each q
    for j, qi in enumerate(q):
        #apply all rotations to qi
        qi_rot = np.round(np.tensordot(rots, qi, axes=1),5) #(48,3)
        for i, q_IBZi in enumerate(q_IBZ):
            if any(np.equal(q_IBZi, qi_rot).all(1)): #if any vector in q_IBZ equals transformed q, set i_q to be i
                i_q[j] = i
                break #go to next q
        else: #otherwise, qi is a new IBZ point
            q_IBZ = np.vstack([q_IBZ, np.round(qi,5)])   
            i_q[j] = q_IBZ.shape[0]
    return q_IBZ, i_q

cell = build_cell_from_input()
k = make_kpts(cell).kpts

def get_1BZ_testing_plots():
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

def get_IBZ_testing_plots():
    k = make_kpts(cell).kpts
    G = cell.reciprocal_vectors()
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
        """
        fig_i = plt.figure()
        ax_i = fig_i.add_subplot(projection='3d')
        ax_i.scatter(ki.T[0],ki.T[1],ki.T[2])
        ax_i.set_title(f'i = {i}')
        for j in range(G.shape[0]):
            ax_i.plot([0,G[j][0]],[0,G[j][1]],[0,G[j][2]],linewidth=2,color='k')
        """

    for i in range(G.shape[0]):
        ax1.plot([0,G[i][0]],[0,G[i][1]],[0,G[i][2]],linewidth=2,color='k')
        ax2.plot([0,G[i][0]],[0,G[i][1]],[0,G[i][2]],linewidth=2,color='k')

    plt.show()