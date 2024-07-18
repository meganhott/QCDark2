from routines import *
from cartesian_moments import *
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

def get_eps():
    cell = build_cell_from_input()

    #scf at k_i and nscf at k_f
    kmf = KS_electronic_structure(cell)
    KS_non_self_consistent_field(kmf)
    convert_to_eV_and_scissor(cell) #updates energies

    #load in energies, coefficients, and k-grids
    k1 = np.load(parmt.store + '/k-pts_i.npy')
    k2 = np.load(parmt.store + '/k-pts_f.npy')

    dft_path = parmt.store + '/DFT/'
    mo_en_i = np.load(dft_path + 'mo_en_i.npy') #is it faster to load these in or take from kmf if possible?
    mo_coeff_i = np.load(dft_path + 'mo_coeff_i.npy')
    mo_occ_i = np.load(dft_path + 'mo_occ_i.npy')
    mo_en_f = np.load(dft_path + 'mo_en_f.npy')
    mo_coeff_f = np.load(dft_path + 'mo_coeff_f.npy')
    #mo_occ_f = np.load(dft_path + 'mo_occ_f.npy')

    #generate all 1BZ q vectors (q = k2 - k1)
    q_1BZ_dic = get_1BZ_q_points(cell) #or load from save
    q_1BZ = np.array(list(q_1BZ_dic.keys())) #unique q

    G_vectors = gen_G_vectors(cell)

    #overlap integrals dictionary
    I = get_all_prim_1D_overlap(cell, q_1BZ, G_vectors) #dictionary of all 1D primitive Gaussian overlaps: this will be referenced later by all nodes in parallel calculation

    #generate list of all atomic orbitals (needed for eta)
    all_ao = gen_all_atomic_orbitals(cell, gen_all_1D_prim_gauss(cell))

    def get_3D_overlap(q_vector, G_vector, k_i_id, k_f_id, i, j, all_ao):
        """
        Still need to test!
        Calculates 3D overlap eta = <jk'|exp(i(q+G)r)|ik> using stored 1D overlaps
        Inputs:
            q_vector:   np.ndarray of shape (3,): one q vector in 1BZ
            G_vector:   np.ndarray of shape (3,): one G vector
            k_i_id:     int: index of k_initial
            k_f_id:     int: index of k_final
            i:          int: initial state band
            j:          int: final state band
            all_ao:     list(AO): all atomic orbitals
        Outputs:
            eta:        complex: 3D overlap <jk'|exp(i(q+G)r)|ik>
        """
        def f(q_vector, G_vector, ao_a, ao_b, R_vector):
            """
            Subroutine to calculate 3D primitive Gaussian overlap integral. ao_a and ao_b are atomic orbitals.
            """
            f_sum = 0
            for m in range(ao_a.Nprim):
                for n in range(ao_b.Nprim):
                    f_sum += ao_a.norm[m]*ao_a.coef[m]*ao_b.norm[n]*ao_b.coef[n] * np.prod([I[q_vector[i],G_vector[i],R_vector[i],ao_a.shell[i],ao_b.shell[i],ao_a.exp[m],ao_b.exp[n],ao_a.loc[i],ao_b.loc[i]] for i in range(3)])
            return f_sum

        R_vectors = construct_R_vectors(cell)[0]
        eta = 0
        for a, ao_a in enumerate(all_ao):
            for b, ao_b in enumerate(all_ao):
                eta += np.conjugate(mo_coeff_f[k_f_id, j, b])*mo_coeff_i[k_i_id, i, a] * sum([np.exp(-1j*np.dot(k2[k_f_id], R))*f(q_vector,G_vector,ao_a,ao_b,R) for R in R_vectors])
    
        return eta

    #epsilon(q, E, G, G')
    """
    What is best way to store epsilon?
    dict[q,E] = array(G,G')?
    """
    eps = {}
    for q in q_1BZ: #will parallelize this step
        num_bands = mo_en_i.shape[1]
        num_G_vectors = G_vectors.shape[0]
        #create and store all eta first for given q
        #dictionary: eta_q[(G_id, k_pair_id, i, j)]
        #want to change this so you can input G_id meshgrid
        eta_q = {}
        for G_id,G in enumerate(G_vectors):
            for k_pair_id,k_pair in enumerate(q_1BZ_dic[q]):
                for i in range(num_bands):
                    for j in range(num_bands):
                        tup = tuple([G_id, k_pair_id, i, j])
                        eta_q[tup] = get_3D_overlap(q,G,k_pair[0],k_pair[1],i,j,all_ao)

        eps[q,E] = np.zeros(num_G_vectors,num_G_vectors)
        for E in np.arange(parmt.dE, parmt.E_max+parmt.dE, parmt.dE): #energies - problem at E = 0 so not inlcuded?
            for G_id, G in enumerate(G_vectors):
                for Gp_id, Gp in enumerate(G_vectors):
                    chi = 0
                    for k_pair_id, k_pair in q_1BZ_dic[q]: 
                        for i in range(num_bands):
                            for j in range(num_bands):
                                chi += np.conjugate(eta_q[G_id, k_pair_id, i, j])*eta_q[Gp_id, k_pair_id, i, j] / (mo_en_i[k_pair[0],i] - mo_en_f[k_pair[1],j] + E ) #+i*eta, also multiply by occupancy
                    eps[q,E][G_id,Gp_id] = -(4*np.pi)**2 / np.linalg.norm(q+G) / np.linalg.norm(q+Gp) * chi
            eps[q,E] = eps[q,E] + np.identity(num_G_vectors) #delta_GGp

            #take inverse for LFE
            if parmt.lfe_q_cutoff is not None:
                if type(parmt.lfe_q_cutoff) != float:
                    raise ValueError("Parameter lfe_q_cutoff in input_parameters.py must be either None or of type float.")
                #LFE calculation

def get_all_prim_1D_overlap(cell, q_vectors, G_vectors):
    """
    Inputs: cell, all 1BZ q vectors, all G vectors
    Returns: dictionary of all 1D primitive Gaussian overlaps: prim_1D_overlap_dic = I[q,G,R,l,m,xi_a,xi_b,A,B]

    This stores the overlap integrals differently than cartesian_moments.get_prim_1D_overlap - this one may have fewer elements stored because unique xi_a, xi_b, A, B are found instead of using primitive Gaussians from prim gauss array. Still need to test
    """
    #generate all 1D primitive Gaussians
    primgauss = gen_all_1D_prim_gauss(cell)
    primgauss_indx_arr, atom_locs = gen_prim_gauss_indices(primgauss)

    #get all unique parameters
    q_unique = get_all_unique_nums_in_array(q_vectors, round_to=10) #q_i
    G_unique = get_all_unique_nums_in_array(G_vectors, round_to=10) #G_i
    R_unique = construct_R_vectors(cell)[1] #R_i
    exp_unique = get_all_unique_nums_in_array(primgauss_indx_arr[:,2],round_to=10) #xi_a and xi_b
    atom_locs_unique = get_all_unique_nums_in_array(atom_locs, round_to=10) #A_i and B_i
    l_max = int(np.max(primgauss_indx_arr[:,1])) #l_i,m_i <= l_max

    #might be able to optimize further since it should be equivalent if {A, l, xi_a} <-> {B, m, xi_b}
    prim_1D_overlap_dic = {}
    for xi_a in exp_unique:
        for xi_b in exp_unique:
            p = xi_a + xi_b
            for A in atom_locs_unique:
                for B in atom_locs_unique:
                    E_ijt = get_E_ijt(xi_a,xi_b,l_max,l_max,A-B)
                    for R in R_unique:
                        P = (xi_b*(B+R) + xi_a*A) / (xi_a+xi_b)
                        for l in range(l_max+1):
                            for m in range(l_max+1):
                                for q in q_unique:
                                    for G in G_unique:
                                        tup = tuple([q,G,R,l,m,xi_a,xi_b,A,B])
                                        prim_1D_overlap_dic[tup] = np.sqrt(np.pi/p) * np.exp(1j*(q+G)*P - (q+G)**2/4/p) * sum([E_ijt[l][m][t]*(1j*(q+G))**t for t in range(l+m)])
    #store dic
    return prim_1D_overlap_dic

def test_get_all_prim_1D_overlap():
    cell = build_cell_from_input()
    q = gen_q_vectors(cell)[:10,:] #only testing for a few vectors to reduce size of dictionary
    G = gen_G_vectors(cell)[:10,:]
    I = get_all_prim_1D_overlap(cell, q, G)
    return I
