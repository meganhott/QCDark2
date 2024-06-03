from routines import *

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

cell = build_cell_from_input()
#kpts = get_kpts(cell)