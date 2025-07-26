import numpy as np
import scipy as sp
import h5py

from dielectric_pyscf.binning import cartesian_to_spherical

# Global constants
lightSpeed     = 299792.458        # km/s
alpha          = 1.0/137.03599908  # EM fine-structure constant at low energy
m_e            = 5.1099894e5       # eV           
kg             = 5.609588e35       # eV per kilogram
pi = np.pi

# Unit conversions
amu2eV         = 9.315e8 
cmInv2eV       = 50677.3093773
BohrInv2eV     = alpha*m_e
Ryd2eV         = 0.5*m_e*alpha**2
cm2sec         = 1/lightSpeed*1e-5                              
sec2yr         = 1/(60.*60.*24*365.25) 

"""Dark Matter astrophysical parameters"""
default_astro = {
    'v0': 238., # km/s
    'vEarth': 250.2, # km/s
    'vEscape': 544.0, # km/s
    'rhoX': 0.3e9, # eV/cm^3
    'sigma_e': 1e-39#1e-37 # reference cross section, cm^2
}
old_astro = {
    'v0': 230.,
    'vEarth': 240.,
    'vEscape': 600.0,
    'rhoX': 0.4e9,
    'sigma_e': 1e-39
}

"""Thomas-Fermi Screening Parameters"""
default_si = {
    'DoScreen': True,
    'eps0': 11.3,       # unitless
    'qTF': 4.13e3,      # eV
    'omegaP':  16.6,    # eV
    'alphaS': 1.563,    # unitless
}
default_ge = {
    'DoScreen': True,
    'eps0': 14.0,       # unitless
    'qTF': 3.99e3,      # eV
    'omegaP': 15.2,     # eV
    'alphaS': 1.563     # unitless
}
default_no_sreen = {
    'DoScreen': False
}
default_screening = default_no_sreen


class df: # Dielectric function class
    def __init__(self, filename=None, eps=None, q=None, E=None, M_cell=None, V_cell=None, dE=None, binned_eps=None, bin_centers=None):
        if filename is None:
            # If filename is not specified, the user can build a df instance without an hdf5 file
            self.eps = eps
            self.q = q
            self.E = E 
            self.M_cell = M_cell
            self.V_cell = V_cell 
            self.dE = dE
            self.binned_eps = binned_eps
            self.bin_centers = bin_centers
        elif filename is not None:
            # If filename is specified, the object is built from the hdf5 file and no parameters besides filename need to be specified
            h5 = h5py.File(filename, 'r')
            self.h5 = h5
            self.eps = h5['epsilon'][:]
            self.q = h5['q'][:] # Momentum in units of alpha*m_e
            self.E = h5['E'][:] # Energy in eV

            # Parameters required for DM rates calculations
            self.M_cell = h5.attrs['M_cell'] # Mass in units of eV
            self.V_cell = h5.attrs['V_cell'] # Unit cell volume in units of (alpha*m_e)^-3
            self.dE = float(h5.attrs['dE']) # Size of energy bins in eV
            # Other parameters can be accessed through self.h5.attrs.items()

            if 'binned_eps' in h5.keys(): # for anisotropic crystal
                self.binned_eps = h5['binned_eps'][:]
                self.bin_centers = h5['bin_centers'][:]
            h5.close() 

    def save_as_hdf5(self, savename):
        # If object was built from existing data, it can be saved as an hdf5 file for later use
        h5 = h5py.File(savename, 'w')
        h5.create_dataset('epsilon', data=self.eps)
        h5.create_dataset('q', data=self.q)
        h5.create_dataset('E', data=self.E)
        h5.attrs['M_cell'] = self.M_cell
        h5.attrs['V_cell'] = self.V_cell
        h5.attrs['dE'] = self.dE
        h5.close()

    def elf(self):
        # Energy loss function
        elf = np.imag(self.eps) / ((np.imag(self.eps))**2 + np.real(self.eps)**2)
        return elf
    
    def S(self):
        # Dynamic structure factor
        S = self.elf() * self.q[:,None]**2 / (2*np.pi*alpha)
        return S
    
    def elf_anisotropic(self):
        # Energy loss function
        elf = np.imag(self.binned_eps) / ((np.imag(self.binned_eps))**2 + np.real(self.binned_eps)**2)
        return elf
    
    def S_anisotropic(self):
        # Dynamic structure factor
        q = self.bin_centers[:,0]
        S = self.elf_anisotropic() * q[:,None]**2 / (2*np.pi*alpha)
        return S

#load angular averaged dielectric function
def load_epsilon(filename):
    """
    Loads angular averaged dielectric function. filename should be the location of epsilon.hdf5 file. 
    """
    epsilon = df(filename)
    return epsilon

def get_elf(eps):
    return np.imag(eps)/((np.imag(eps))**2+np.real(eps)**2)

def get_F_DM(q, m_V=0):
    """
    Return the dark matter form factor F_DM for a given mediator mass and momentum
    Inputs:
        q: (N_q,) In units of ame
        m_V: In units of eV
    """
    F_DM = (m_V**2 + (alpha*m_e)**2) / (m_V**2 + (alpha*m_e*q)**2)
    return F_DM #(q)

def get_eta_MB(q, E, m_X, astro_model=default_astro):   # In units of c^-1
    """
    Calculates the integrated Maxwell-Boltzmann distribution eta(v_{min}(q,E)).
    Inputs:
        q:             (N_q,) np.ndarray: |q| float in ame
        E:             (N_E,) np.ndarray: float
        mX:            float, dark matter mass in eV
        astro_model:   dict containing astrophysical DM parameters
    Output:
        eta:           (N_q, N_E) np.ndarray
    """
    vEscape   = astro_model['vEscape']/lightSpeed
    vEarth    = astro_model['vEarth']/lightSpeed
    v0        = astro_model['v0']/lightSpeed

    q = alpha*m_e * q # converting q to eV

    N_q = q.shape[0]
    N_E = E.shape[0]

    val = np.empty((N_q,N_E), dtype='float')
    for i_q in range(N_q):
        for i_E in range(N_E):
            vMin = q[i_q]/2.0/m_X + E[i_E]/q[i_q]

            if (vMin < vEscape - vEarth):
                val[i_q,i_E] = -4.0*vEarth*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf((vMin+vEarth)/v0) - sp.special.erf((vMin - vEarth)/v0))
            elif (vMin < vEscape + vEarth):
                val[i_q,i_E] = -2.0*(vEarth+vEscape-vMin)*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf(vEscape/v0) - sp.special.erf((vMin - vEarth)/v0))
            else:
                val[i_q,i_E] = 0.0
     
    K = (v0**3) * (-2.0*pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (pi**1.5)*sp.special.erf(vEscape/v0))
    eta_MB = (v0**2) * pi / (2.0*vEarth*K) * val

    return eta_MB #(q,E)

def momentum_integrand(epsilon, m_X, mediator, astro_model, velocity_dist):
    q = epsilon.q
    E = epsilon.E
    #F_DM = get_F_DM(q, m_V)
    if mediator == 'light':
        F_DM = 1 / q**2
    elif mediator == 'heavy':
        F_DM = np.ones_like(q)
    else:
        raise(ValueError('mediator must be set to "light" or "heavy" to determine form of F_DM'))
     
    if velocity_dist == 'MB':
        integrand = q[:,None] * F_DM[:,None]**2 * epsilon.S() * get_eta_MB(q, E, m_X, astro_model) #(q,E)
    elif velocity_dist == 1:
        integrand = q[:,None] * F_DM[:,None]**2 * epsilon.S()  
    return integrand

def get_dR_dE(epsilon, m_X, mediator, astro_model, screening, velocity_dist):
    """
    Returns differential rate with respect to energy. This can be used to calculate the total rate or the rate per ionization.
    Inputs:
        eps:    hdf5 object
        m_X:
        m_V:
        astro_model:
    """
    rho_X = astro_model['rhoX']
    cross_section = astro_model['sigma_e']

    rho_T = epsilon.M_cell / kg / epsilon.V_cell # density of target in units of kg/bohr^3
    reduced_mass = m_X * m_e /(m_X + m_e)

    prefactor = (1/rho_T) * (rho_X/m_X)  * (cross_section/reduced_mass**2) / (4*np.pi)

    integrand = momentum_integrand(epsilon, m_X, mediator, astro_model, velocity_dist)

    # Option to implement different screening
    if screening in ['RPA', 'TF', 'Lindhard', None]:
        if screening in ['TF', 'Lindhard', None]:
            q, E = epsilon.q*alpha*m_e, epsilon.E
            if screening == 'TF':
                eps_screening = ThomasFermi(E, q)
            elif screening == 'Lindhard':
                eps_screening = Lindhard(E, q, 0.1)
            elif screening == None:
                eps_screening = np.ones((q.shape[0], E.shape[0]), dtype='complex')
            integrand = integrand * np.abs(epsilon.eps)**2 / np.abs(eps_screening)**2 # Replacing RPA screening
    else:
        print('Warning: invalid screening specified. Screening must be "RPA", "TF", "Lindhard", or None. Any other inputs will result in the default RPA screening')

    q = epsilon.q # momentum in units of bohr^-1
    E  = epsilon.E # energy in units of eV

    dR_dE = np.empty(E.shape[0], dtype='float')
    for i_E in range(E.shape[0]):
        dR_dE[i_E] = sp.integrate.simpson(integrand[:,i_E], q)
    dR_dE = prefactor * dR_dE * alpha*m_e  / cm2sec / sec2yr

    return dR_dE, E #(E,)

def rate(epsilon, m_X, astro_model, mediator='light', screening='RPA', velocity_dist='MB'):
    """
    Returns total rate by integrating dR/dE over energy
    """
    dR_dE, E = get_dR_dE(epsilon, m_X, mediator, astro_model, screening, velocity_dist)

    R = sp.integrate.simpson(dR_dE, E) #need to multiply by probability from RK secondary ionization
    return R

def d_rate_RamanathanQ(epsilon, m_X, ionization_file, mediator, astro_model=default_astro, screening='RPA', velocity_dist='MB'):

    dR_dE, E = get_dR_dE(epsilon, m_X, mediator=mediator, astro_model=astro_model, screening=screening, velocity_dist=velocity_dist)
     
    ionization_inp = np.genfromtxt(ionization_file).transpose()
    E_ionization = ionization_inp[0]
    pair_creation_prob = ionization_inp[1:]
    E_min, E_max = E_ionization.min(), E_ionization.max()
    #print('Input file has probabilities listed for {:.2f} eV <= E <= {:.2f} eV.\nAll rates outside this range will be ignored.'.format(E_min, E_max))

    E = np.round(E, 5)
    E, dR_dE = E[(E >= E_min) & (E <= E_max)], dR_dE[(E >= E_min) & (E <= E_max)]

    dE = epsilon.dE
  
    N_max_pairs = pair_creation_prob.shape[0]
    R_Q = np.zeros(N_max_pairs, dtype='float')
    for n in range(N_max_pairs):
        R_Q[n] = np.sum(dR_dE*dE*np.interp(E, E_ionization, pair_creation_prob[n]))

    return R_Q

def crystal_form_factor2_epsilon(epsilon):
    """
    Calculates the squared crystal form factor from Im(eps) using eq. 16 in darkELF paper
    """
    eps = epsilon.eps
    V_cell = epsilon.V_cell / (alpha * m_e)**3
    q = epsilon.q*alpha*m_e

    crystal_form_factor2 = q[:,None]**5 * V_cell / 8 / pi**2 * np.imag(eps) / (alpha * m_e)**2 # If q in eV, this should be / (alpha * m_e)**2

    return crystal_form_factor2 #(q,E)


#exclusion plot
def ex(epsilon, mediator, astro_model=default_astro, screening='RPA', velocity_dist='MB'):
    simga_e_0 = astro_model['sigma_e']

    m_X_list = np.logspace(5, 9, 100)
    sigma_e = np.empty_like(m_X_list)
    for i, m_X in enumerate(m_X_list):
        sigma_e[i] = 2.3*simga_e_0 / rate(epsilon, m_X, astro_model=astro_model, mediator=mediator, screening=screening, velocity_dist=velocity_dist)
    return m_X_list, sigma_e


#Thomas-Fermi model for Si
def ThomasFermi(E, q):
    E_mesh, q_mesh = np.meshgrid(E, q)

    eps0 = 11.3
    tau = 1.563 
    E_plasmon = 16.6 #eV
    q_TF = 4.13e3 #eV

    #q_mesh = q_mesh*alpha*m_e #convert q from ame to eV

    eps = 1 + ( 1/(eps0-1) + tau*(q_mesh/q_TF)**2 + q_mesh**4/(4*m_e**2*E_plasmon**2) - (E_mesh/E_plasmon)**2 )**(-1)
    return eps

#Lindhard model for Si
def Lindhard(E, q, fp):
    VCell = 5.209e-9
    nValence = 8
    MCell = 52322355000.0
    mElectron = 5.1099894e5
    alpha = 1.0/137.03599908

    E_mesh, q_mesh = np.meshgrid(E, q)
    #q_mesh = q_mesh*alpha*mElectron #convert q from ame to eV

    def plog(x):
        return np.log(np.abs(x)) + 1j*np.angle(x)
    ne = nValence/VCell
    kF = (3*np.pi**2*ne)**(1./3.)
    omp = np.sqrt(4*np.pi*alpha*ne/mElectron)
    vF = kF/mElectron
    Gp = fp*omp
    Qp = q_mesh/(2*kF) + (E_mesh + 1j*Gp)/(q_mesh*vF)
    Qm = q_mesh/(2*kF) - (E_mesh + 1j*Gp)/(q_mesh*vF)
    factor1 = 3*(omp**2)/(q_mesh**2)/(vF**2)
    factor2 = 0.5 + kF/(4*q_mesh)*(1-Qm**2)*plog((Qm+1)/(Qm-1)) + kF/(4*q_mesh)*(1-Qp**2)*plog((Qp+1)/(Qp-1))
    return 1 + factor1*factor2

def integrate_3d(bin_centers, integrand):
    """
    bin_centers in spherical coords
    """
    q = np.unique(bin_centers[:,0])
    theta = np.unique(bin_centers[:,1])
    cos_theta = np.cos(theta)
    phi = np.unique(bin_centers[:,2])

    N_q = q.shape[0]
    N_th = theta.shape[0]
    N_phi = phi.shape[0]

    vol_phi = 2*np.pi/N_phi

    I = np.zeros(integrand.shape[1])
    for i, c in enumerate(bin_centers):
        q_n = np.where(c[0] == q)[0][0]
        if q_n == 0:
            d_q = q[q_n]**3 / 3
        elif q_n == N_q-1:
            d_q = 0
        else:           
            d_q = ((q[q_n] + q[q_n+1])**3 - (q[q_n] + q[q_n-1])**3) / 24

        th_n = np.where(c[1] == theta)[0][0]
        if th_n == 0: #theta = 0
            d_th = 0.5*(cos_theta[0] - cos_theta[1])
            d_phi = 2*np.pi
        elif th_n == N_th-1: #theta = pi
            d_th = 0.5*(cos_theta[th_n-1] - cos_theta[th_n])
            d_phi = 2*np.pi
        else:
            d_th = 0.5*(cos_theta[th_n-1] + cos_theta[th_n+1]) - cos_theta[th_n]
            d_phi = vol_phi

        d_vol = d_q*d_th*d_phi

        I  = I + d_vol*integrand[i]

    return 4*np.pi*I # where am I missing this 4pi factor????

### Anisotropic dielectric function ###

def get_eta_MB_anisotropic(bin_centers, E, m_X, astro_model, v_earth_dir):   # In units of c^-1
    """
    Calculates the integrated Maxwell-Boltzmann distribution eta(v_{min}(q,E)).
    Inputs:
        q:             (N_q,) np.ndarray: |q| float in ame
        E:             (N_E,) np.ndarray: float
        mX:            float, dark matter mass in eV
        astro_model:   dict containing astrophysical DM parameters
    Output:
        eta:           (N_q, N_E) np.ndarray
    """
    vEscape   = astro_model['vEscape']/lightSpeed
    v0        = astro_model['v0']/lightSpeed
    
    v_earth_dir_sph = cartesian_to_spherical(v_earth_dir)
    vEarth    = np.array([astro_model['vEarth']/lightSpeed, v_earth_dir_sph[1], v_earth_dir_sph[2]])

    q = alpha*m_e * bin_centers[:,0] # converting q to eV
    th = bin_centers[:,1]
    phi = bin_centers[:,2]

    K = (v0**3) * (-2.0*pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (pi**1.5)*sp.special.erf(vEscape/v0)) # normalization

    q_dot_vEarth = vEarth[0]*q * np.sin(vEarth[1]*np.sin(th) * np.cos(vEarth[2] - phi) + np.cos(vEarth[1])*np.cos(th))

    v_min = (1/q)[:,None] * np.abs(q_dot_vEarth[:,None] + (q**2/(2*m_X))[:,None] + E[None,:])
    mask = v_min < vEscape

    g_MB = np.pi*v0**2 / (K * q[:,None]) * (np.exp(-(v_min/v0)**2) - np.exp(-(vEscape/v0)**2)) * mask

    return g_MB #(bin,E)

def momentum_integrand_anisotropic(epsilon, m_X, mediator, astro_model, velocity_dist, v_earth_dir):
    q = epsilon.bin_centers[:,0]
    E = epsilon.E
    #F_DM = get_F_DM(q, m_V)
    if mediator == 'light':
        F_DM = 1 / q**2
    elif mediator == 'heavy':
        F_DM = np.ones_like(q)
    else:
        raise(ValueError('mediator must be set to "light" or "heavy" to determine form of F_DM'))
     
    if velocity_dist == 'MB':
        integrand = q[:,None] * F_DM[:,None]**2 * epsilon.S_anisotropic() * get_eta_MB_anisotropic(epsilon.bin_centers, E, m_X, astro_model, v_earth_dir) #(bin,E)
    elif velocity_dist == 1:
        integrand = q[:,None] * F_DM[:,None]**2 * epsilon.S_anisotropic()  
    return integrand #(bin,E)

def get_dR_dE_anisotropic(epsilon, m_X, mediator, astro_model, screening, velocity_dist, v_earth_dir):
    """
    Returns differential rate with respect to energy. This can be used to calculate the total rate or the rate per ionization.
    Inputs:
        eps:    hdf5 object
        m_X:
        m_V:
        astro_model:
    """
    rho_X = astro_model['rhoX']
    cross_section = astro_model['sigma_e']

    rho_T = epsilon.M_cell / kg / epsilon.V_cell # density of target in units of kg/bohr^3
    reduced_mass = m_X * m_e /(m_X + m_e)

    prefactor = (1/rho_T) * (rho_X/m_X)  * (cross_section/reduced_mass**2) / (4*np.pi)

    integrand = momentum_integrand_anisotropic(epsilon, m_X, mediator, astro_model, velocity_dist, v_earth_dir)

    if screening != 'RPA':
        raise NotImplementedError('Non-RPA screening not implemented for anisotropic calculations')
    '''
    # Option to implement different screening
    if screening in ['RPA', 'TF', 'Lindhard', None]:
        if screening in ['TF', 'Lindhard', None]:
            q, E = epsilon.q*alpha*m_e, epsilon.E
            if screening == 'TF':
                eps_screening = ThomasFermi(E, q)
            elif screening == 'Lindhard':
                eps_screening = Lindhard(E, q, 0.1)
            elif screening == None:
                eps_screening = np.ones((q.shape[0], E.shape[0]), dtype='complex')
            integrand = integrand * np.abs(epsilon.eps)**2 / np.abs(eps_screening)**2 # Replacing RPA screening
    else:
        print('Warning: invalid screening specified. Screening must be "RPA", "TF", "Lindhard", or None. Any other inputs will result in the default RPA screening')
    '''
    E  = epsilon.E # energy in units of eV

    dR_dE = integrate_3d(epsilon.bin_centers, integrand)
    dR_dE = prefactor * dR_dE * alpha*m_e  / cm2sec / sec2yr

    return dR_dE, E #(E,)