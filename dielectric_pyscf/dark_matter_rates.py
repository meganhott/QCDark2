import numpy as np
import scipy as sp
import h5py

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
    'v0': 238.,
    'vEarth': 250.2,
    'vEscape': 544.0,
    'rhoX': 0.3e9,
    'sigma_e': 1e-39
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
    def __init__(self, filename=None, eps=None, q=None, E=None, M_cell=None, V_cell=None, dE=None):
        if filename is None:
            # If filename is not specified, the user can build a df instance without an hdf5 file
            self.eps = eps
            self.q = q
            self.E = E 
            self.M_cell = M_cell
            self.V_cell = V_cell 
            self.dE = dE
        elif filename is not None:
            # If filename is specified, the object is built from the hdf5 file and no parameters besides filename need to be specified
            h5 = h5py.File(filename, 'r')
            self.h5 = h5
            self.eps = np.array(h5['epsilon'])
            self.q = np.array(h5['q']) # Momentum in units of alpha*m_e
            self.E = np.array(h5['E']) # Energy in eV

            # Parameters required for DM rates calculations
            self.M_cell = h5.attrs['M_cell'] # Mass in units of eV
            self.V_cell = h5.attrs['V_cell'] # Unit cell volume in units of (alpha*m_e)^-3
            self.dE = float(h5.attrs['dE']) # Size of energy bins in eV
            # Other parameters can be accessed through self.h5.attrs.items()

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

def get_eta_MB(q, E, m_X, astro_model):   # In units of c^-1
    """
    Calculates the integrated Maxwell-Boltzmann distribution eta(v_{min}(q,E)).
    Inputs:
        q:             (N_q,) np.ndarray: |q| float
        E:             (N_E,) np.ndarray: float
        mX:            float, dark matter mass in eV
        astro_model:   dict containing astrophysical DM parameters
    Output:
        eta:           (N_q, N_E) np.ndarray
    """
    vEscape   = astro_model['vEscape']/lightSpeed
    vEarth    = astro_model['vEarth']/lightSpeed
    v0        = astro_model['v0']/lightSpeed

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

def momentum_integrand(q, E, cff2, eps_screening, m_X, mediator, astro_model):
    #F_DM = get_F_DM(q, m_V)
    if mediator == 'light':
        F_DM = (alpha*m_e)**2 / q**2
    elif mediator == 'heavy':
        F_DM = np.ones_like(q)
    else:
        raise(ValueError('mediator must be set to "light" or "heavy" to determine form of F_DM'))
     
    eps_screening = 1/(np.real(eps_screening)**2 + np.imag(eps_screening)**2)
     
    integrand = 1 / q[:,None]**2 * get_eta_MB(q, E, m_X, astro_model) * F_DM[:,None]**2 * cff2[:,:] * eps_screening[:,:] #(q,E)
    return integrand

def get_dR_dE(epsilon, m_X, mediator, astro_model, eps_screening):
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

    M_cell = epsilon.M_cell # mass of cell
    q = epsilon.q*alpha*m_e # momentum
    E  = epsilon.E # energy

    N_cell = kg/M_cell #number of cells in one kg of material
    reduced_mass = m_X * m_e /(m_X + m_e)

    prefactor = (rho_X/m_X) * N_cell * cross_section  * alpha * m_e**2 / reduced_mass**2 #this might be off by factor of 2pi

    cff2 = crystal_form_factor2_epsilon(epsilon)
    integrand = momentum_integrand(q, E, cff2, eps_screening, m_X, mediator, astro_model)

    dR_dE = np.empty(E.shape[0], dtype='float')
    for i_E in range(E.shape[0]):
        dR_dE[i_E] = sp.integrate.simpson(integrand[:,i_E], q)
    dR_dE = prefactor * dR_dE / cm2sec / sec2yr

    return dR_dE, E #(E,)

def rate(epsilon, m_X, astro_model, mediator='light', screening='RPA'):
    """
    Returns total rate by integrating dR/dE over energy
    """
    if screening == 'RPA':
        eps_screening = epsilon.eps
    elif screening == 'TF':
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = ThomasFermi(E, q)
    elif screening == 'Lindhard':
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = Lindhard(E, q, 0.1)
    elif screening == None:
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = np.ones((q.shape[0], E.shape[0]), dtype='complex')

    dR_dE, E = get_dR_dE(epsilon, m_X, mediator, astro_model, eps_screening=eps_screening)

    R = sp.integrate.simpson(dR_dE, E) #need to multiply by probability from RK secondary ionization
    return R


def crystal_form_factor2_epsilon(epsilon):
    """
    Calculates the squared crystal form factor from Im(eps) using eq. 16 in darkELF paper
    """
    eps = epsilon.eps
    V_cell = epsilon.V_cell / (alpha * m_e)**3
    q = epsilon.q*alpha*m_e

    crystal_form_factor2 = q[:,None]**5 * V_cell / 8 / pi**2 * np.imag(eps) / (alpha * m_e)**2 # If q in eV, this should be / (alpha * m_e)**2

    return crystal_form_factor2 #(q,E)

def d_rate_RamanathanQ(epsilon, m_X, ionization_file, mediator, astro_model=default_astro, screening='RPA'):

    if screening == 'RPA':
        eps_screening = epsilon.eps
    elif screening == 'TF':
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = ThomasFermi(E, q)
    elif screening == 'Lindhard':
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = Lindhard(E, q, 0.1)
    elif screening == None:
        q, E = epsilon.q*alpha*m_e, epsilon.E
        eps_screening = np.ones((q.shape[0], E.shape[0]), dtype='complex')

    dR_dE, E = get_dR_dE(epsilon, m_X, mediator=mediator, astro_model=astro_model, eps_screening=eps_screening)
     
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

#exclusion plot
def ex(epsilon, mediator, astro_model=default_astro, screening='RPA'):
    simga_e_0 = astro_model['sigma_e']

    m_X_list = np.logspace(5, 9, 100)
    sigma_e = np.empty_like(m_X_list)
    for i, m_X in enumerate(m_X_list):
        sigma_e[i] = 2.3*simga_e_0 / rate(epsilon, m_X, astro_model=astro_model, mediator=mediator, screening=screening)
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