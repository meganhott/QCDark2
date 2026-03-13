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

class df: # Dielectric function class
    def __init__(self, filename=None, eps=None, q=None, E=None, M_cell=None, V_cell=None, dE=None, binned_eps=None, bin_centers=None):
        if filename is None:
            # If filename is not specified, the user can build a df instance without an hdf5 file
            self.eps = eps
            self.q = q
            self.E = E 
            self.M_cell = M_cell
            self.V_cell = V_cell 
            if dE is None:
                self.dE = E[1] - E[0]
            else:
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
            self.V_cell = h5.attrs['V_cell'] # Unit cell volume in units of Bohr = (alpha*m_e)^-3
            self.dE = float(h5.attrs['dE']) # Size of energy bins in eV
            # Other parameters can be accessed through self.h5.attrs.items()

            #if 'binned_eps' in h5.keys(): # for anisotropic crystal - still in development
            #    self.binned_eps = h5['binned_eps'][:]
            #    self.bin_centers = h5['bin_centers'][:]
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

def load_epsilon(filename):
    """
    Loads angular averaged dielectric function. filename should be the location of epsilon.hdf5 file. 
    """
    epsilon = df(filename)
    return epsilon

def get_elf(eps):
    return np.imag(eps)/((np.imag(eps))**2+np.real(eps)**2)

def get_F_DM(q, m_A=0):
    """
    Return the non-rel. dark matter form factor F_DM for a given mediator mass and momentum
    Inputs:
        q: (N_q,) np.ndarray: momentum in ame
        m_A: float: dark mediator mass in units of eV
    Output:
        F_DM: (N_q, ) np.ndarray: DM form factor
    """
    F_DM = (m_A**2 + (alpha*m_e)**2) / (m_A**2 + (alpha*m_e*q)**2)
    return F_DM

def get_eta_MB(q, E, m_X, astro_model=default_astro):
    """
    Calculates the integrated Maxwell-Boltzmann distribution eta(v_{min}(q,E)).
    Inputs:
        q: (N_q,) np.ndarray: momentum in ame
        E: (N_E,) np.ndarray: energy in eV
        m_X: float: dark matter mass in eV
        astro_model: dict: astrophysical DM parameters
    Output:
        eta: (N_q, N_E) np.ndarray: unitless (or c^-1)
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

def get_dR_dE(epsilon, m_X, mediator, astro_model=default_astro, screening='RPA', velocity_dist='MB'):
    """
    Returns differential rate with respect to energy for non-rel. DM. This can be used to calculate the total rate or the rate per ionization.
    Inputs:
        epsilon: df: dielectric function object
        m_X: float: DM mass in eV
        mediator: str: 'light' or 'heavy'
        astro_model: dict: astrophysical DM parameters
        screening: str: screening model to use. See choose_screening function for options
        velocity_dist: str: Non-rel. velocity distribution (e.g. halo DM = 'MB)
    Output:
        dR_dE: (N_E, ) np.ndarray: scattering rate in events/kg/year/eV
        E: (N_E, ) np.ndarray: energies (in eV) corresponding to dR_dE
    """
    rho_X = astro_model['rhoX']
    cross_section = astro_model['sigma_e']

    rho_T = epsilon.M_cell / kg / epsilon.V_cell # density of target in units of kg/bohr^3
    reduced_mass = m_X * m_e /(m_X + m_e)

    prefactor = (1/rho_T) * (rho_X/m_X)  * (cross_section/reduced_mass**2) / (4*np.pi)

    integrand = momentum_integrand(epsilon, m_X, mediator, astro_model, velocity_dist)

    # Option to implement different screening
    eps_screening = choose_screening(epsilon, screening)
    integrand = integrand * np.abs(epsilon.eps)**2 / np.abs(eps_screening)**2 # Replacing RPA screening

    q = epsilon.q * alpha * m_e # momentum in units of eV
    E  = epsilon.E # energy in units of eV

    dR_dE = np.empty(E.shape[0], dtype='float')
    for i_E in range(E.shape[0]):
        dR_dE[i_E] = sp.integrate.simpson(integrand[:,i_E], q)
    dR_dE = prefactor * dR_dE  / cm2sec / sec2yr #* alpha*m_e

    return dR_dE, E 

def choose_screening(epsilon, screening):
    """
    Returns dielectric function to use for screening.
    """
    E = epsilon.E
    q = epsilon.q*alpha*m_e
    if screening in ['TF', 'TF Si', 'MTF', 'MTF Si']:
        eps_screening = ThomasFermi(E, q, 'Si')
    elif screening in ['TF Ge','MTF Ge']:
        eps_screening = ThomasFermi(E, q, 'Ge')
    elif screening in ['Lindhard', 'Si Lindhard']:
        eps_screening = Lindhard(E, q, 0.1)
    elif screening in ['Lindhard Ge']:
        raise NotImplementedError('Lindhard screening not implemented for Ge.')
    elif screening in [None, 'None', 'No']:
        eps_screening = np.ones((q.shape[0], E.shape[0]), dtype='complex')
    elif screening == 'RPA':
        eps_screening = epsilon.eps
    else:
        raise KeyError(f'Invalid dielectric screening option specified: {screening}')
    return eps_screening

def rate(epsilon, m_X, astro_model=default_astro, mediator='light', screening='RPA', velocity_dist='MB'):
    """
    Returns total rate for non-rel. DM by integrating dR/dE over energy
    Inputs:
        epsilon: df: dielectric function object
        m_X: float: DM mass in eV
        astro_model: dict: astrophysical DM parameters
        mediator: str: 'light' or 'heavy'
        screening: str: screening model to use. See choose_screening function for options
        velocity_dist: str: Non-rel. velocity distribution (e.g. halo DM = 'MB)
    Output:
        R: float: rate in units of events/kg/year
    """
    dR_dE, E = get_dR_dE(epsilon, m_X, mediator, astro_model, screening, velocity_dist)

    R = sp.integrate.simpson(dR_dE, E)
    return R

def crystal_form_factor2_epsilon(epsilon):
    """
    Calculates the squared crystal form factor from Im(eps) using eq. 16 in DarkELF paper
    """
    eps = epsilon.eps
    V_cell = epsilon.V_cell / (alpha * m_e)**3
    q = epsilon.q*alpha*m_e

    crystal_form_factor2 = q[:,None]**5 * V_cell / 8 / pi**2 * np.imag(eps) / (alpha * m_e)**2 # If q in eV, this should be / (alpha * m_e)**2

    return crystal_form_factor2 #(q,E)

#exclusion plot
def ex(epsilon, mediator, astro_model=default_astro, screening='RPA', velocity_dist='MB', cl=0.9, m_X_min=1e5, m_X_max=1e9, N_m_X=100):
    N_exp = -np.log(1 - cl) #number of expected events for given confidence level. E.g. for 0.9, N_exp = 2.3

    simga_e_0 = astro_model['sigma_e']

    m_X_list = np.logspace(np.log10(m_X_min), np.log10(m_X_max), N_m_X)
    sigma_e = np.empty_like(m_X_list)
    for i, m_X in enumerate(m_X_list):
        s = N_exp * simga_e_0 / rate(epsilon, m_X, astro_model=astro_model, mediator=mediator, screening=screening, velocity_dist=velocity_dist)
        if s < 0:
            sigma_e[i] = np.nan
        else:
            sigma_e[i] = s
    return m_X_list, sigma_e

def dsigma_rel2(epsilon, v, sigma_e, m_X, m_A, mediator='vector', screening='RPA'):
    """
    Calculates the cross section for DM with velocity v (can be relativistic)
    Inputs:
        epsilon: df: dielectric function object
        v: float: velocity (unitless)
        sigma_e: float: reference cross section in cm^2
        m_X: float: DM mass in eV
        m_A: float: dark mediator mass in eV
        mediator: str: type of dark mediator: 'vector', 'scalar', or 'approx' (non-rel. approximation)
        screening: str: screening model to use. See choose_screening function for options
    Output:
        d\sigma_dE: (N_E, ) np.ndarray: Cross section in events cm^2 / eV / atom
    """
    V = epsilon.V_cell / (alpha*m_e)**3 # cell volume in eV^-3
    N_cell = 2 # may need to change for new materials!
    n = N_cell / V # number density
    E = epsilon.E
    q = epsilon.q * alpha * m_e

    m_Xe = m_e * m_X / (m_e + m_X) # DM-e reduced mass

    gamma = 1/np.sqrt(1 - v**2)
    E_X = gamma * m_X

    if mediator == 'vector':
        integrand = q[:,None]**3 / (E_X - E[None,:]) * ((2*E_X - E[None,:])**2 - q[:,None]**2) / (E[None,:] - q[:,None]**2 - m_A**2)**2 * epsilon.elf()
    elif mediator == 'scalar':
        integrand = q[:,None]**3 / (E_X - E[None,:]) * (q[:,None]**2 - E[None,:]**2 + 4*m_X**2) / (E[None,:] - q[:,None]**2 - m_A**2)**2 * epsilon.elf()
    elif mediator == 'approx':
        integrand = q[:,None]**3 / (E_X - E[None,:]) * 4*m_X**2 / (E[None,:] - q[:,None]**2 - m_A**2)**2 * epsilon.elf()
    elif mediator == 'approx full':
        integrand = q[:,None]**3 / m_X * 4*m_X**2 / (q[:,None]**2 + m_A**2)**2 * epsilon.elf() * gamma
    else:
        raise KeyError('Mediator must be "vector" or "scalar".')

    eps_screening = choose_screening(epsilon, screening)
    integrand = integrand * np.abs(epsilon.eps)**2 / np.abs(eps_screening)**2 # Replacing RPA screening

    prefactor = sigma_e/(32 * np.pi**2 * alpha * v**2 * E_X) * (m_A**2 + (alpha*m_e)**2)**2 / m_Xe**2 / n
    dsigma = np.empty(E.shape[0], dtype='float')

    for i, E_i in enumerate(E):
        if (gamma*m_X - E_i)**2 < m_X**2:
            dsigma[i] = 0
        else:
            q_min = gamma*v*m_X - np.sqrt((gamma*m_X - E_i)**2 - m_X**2)
            q_max = gamma*v*m_X + np.sqrt((gamma*m_X - E_i)**2 - m_X**2)
            try:
                q_i = np.min(np.where(q > q_min)[0]) # index of q_min
                q_f = np.max(np.where(q < q_max)[0]) # index of q_max
                dsigma[i] = sp.integrate.trapezoid(integrand[q_i:q_f,i], q[q_i:q_f])
            except:
                dsigma[i] = 0
    
    return prefactor * dsigma # cm^2 / eV / atom

def recoil_spectrum(dR, ionization_file='../secondary_ionization/p100K.dat', E_max=50, dE=0.1):
    """
    Calculates the electron recoil spectrum
    Inputs:
        dR: (N_E, ) np.ndarray: differential rate (dR_dE) in events/kg/year/eV
        ionization_file: str: location of ionization yield model
        E_max: float
        dE: float
    Output:
        R_Q: (N_Q, ) np.ndarray: \Delta R_Q - events for a given ionization yield Q in events/kg/year
    """
    E = np.arange(0, E_max + dE, dE)

    ionization_inp = np.genfromtxt(ionization_file).transpose()
    E_ionization = ionization_inp[0]
    pair_creation_prob = ionization_inp[1:]
    E_min, E_max = E_ionization.min(), E_ionization.max()
    #print('Input file has probabilities listed for {:.2f} eV <= E <= {:.2f} eV.\nAll rates outside this range will be ignored.'.format(E_min, E_max))

    E = np.round(E, 5)
    E, dR = E[(E >= E_min) & (E <= E_max)], dR[(E >= E_min) & (E <= E_max)]
  
    N_max_pairs = pair_creation_prob.shape[0]
    R_Q = np.zeros(N_max_pairs, dtype='float')
    for n in range(N_max_pairs):
        R_Q[n] = np.sum(dR*dE*np.interp(E, E_ionization, pair_creation_prob[n]))

    return R_Q

def get_rate_flux(epsilon, m_X, sigma_e, flux, v_list, m_A=0, mediator='vector', screening='RPA'):
    """
    Returns the scattering rate dR_dE for a given flux as a function of velocity
    Inputs:
        epsilon: df: dielectric function object
        m_X: float: DM mass in eV
        sigma_e: float: reference cross section in cm^2
        flux: (N_v, ) np.ndarray: Flux d\Phi/dv in cm^-2s^-1
        v_list: (N_v, ) np.ndarray: velocity (unitless)
        m_A: float: dark mediator mass in eV
        mediator: str: type of dark mediator: 'vector', 'scalar', or 'approx' (non-rel. approximation)
        screening: str: screening model to use. See choose_screening function for options
    Output:
        dR_dE: (N_E, ) np.ndarray: Scattering rate in events / kg / year / eV
    """
    E = epsilon.E

    M_cell = epsilon.M_cell # cell mass in eV
    N_cell = 2 # may need to change for new materials!
    prefactor = N_cell/M_cell #atom/mass
    
    dsigma = np.zeros((v_list.shape[0], E.shape[0]))
    for i, v in enumerate(v_list): # calculate cross sections for all velocities
        dsigma[i] = dsigma_rel2(epsilon, v, sigma_e, m_X, m_A, mediator=mediator, screening=screening) #cm^2/eV/atom

    integrand = dsigma * flux[:,None]
    dR = np.zeros(E.shape[0])
    for i, E_i in enumerate(E): #integrate over velocity to get rate
        dR[i] = sp.integrate.trapezoid(integrand[:,i], v_list)

    return prefactor * dR * kg / sec2yr # events / kg / year / eV


#Modified Thomas-Fermi model for Si and Ge
def ThomasFermi(E, q, material='Si'):
    """
    Returns the "modified Thomas-Fermi" dielectric function, i.e. the model from Cappellini et al., https://link.aps.org/doi/10.1103/PhysRevB.47.9892
    Inputs:
        E: (N_E, ) np.ndarray: energy in eV
        q: (N_q, ) np.ndarray: momentum in eV (not in ame!!)
        material: str: 'Si' or 'Ge'
    Output:
        eps_MTF: (N_q, N_E) np.ndarray: MTF dielectric function (only has real part)
    """
    E_mesh, q_mesh = np.meshgrid(E, q)

    if material == 'Si':
        eps0 = 11.3
        tau = 1.563 
        E_plasmon = 16.6 #eV
        q_TF = 4.13e3 #eV
    elif material == 'Ge':
        eps0 = 14.0
        tau = 1.563 
        E_plasmon = 15.2 #eV
        q_TF = 3.99e3 #eV
    else:
        raise KeyError('Thomas-Fermi model of the dielectric function is only implemented for Si and Ge.')

    eps = 1 + ( 1/(eps0-1) + tau*(q_mesh/q_TF)**2 + q_mesh**4/(4*m_e**2*E_plasmon**2) - (E_mesh/E_plasmon)**2 )**(-1)
    return eps

#Lindhard model for Si
def Lindhard(E, q, fp):
    """
    Returns the Lindhard dielectric function
    Inputs:
        E: (N_E, ) np.ndarray: energy in eV
        q: (N_q, ) np.ndarray: momentum in eV (not in ame!!)
        fp: float: plasmon peak width - 0.1 is recommended for Si
    Output:
        eps_Lindhard: (N_q, N_E) np.ndarray: Lindhard dielectric function
    """
    VCell = 5.209e-9
    nValence = 8
    MCell = 52322355000.0
    mElectron = 5.1099894e5
    alpha = 1.0/137.03599908

    E_mesh, q_mesh = np.meshgrid(E, q)

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