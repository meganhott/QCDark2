import numpy as np
import math
from scipy.special import factorial2 as fact2
from scipy.special import gamma

e = math.e

def get_E_ijt(a: float, b: float, i_max: int, j_max: int, Qx: float) -> list[list[list[float]]]:
     """
     Recursively calculate Hermite Gaussian Coefficients for 1d primitive gaussians.
     Inputs:
          a, b           = exponents of 1d primitive Gaussian 'a' and 'b' respectively
          i_max, j_max   = maximum angular momentum number on 1d primitive Gaussian 'a' and 'b' respectively
          Qx             = distance between origins of 1d primitive Gaussian 'a' and 'b', Qx = Ax - Bx
     Returns:
          list[list[list[float]]]: E_{ij}^t for a, b, Qx
     """
     
     p = a + b
     q = a*b/p

     if i_max < 0 or j_max < 0:
          raise ValueError('i and j must be greater than 0. Given i = {}, j = {}'.format(i_max, j_max))


     E_ijt = []
     for i in range(i_max + 1):
          tem1 = []
          for j in range(j_max+1):
               tem1.append([None]*(i+j+1))
          E_ijt.append(tem1)

     def E(a: float, b: float, i: int, j: int, t: int, Qx: float, p: float, q: float) -> float:
          """
          Recursively calculate Hermite Gaussian Coefficients for 1d primitive gaussians.
          Inputs:
               a, b = exponents of 1d primitive Gaussian 'a' and 'b' respectively
               i, j = angular momentum number on 1d primitive Gaussian 'a' and 'b' respectively
               t    = number of nodes in the Hermite
               Qx   = distance between origins of 1d primitive Gaussian 'a' and 'b', Qx = Ax - Bx
               p, q = derived quantities from a and b used frequently. Speeds up calculation.
          Returns:
               float: E_{ij}^t for a, b, Qx
          """
          nonlocal E_ijt

          if (t < 0) or (t > i+j):                                    # 0 < t < i + j
               return 0.
          if not E_ijt[i][j][t]:
               if i == j == t == 0:                                      # E_{00}^0 = K_{AB} = e^{-q Qx^2}
                    E_ijt[i][j][t] = e**(-q*Qx*Qx)
               elif j == 0:                                                # Decrement index i
                    E_ijt[i][j][t] =    1./(2.*p)*E(a, b, i-1, j, t-1, Qx, p, q) - \
                                        q*Qx/a*E(a, b, i-1, j, t, Qx, p, q)      + \
                                        (t+1.)*E(a, b, i-1, j, t+1, Qx, p, q)
               else:                                                       # Default decrement index j
                    E_ijt[i][j][t] =    1./(2.*p)*E(a, b, i, j-1, t-1, Qx, p, q) + \
                                        q*Qx/b*E(a, b, i, j-1, t, Qx, p, q)      + \
                                        (t+1.)*E(a, b, i, j-1, t+1, Qx, p, q)
          return E_ijt[i][j][t]
     

     for i in range(i_max + 1):
          tem1 = []
          for j in range(j_max+1):
               tem2 = []
               for t in range(i + j + 1):
                    E_ijt[i][j][t] = E(a, b, i, j, t, Qx, p, q)
     
     return E_ijt

"""
Test object for now -- this is definitely too complicated. 
Q -> How do we simplify this?

Once calculated for all exponents and locations, final goal is to modify the output dictionaries 
to the dict explained below and dump into a read only hdf5 file. This file will be accessed by all processes 
later on. This process will take place inside of a function defined in routines and not in cartesian_moments.

ovlp[a] = dict
     ovlp[a][b] = dict
          ovlp[a][b][d] = dict
               ovlp[a][b][d][k] = dict
                    ovlp[a][b][d][k][R] = float

a,b: index of primitive_gaussian_1d object containing information on exponents, i, j, and locations
d:   directional index, d = 0 (x), d = 1 (y), d = 2 (z)
k:   unique k in direction d
R:   unique R in direction d

If this is indeed most optimal, 
     Q: how fast can we forward this dict to multiprocessing?
          A: ?
     Q: Should we implement a multiprocessing.Manager.dict object?
          A: ?
     Q: Is it faster to dump data to hdf5 file and read from all processes separately? 
          A:   likely slower. hdf5 file stores keys as str objects, and we can do everything in dict with float.
               However, dict object can be heavy to pass to the functions, because these will be a dict containing
               O(100)*O(100)*3*O(100)*O(25) ~ O(100 million) floats ~ 700 MB
"""
def get_prim_1D_overlap(exp1: float, exp2: float, max_i: int, max_j: int, A_vec: list[float], B_vec: list[float], R_vecs: np.ndarray, kx: np.ndarray) -> None:
     """
     Function to get 1D overlap matrix, \int dx G_i (x - A - R, a) * e^{i kx x} * G_j (x - B, b).
     
     
     Inputs:
          exp1:     float, exponent of first gaussian
          exp2:     float, exponent of second gaussian
          max_i:    int, maximum angular momentum of first gaussian
          max_j:    int, maximum angular momentum of second gaussian
          A_vec:    list[float] of len == 3, origin of gaussian number 1
          B_vec:    list[float] of len == 3, origin of gaussian number 2
          Rvecs:    np.ndarray object of shape (N_R, 3)
                    contains all R vectors being considered.
          k:        np.ndarray object of shape (N_G, N_k, N_k, 3)
                    contains all vectors {q} that can be constructed from k2[None,None,:,:] - k1[None,:,None,:] + G[:,None,None,:]
     Returns:
          ovlp:     dict object containing the following:
                    ovlp[i] = dict object containing values for 0 <= i <= max_i (number of keys = i_max + 1)
                         ovlp[i][j] =   dict object containing values for 0 <= j <= max_j (number of keys = i_max + 1)
                              ovlp[i][j][d] =     dict object containing values in 
                                                  direction x (d = 0), y (d = 1) or z (d = 2)
                                   ovlp[i][j][kx][d] =    dict object containing values for given unique kx in direction d
                                        ovlp[i][j][kx][d][Rx] =  float, 
                                                                 \int dx G_i (x - A_d - R_d, a) * e^{i kx x} * G_j (x - B_d, b)
     """
     return None

class ao(object):
     """
     Object that defines an atomic orbital. 
     Attributes:
          atom:          int, index of atom
          loc:           np.ndarray of shape (3,), contains origin of atom
          shell:         tuple of len = 3, angular momentum components for all 3 directions
          exp:           np.ndarray of shape (N_prim, )
                         contains all exponents in the contracted gaussian
          coef:          np.ndarray of shape (N_prim, )
                         contains all coefficients in the contracted gaussian
          norm:          np.ndarray of shape (N_prim, )
                         contains all normalizations in the contracted gaussian
          prim_indices:  np.ndarray of shape (3, N_prim)
                         contains indices of primitive gaussian in 1D corresponding to 
                         angular momentum component in each direction and each exponent
                         as contained in primgauss np.ndarray.
     Requires following input parameters to initialize:
          atom_index:    int, index of atom being considered
          ijk:           tuple of len = 3, angular momentum components for all 3 directions
          exps:          np.ndarray of shape (N_prim, )
                         contains all exponents in the contracted gaussian
          coeffs:        np.ndarray of shape (N_prim, )
                         contains all coefficients in the contracted gaussian
          primgauss:     np.ndarray of shape (tot_prim, 7)
                         contains details of all primgauss elements.
     """
     def __init__(self, atom_index: int, ijk: tuple, exps: np.ndarray, coeffs: np.ndarray, primgauss: np.ndarray) -> None:
          self.atom = atom_index
          self.loc = None
          self.shell = tuple(ijk)
          self.exp = exps
          self.coef = coeffs
          self.norm = None
          self.normalize()
          self.prim_indices = None
          self.find_location_and_indices(primgauss)
          return None

     def normalize(self) -> None:
          ''' Routine to normalize the basis functions, in case they
               do not integrate to unity.
          '''
          l,m,n = self.shell
          L = l+m+n
          if L < 2:
          # self.norm is a list of length equal to number primitives
          # normalize primitives first (PGBFs)
               self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                         np.power(self.exp,l+m+n+1.5)/
                         fact2(2*l-1)/fact2(2*m-1)/
                         fact2(2*n-1)/np.power(np.pi,1.5))
                    # now normalize the contracted basis functions (CGBFs)
               # Eq. 1.44 of Valeev integral whitepaper
               prefactor = np.power(np.pi,1.5)*\
                    fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)

               N = 0.0
               num_exps = len(self.exp)
               for ia in range(num_exps):
                    for ib in range(num_exps):
                         N += self.norm[ia]*self.norm[ib]*self.coef[ia]*self.coef[ib]/\
                              np.power(self.exp[ia] + self.exp[ib],L+1.5)

               N *= prefactor
               N = np.power(N,-0.5)
               for ia in range(num_exps):
                    self.coef[ia] *= N
          else:
               self.norm = np.sqrt(np.power(2, L + 2.5)*
                                   np.power(self.exps, L + 1.5)/
                                   gamma(1.5 + L))
          return None

     def find_location_and_indices(self, primgauss: np.ndarray) -> None:
          self.prim_indices = []
          cond0 = primgauss[:,0].astype(int) == self.atom
          self.loc = primgauss[np.where(cond0)[0][0]][4:]
          for i in self.shell:
               a = []
               cond1 = cond0*(primgauss[:,2]==i)
               for exp in self.exp:
                    cond2 = cond1*primgauss[:,3] == exp
                    a.append(np.where(cond2)[0][0])
               self.prim_indices.append(a)
          self.prim_indices = np.array(self.prim_indices)
          return None