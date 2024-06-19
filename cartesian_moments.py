import numpy as np
import math
from scipy.special import factorial2
from scipy.special import gamma

e = math.e
etol = 10.**8

def fact2(n):
     if n <= 1:
          return 1.
     else:
          return factorial2(n)

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
          for j in range(j_max+1):
               for t in range(i + j + 1):
                    E_ijt[i][j][t] = E(a, b, i, j, t, Qx, p, q)
     
     return E_ijt

def get_prim_1D_overlap(i: int, j: int, prim_array: np.ndarray, locations: np.ndarray, unique_R: np.ndarray, unique_k: np.ndarray, target: dict) -> None:
     """
     NOTE: UNTESTED!
     Function to get 1D overlap matrix, \int dx G_i (x - A - R, a) * e^{i kx x} * G_j (x - B, b).
     Inputs:
          i:             int, index of 1D primitive gaussian #1 in prim_array
          j:             int, index of 1D primitive gaussian #2 in prim_array
          prim_array:    np.ndarray of shape (N,4) from routines.py/gen_prim_gauss_indices
                         each row contains the following information: 
                              [atom_id, <total angular momentum on l>, exponent on primitive gaussian, <index of first point in primgauss>]
          locations:     np.ndarray of shape (n_atom, 3), location of atom with atom_id
          unique_R:      np.ndarray of shape (nR, ) containing all unique R from the set of R_vectors
          unique_k:      np.ndarray of shape (nk, ) containing all unique k from the set of k_vectors
          target:        multiprocessing.Manager().dict() pointer
                         results get added to this dictionary like object.
                         target[int(prim_array[i, 3])][int(prim_array[i, 4])][direction][R][k] = complex128
     Returns:
          None
     """
     prim1, prim2 = prim_array[i], prim_array[j]
     atom1, atom2 = locations[int(prim1[0]), int(prim2[0])]
     i_max, j_max = int(prim1[1], prim2[1])
     a, b = prim1[2], prim2[1]
     init1, init2 = int(prim1[3]), int(prim2[3])
     p = a + b

     for dir in range(3):
          Bx = atom2[dir]
          for R in unique_R:
               Ax = atom1[dir] + R
               Px = (a*Ax + b*Bx)/p
               E_ijt = get_E_ijt(a, b, i_max, j_max, Ax - Bx)
               re1 = {}
               for i in range(i_max + 1):
                    for j in range(j_max + 1):
                         target[init1+i][init2+j][dir] = {}
                         target[init1+i][init2+j][dir][R] = {}
                         for k in unique_k:
                              target[init1+i][init2+j][dir][R][k] = 0
                              for t in range(i+j+1):
                                   target[init1+i][init2+j][dir][R][k] += E_ijt[i][j][k] * (1.j * k)**t
                              target[init1+i][init2+j][dir][R][k] *= ((np.pi)**0.5)*(np.e**(1.j*k*Px - (k**2)/(4.*p)))
     return None

class AO(object):
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
     
     Q: can it be sped up? Can we eliminate the need to read in primgauss object?
     Notes:    
          1.   It is easy to pass the location array.
          2.   For any exp in self.exp and i in self.shell, we can use exp and i as keys in dictionary.
          3.   This would have to change the goal, and we will have to list atom_id = self.atom as a key
               to be passed to the overlap dictionary. This is because of the following:
               self.prim_indices contains information on the location already. 
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
          '''
          Routine to normalize the basis functions. Note that s and p orbitals have different normalizations
          for cartesian and spherical shells, while higher orbitals carry same normalization scheme no matter what.
          Function modifies attribute:
               self.norm
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
               N = N**-.5
               for ia in range(num_exps):
                    self.coef[ia] *= N
          else:
               self.norm = np.sqrt(np.power(2, L + 2.5)*
                                   np.power(self.exp, L + 1.5)/
                                   gamma(1.5 + L))
          return None

     def find_location_and_indices(self, primgauss: np.ndarray) -> None:
          """
          Routine to find location and index in primgauss for each direction.
          Modifies attributes:
               self.loc
               self.prim_indices
          """
          self.prim_indices = []
          cond0 = primgauss[:,0].astype(int) == self.atom
          self.loc = primgauss[np.where(cond0)[0][0]][4:]
          for i in self.shell:
               a = []
               cond1 = cond0*(primgauss[:,2]==i)
               for exp in self.exp:
                    cond2 = cond1*np.isclose(primgauss[:,3], exp)
                    a.append(np.where(cond2)[0][0])
               self.prim_indices.append(a)
          self.prim_indices = np.array(self.prim_indices)
          return None