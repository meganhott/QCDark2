import numpy as np
 
def get_E(a: float, b: float, i: int, j: int, t: int, Qx: float) -> float:
     """
     Check contraints, calculate p and q for Hermite Gaussian Coefficients, and return Hermite
     Gaussian Coefficients is allowed.
     Inputs:
          a, b = exponents of 1d primitive Gaussian 'a' and 'b' respectively
          i, j = angular momentum number on 1d primitive Gaussian 'a' and 'b' respectively
          t    = number of nodes in the Hermite
          Qx   = distance between origins of 1d primitive Gaussian 'a' and 'b', Qx = Ax - Bx
     Returns:
          float: E_{ij}^t for a, b, Qx
     
     Upgrades to do:
          1.   Implement storage procedure for E(a, b, i, j, t, Qx): for all i, j and t calculated. 
               Do not need to recalculate these values. Input i_max and j_max and fill up a list constructed?
     """

     p = a + b
     q = a*b/p

     if i < 0 or j < 0:
          raise ValueError('i and j must be greater than 0. Given i = {}, j = {}'.format(i, j))

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
          if (t < 0) or (t > i+j):                                    # 0 < t < i + j
               return 0.
          elif i == j == t == 0:                                      # E_{00}^0 = K_{AB} = e^{-q Qx^2}
               return np.exp(-q*Qx*Qx)
          elif j == 0:                                                # Decrement index i
               return    1./(2.*p)*E(a, b, i-1, j, t-1, Qx, p, q) - \
                         q*Qx/a*E(a, b, i-1, j, t, Qx, p, q)      + \
                         (t+1.)*E(a, b, i-1, j, t+1, Qx, p, q)
          else:                                                       # Default decrement index j
               return    1./(2.*p)*E(a, b, i, j-1, t-1, Qx, p, q) + \
                         q*Qx/b*E(a, b, i, j-1, t, Qx, p, q)      + \
                         (t+1.)*E(a, b, i, j-1, t+1, Qx, p, q)

     return E(a, b, i, j, t, Qx, p, q)

def get_E_max_ij(a: float, b: float, i_max: int, j_max: int, Qx: float) -> list[list[float]]:

     E_ijt = []
     for i in range(i_max + 1):
          tem1 = []
          for j in range(j_max+1):
               tem1.append([None]*(i+j+1))
          E_ijt.append(tem1)

     def func():
          nonlocal E_ijt
     return None