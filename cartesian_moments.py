import numpy as np
import math

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

class primitive_gaussian_1D(object):
     def __init__(self, exp: float, l: int, loc: list[float] = [0., 0., 0.]) -> None:
          self.origin = loc
          self.exp = exp
          self.l = l

