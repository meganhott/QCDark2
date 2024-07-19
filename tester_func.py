import numpy as np
from cartesian_moments import get_E_ijt
def primgauss_1D_overlaps_uR(R: float, primindices: np.ndarray, q: np.ndarray, atom_locs: np.ndarray) -> np.ndarray:
    TN = int(round(primindices[-1, -1] + primindices[-1, 1] + 1))
    res = np.zeros((TN, TN, len(q))).astype('complex128')
    for p1 in primindices:
        a1, l1, e1, I1 = int(p1[0]), int(p1[1]), p1[2], int(p1[3])
        A = atom_locs[a1]
        for p2 in primindices:
            a2, l2, e2, I2 = int(p2[0]), int(p2[1]), p2[2], int(p2[3])
            B = atom_locs[a2]
            E_ijt = get_E_ijt(e1,e2,l1,l2,A-B-R)
            p = e1 + e2
            P = (e2*(B+R) + e1*A) / p
            for i1 in range(l1+1):
                I = I1 + i1
                for i2 in range(l2+1):
                    J = I2+i2
                    res[I, J, :] = np.sqrt(np.pi/p) * np.exp(1j*q*P - q**2/4/p)*sum([E_ijt[i1][i2][t]*(1j*q)**t for t in range(i1+i2+1)])
    return res