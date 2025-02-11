# dielectric_pyscf
Dielectric function calculation with Pyscf, analytical calculation of exact RPA $\epsilon(\omega, \mathbf{q})$

Requirements:
 - python version, > 3.9.6
 - pyscf version, > 2.5.0

To execute code from terminal:\
`python3 -m dielectric_pyscf`\
(This executes dielectric_pyscf/__main__.py)

Can also execute within python instance:
```
import dielectric_pyscf
dielectric_pyscf.dielectric_functions.main()
```

To perform bandstucture calculation (temporary):\
`python3 bandstructure_testing/dft_testing_functions.py`