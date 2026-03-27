---
title: Input Parameters
layout: default
nav_order: 4
--- 
# Input Parameters
This page details all user inputs that can be specified in the <>.in file. The inputs are read by `input_parameters.py`.

Comments are denoted by "#" as in python

## Example for required inputs only
```
# Example for silicon
save_path = dielectric_pyscf_results
name = Si_cc-pvtpdz_pbe_8k

lattice_vectors = [[2.715, 2.715, 0.0], [0.0, 2.715, 2.715], [2.715, 0.0, 2.715]]
atom = Si, 0., 0., 0.; Si, 1.3575, 1.3575, 1.3575
basis = Si: cc-pv(t+d)z

xcfunc = pbe
k_grid = [8,8,8]
```
## Example for all inputs
```
# Example for silicon
save_path = dielectric_pyscf_results
name = Si_cc-pvtpdz_pbe_8k
qcdark_outfile = dielectric_pyscf_results/Si_cc-pvtpdz_pbe_8k_resources/Si_cc-pvtpdz_pbe_8k_pyscf.log
pyscf_outfile = dielectric_pyscf_results/Si_cc-pvtpdz_pbe_8k_resources/Si_cc-pvtpdz_pbe_8k_eps.log
DFT_resources_path = dielectric_pyscf_results
debug_logging = False 
mpi = False

unit = angstrom
lattice_vectors = [[2.715, 2.715, 0.0], [0.0, 2.715, 2.715], [2.715, 0.0, 2.715]]
atom = Si, 0., 0., 0.; Si, 1.3575, 1.3575, 1.3575
basis = Si: cc-pv(t+d)z
precision_pyscf = 1e-12 # cell precision in pyscf
precision_Rcut = 1e-9 # precision in dielectric function calculation
pyscf_outlev = 4 # verbosity of pyscf output
effectice_core_potential = None
pseudo = None

xcfunc = pbe
orthogonalize_dft = False # use for very large basis sets
density_fitting = MDF 
q_shift = [0.01,0,0] # this can be specified manually, but it is recommended to use the default shift of 1/2 the k-grid spacing
k_grid = [8,8,8]

N_valence_bands = auto
N_conduction_bands = auto
scissor_bandgap = 1.1
include_lfe = False

q_start = None
q_stop = None

dq = 0.02
q_min = 0
q_max = 5

dE = 0.1
E_max = 50

N_theta = 9
N_phi = 16

save_3d = False
```

## Material Parameters  
`unit` : angstrom or bohr  
The units of the lattice_vectors and atom inputs. Default is angstrom  
`lattice_vectors` : [[R1_x, R1_y, R1_z], [R2_x, R2_y, R2_z], [R3_x, R3_y, R3_z]]  
Array of the real space lattice vectors in specified units   
`atom` : atom1_name, x1, y1, z1; atom2_name, x2, y2, z2  
Element name and location in unit cell for all atoms in unit cell  
`basis`  element1: basis1, element2: basis2, ...  
Gaussian basis set specified for each element in the unit cell. See the [pyscf docs](https://pyscf.org/user/pbcgto.html) for some examples.  
`precision_pyscf` : float, default : 1e-12  
Precision of pyscf Cell. It is recommended to keep this around 1e-9 to 1e-12.  
`precision_Rcut` : float, default : 1e-9    
Precision of the overlaps calculated for the dielectric function. It is recommended to keep this around 1e-9.  
`pyscf_outlev` : int, default : 4    
Verbosity level of pyscf output log from 1 to 10. Can be useful to set higher for debugging.  
`effective_core_potential` : None  
Not currently implemented  
`pseudo` : None  
Not currently implemented  

## DFT Parameters
`xcfunc`  : string  
Exchange-correlation fuctional. PBE is commonly used but there are many different functionals one can choose from. See the [pyscf docs](https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases) for more information  
`orthogonalize_dft` : True or False, default : False  
Determines if the basis is orthogonalized before SCF calculation is performed. May be required if pyscf DFT does not converge. See more information in the [pyscf docs](https://pyscf.org/user/scf.html#linear-dependencies).  
`density_fitting` : MDF, GDF, or RSDF, default : MDF  
Density fitting scheme used by pyscf. MDF is the default, but many materials can see significantly faster DFT calculations when RSDF is used. See more information in the [pyscf docs](https://pyscf.org/user/pbc/df.html).  
`q_shift` : [float, float, float] ($\alpha m_e$), default : 1/2 k-grid spacing    
The q-shift in the final state Monkhorst-Pack grid is a vector in reciprocal space (in units of inverse bohr). If this shift is set too large, interpolation of small bins may be impossible. This can be set to [0,0,0] to use the same DFT calculation for the initial and final states. By default, the q-shift will be set to half the k-grid spacing.  
`k_grid` : [int, int, int]    
The Monkhorst-Pack k-grid is specified as a list of the number of k-points sampled along each reciprocal lattice vector. For example, to sample 8 k-points along each direction, k_grid = [8,8,8].  


## Dielectric Function Parameters
`N_valence_bands` : auto, all, or an integer, default : auto    
The number of valence bands to include in the dielectric function calculation. Must be "all", "auto", or an integer. "auto" excludes irrelevant bands based on E_max and is recommended.  
`N_conduction_bands` : auto, all, or an integer, default : auto    
The number of conduction bands to include in the dielectric function calculation.  
`scissor_bandgap` : None or float (eV), default : None  
Scissor-corrected bandgap between valence and conduction bands. Shifts all conduction bands up by scissor_bandgap - DFT_bandgap.  
`include_lfe` : True or False, default : False  
Determines if local field effects (LFEs) are included in RPA calculation.   
`q_start` : None or integer, default : None  
Which unqiue q in the first Brillouin zone to start dielectric function calculation at. Useful if a large calculation must be split into multiple jobs. Setting this to None will start at the first unique q.  
`q_stop` : None or integer, default : None   
Which unqiue q in the first Brillouin zone to stop dielectric function calculation at. Setting this to None will include all q.  
`save_3d` : True or False default : False  
If True, 3D binned dielectric function is saved to hdf5 output. If False, only angular-averaged dielectric function is saved.  

## Binning Parameters
`dq` : float ($\alpha m_e$), default : 0.02  
The size of momentum bins  
`q_min` : float ($\alpha m_e$), default : 0  
The minimum transferred momentum.  
`q_max` : float ($\alpha m_e$)  
The maximum transferred momentum.  
`dE` : float (eV), default : 0.1  
The size of energy bins  
`E_max` : float (eV), default : 50  
The maximum transferred energy. (Make sure that your basis set includes enough bands to accurately capture transitions at this energy!)  
`N_theta` : int, default : 9  
Number of theta bins ($0 \leq \theta \leq 2\pi$)  
`N_phi` : int, default : 16  
Number of phi bins ($0 \leq \phi < 2\pi$)  

## Output Files and MPI
`save_path` : str
This is the folder where all DFT and dielectric function results will be stored. Ensure that this directory has plenty of extra storage  
`name` : str  
The log files, intermediate calculations, and dielectric function output will be stored in \<save_path\>/\<name\>_resources/.  
`DFT_resources_path` : str, defaut : \<save_path\>  
Folder where DFT outputs are/will be stored (default location of \<save_path\> is recommended)  
`mpi` : True or False, default : False  
If True, calculation will be parallelized across multiple *nodes* (parallelization across multiple *cores* on one node will always occur). You must ensure mpi4py has been corrected configured for your system to use this feature.  