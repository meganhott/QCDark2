"""
This script parses the given input file. Custom default options may be specified below.
"""

defaults = {'mpi':False, 'save_3d':False, 'dir_1d':None, 'dir_1d_exact_angle': False, 'binning_1d':False, 'optical_limit':False, 'effective_core_potential':None, 'pseudo':None, 'orth':False, 'density_fitting':'MDF', 'precision':1e-12, 'precision_R':1e-9, 'q_shift_dir':[1,1,1], 'q_shift':0.01, 'dq':0.02, 'N_theta':9, 'N_phi':16, 'dE':0.1, 'E_max':50.0}

import argparse
import numpy as np
from pyscf.pbc import gto

parser = argparse.ArgumentParser()
parser.add_argument('input_file')

args = parser.parse_args()
d = {} # inputs dict

file = args.input_file
f = open(file)
for line in f: # remove comments and spaces from input and add to dict
    line = line.partition('#')[0].strip().replace(' ', '')
    if '=' in line:
        split_input = line.split('=')
        d[split_input[0]] = split_input[1]
f.close()

generic_error = 'Input Error: {} must be specified in input file. '
true_list = ['True', 'true', 'T', 't', '1']
false_list = ['False', 'false', 'F', 'f', '0']

### Building cell. Real space lattice in angstroms ###
#implement option to have fractional lattice vectors based on lattice constant
try:
    unit = d['unit']
    if unit in ['B', 'b', 'Bohr', 'bohr', 'AU', 'au']:
        unit = 'bohr'
        c = 1.8897259886 # unit conversion to angstrom
    elif unit in ['A', 'a', 'Angstrom', 'angstrom', 'Ang', 'ang']:
        unit = 'angstrom'
        c = 1 # unit conversion to angstrom
    else:
        raise Exception('Input Error: cell units must either be "unit = bohr" or "unit = angstrom". The default unit is angstrom.')
except KeyError:
    unit = 'angstrom' #default
    c = 1

try:
    vec_str = [i.replace('[', '').replace(']', '') for i in d['lattice_vectors'].split('],[')]
    vec_str2 = [i.split(',') for i in vec_str]
    lattice_vectors = [[round(c*float(i), 6) for i in j] for j in vec_str2]
except KeyError:
    raise Exception(generic_error.format('lattice_vectors'))
    #handle other formatting errors as well

try:
    atom = d['atom']
    test_cell = gto.M(a=lattice_vectors, atom=atom, unit=unit) # test atom in pyscf
    formatted_atom_bohr = test_cell.format_atom(test_cell.atom, unit=unit) # pyscf gives standard format in bohr
    atom = [(atom[0], [round(coord/1.8897259886, 6) for coord in atom[1]]) for atom in formatted_atom_bohr] # formatted atom locations in angstrom
except KeyError:
    raise Exception(generic_error.format('atom'))
except Exception: #other errors from pyscf
    print('Error in pyscf parsing atom. A working example is: atom =  Si, 0., 0., 0.; Si, 1.3575, 1.3575, 1.3575')
    raise

try:
    basis_str = d['basis']
    basis_elements = basis_str.split(',')
    basis_dict = {}
    for el in basis_elements:
        key, value = el.split(':')
        basis_dict[key] = value
    mybasis = basis_dict

    #test in pyscf

except KeyError:
    print(generic_error.format('basis'))

try: #check that this is consistent with basis?
    effective_core_potential = d['effective_core_potential']
    if effective_core_potential == 'None':
        effective_core_potential = None
except KeyError:
    effective_core_potential = defaults['effective_core_potential']

try:
    pseudo = d['pseudo']
    if pseudo == 'None':
        pseudo = None
except KeyError:
    pseudo = defaults['pseudo']


### DFT parameters ###
try:
    orth = d['orthogonalize_dft']
    if orth in true_list:
        orth = True
    elif orth in false_list:
        orth = False
    else:
        raise Exception('Input Error: orthogonalize_dft must be either True or False.')
except KeyError:
    orth = defaults['orth']

try:
    density_fitting = d['density_fitting']
    if density_fitting not in ['GDF', 'MDF', 'RSDF', 'FFTDF']:
        raise ValueError('density_fitting must be "GDF" for Gaussian Density Fitting, "MDF" for Mixed Density Fitting, "RSDF" for Range-Separated Density Fitting, or "FFTDF" for Fast Fourier Transform Density Fitting. Do not use FFTDF for all-electron calculations due to memory constraints.')
    if density_fitting == 'FFTDF':
        print('FFT density fitting selected for DFT calculation. FFTDF is not recommended for all-electron calculations due to memory constraints but can be used for ECP or pseudopotential DFT.')
except KeyError:
    density_fitting = defaults['density_fitting']

try:
    numval = d['N_valence_bands']
    if numval in ['auto', 'Auto']:
        numval = 'auto'
    elif numval in ['all', 'All']:
        numval = 'all'
    else:
        numval = int(numval)
except ValueError:
    print('The number of valence bands to include in the dielectric function calculation, N_valence_bands, must be "all", "auto", or an integer. "auto" excludes irrelevant bands based on E_max and is recommended.')
except KeyError:
    numval = 'auto' #default

try:
    numcon = d['N_conduction_bands']
    if numcon in ['auto', 'Auto']:
        numcon = 'auto'
    elif numcon in ['all', 'All']:
        numcon = 'all'
    else:
        numcon = int(numcon)
except ValueError:
    print('The number of conduction bands to include in the dielectric function calculation, N_conduction_bands, must be "all", "auto", or an integer. "auto" excludes irrelevant bands based on E_max and is recommended.')
except KeyError:
    numcon = 'auto' #default

try:
    precision = float(d['precision_pyscf'])
except ValueError:
    raise ValueError('precision_pyscf specifies the precision of DFT calculations in pyscf and must be a number. It is recommended to keep this around 1e-9 to 1e-12.')
except KeyError:
    precision = defaults['precision']

try:
    precision_R = float(d['precision_Rcut'])
except ValueError:
    raise ValueError('precision_Rcut specifies the precision of overlaps calculated for dielectric function and must be a number. It is recommended to keep this around 1e-9.')
except KeyError:
    precision_R = defaults['precision_R']

try:
    xcfunc = d['xcfunc']
    #check that this is valid in pyscf
except KeyError:
    raise Exception(generic_error.format('xcfunc (exchange-correlation functional)'))

try: #Will this work for 2D?
    k_grid = [int(k) for k in d['k_grid'].replace(']', '').replace('[', '').split(',')]
    N_k = k_grid[0]*k_grid[1]*k_grid[2]
except ValueError:
    raise ValueError('The Monkhorst-Pack k-grid must be specified as a list of the number of k-points sampled along each reciprocal lattice vector. For example, to sample 8 k-points along each direction, k_grid = [8,8,8]')
except KeyError:
    raise Exception(generic_error.format('k_grid'))

try:
    q_shift_dir = [float(q) for q in d['q_shift_dir'].replace(']', '').replace('[', '').split(',')]

    if q_shift_dir == [0., 0., 0.]:
        raise Exception('q_shift_dir cannot be [0,0,0]')
except ValueError:
    raise ValueError('The direction of the shift in the final state Monkhorst-Pack k-grid must be specified as a vector in reciprocal space. For example, to shift in the k_x direction, q_shift_dir = [1,0,0]. This does not need to be specified if q_shift_mag = 0.')
except KeyError:
    q_shift_dir = defaults['q_shift_dir']

try:
    q_shift = float(d['q_shift_mag'])
except ValueError:
    raise ValueError('The magnitude of the shift in the final state Monkhorst-Pack grid must be a float (in units of inverse bohr). If this is set too large, interpolation of small bins may be impossible. This can be set to 0.')
except KeyError:
    q_shift = defaults['q_shift']

if q_shift == 0:
    optical_limit = True
    q_shift_dir = None
else:
    optical_limit = False

try:
    dq = float(d['dq'])

    test_cell = gto.M(a=lattice_vectors, atom=atom)
    R = test_cell.reciprocal_vectors()
    R_mag = np.linalg.norm(R, axis=0)
    dk = R_mag / k_grid # distance between k-grid points
    if (dq > dk).any():
        optical_first_bins = False
        print(f'Warning: The selected magnitude of q-bins (dq = {dq}) is larger than the distance between points in the DFT k-grid along each reciprocal lattice vector (dk = {dk}). It is recommended to make dq < all dk for proper q -> 0 behavior.')
        # This problem occurs for calculations both with and without a q-shift
    else:
        optical_first_bins = True
except ValueError:
    raise ValueError('The size of momentum bins, dq, must be a float (in units of inverse bohr).')
except KeyError:
    dq = defaults['dq'] #inverse bohr, default

### Dielectric function calculation parameters ###
try:
    q_start = d['q_start']
    if q_start == 'None':
        q_start = None
    else:
        q_start = int(q_start)
        if q_start < 0 or q_start > N_k:
            raise Exception(f'q_start must be between 0 and the total number of k in k_grid, {N_k}')
except KeyError:
    q_start = None #default
except Exception:
    raise

try:
    q_stop = d['q_stop']
    if q_stop == 'None':
        q_stop = None
    else:
        q_stop = int(q_stop)
        if q_stop < 0 or q_stop > N_k or q_stop < q_start:
            raise Exception(f'q_stop must be between 0 and the total number of unique q (which is often the size of the k_grid, {N_k}), and must be larger than q_start.')
except KeyError:
    q_stop = None #default
except Exception:
    raise

try:
    scissor_bandgap = d['scissor_bandgap']
    if scissor_bandgap == 'None':
        scissor_bandgap = None
    else:
        scissor_bandgap = float(scissor_bandgap)
except ValueError:
    raise ValueError('Scissor-corrected bandgap must be a float (in units of eV) or None.')
except KeyError:
    scissor_bandgap = None # No scissor correction by default

try:
    include_lfe = d['include_lfe']
    if include_lfe in true_list:
        include_lfe = True
    elif include_lfe in false_list:
        include_lfe = False 
    else:
        raise Exception('Input Error: include_lfe must be either True or False.')
except KeyError:
    include_lfe = False #default

### Binning parameters ###
try:
    q_max = float(d['q_max'])
except ValueError:
    raise ValueError('The maximum momentum tranferred, q_max, must be a float (in units of inverse bohr).')
except KeyError:
    raise Exception(generic_error.format('the maximum momentum tranferred, q_max,'))

try:
    q_min = float(d['q_min'])
    if include_lfe and q_min != 0:
        raise Exception('If local field effects are included in dielectric function calculation (include_lfe parameter), q_min must be 0.')
except ValueError:
    raise ValueError('The minimum momentum tranferred, q_max, must be a float (in units of inverse bohr).')
except KeyError:
    q_min = 0 #default

try:
    N_theta = int(d['N_theta'])

    if not N_theta%2:
        print(f'Warning: N_theta = {N_theta} is even, so the dielectric function with not contain points in the x-y plane. The accuracy of the dielectric function will remain unaffected.')

except ValueError:
    raise ValueError('The number of theta bins, N_theta, must be an integer.')
except KeyError:
    N_theta = defaults['N_theta']

try:
    N_phi = int(d['N_phi'])
except ValueError:
    raise ValueError('The number of phi bins, N_phi, must be an integer.')
except KeyError:
    N_phi = defaults['N_phi']

try:
    dE = float(d['dE'])
except ValueError:
    raise ValueError('The size of energy bins, dE, must be a float (in units of eV).')
except KeyError:
    dE = defaults['dE']

try:
    E_max = float(d['E_max'])
except ValueError:
    ValueError('The maximum energy tranferred, E_max, must be a float (in units of eV).')
except KeyError:
    E_max = defaults['E_max']


### Output files, logging, and MPI ###
try:
    save_path = d['save_path']
except KeyError:
    raise Exception(generic_error.format('save_path') + 'This is the folder where all DFT and dielectric function results will be stored. Ensure that this directory has plenty of extra storage.')

try:
    name = d['name']
    if q_start is not None and q_stop is not None:
        name = f'{name}_{q_start}_{q_stop}'
    system_name = f'{save_path}/{name}'
    store = system_name + '_resources'
except KeyError:
    raise Exception(generic_error.format('name') + 'The log files, intermediate calculations, and dielectric function output will be stored in <save_path>/<name>_resources.')

try:
    qcdark_outfile = str(d['qcdark_outfile'])
except KeyError:
    qcdark_outfile = f'{store}/{name}_eps.log' # default dielectric function log file

try:
    pyscf_outfile = str(d['pyscf_outfile'])
except KeyError:
    pyscf_outfile = f'{store}/{name}_pyscf.log' # default pyscf log file

try:
    DFT_resources_path = d['DFT_resources_path']
except KeyError:
    DFT_resources_path = save_path # is saved to same directory as dielectric function results by default

try:
    pyscf_outlev = int(d['pyscf_outlev'])
except:
    pyscf_outlev = 4

try:
    debug_logging = d['debug_logging']
    if debug_logging in true_list:
        debug_logging = True
    elif debug_logging in false_list:
        debug_logging = False
    else:
        raise Exception('Input Error: debug_logging parameter must be either True or False.')
except KeyError:
    debug_logging = False

try:
    mpi = d['mpi']
    if mpi in true_list:
        mpi = True
    elif mpi in false_list:
        mpi = False
    else:
        raise Exception('Input Error: mpi parameter must be either True or False.')
except KeyError:
    mpi = defaults['mpi']

try:
    save_3d = d['save_3d']
    if save_3d in true_list:
        save_3d = True
    elif save_3d in false_list:
        save_3d = False
    else:
        raise Exception('Input Error: save_3d parameter must be either True or False.')
except KeyError:
    save_3d = defaults['save_3d']

try:
    dir_1d = d['dir_1d']
    if dir_1d == 'None':
        dir_1d == None
    else:
        dir_1d = [float(a) for a in dir_1d.replace('[', '').replace(']', '').split(',')]

    if include_lfe:
        raise Exception('1D dielectric function calculation is not supported for LFEs. If you only want to calculate the dielectric function along a single axis, you must set include_lfe = False.')

except ValueError:
    ValueError('dir_1d specifies a direction to calculate the 1D dielectric function along, and must be a cartesian coordinate. For example: dir_1d = [1,1,1]')
except KeyError:
    dir_1d = defaults['dir_1d']

try:
    dir_1d_exact_angle = d['dir_1d_exact_angle']
    if dir_1d_exact_angle in true_list:
        dir_1d_exact_angle = True
    elif dir_1d_exact_angle in false_list:
        dir_1d_exact_angle = False
    else:
        raise Exception('Input Error: dir_1d_exact_angle must be either True or False.')
except KeyError:
    dir_1d_exact_angle = defaults['dir_1d_exact_angle']

try:
    binning_1d = d['binning_1d']
    if binning_1d in true_list:
        binning_1d = True
    elif binning_1d in false_list:
        binning_1d = False
    else:
        raise Exception('Input Error: binning_1d must be either True or False.')
except KeyError:
    binning_1d = defaults['binning_1d']