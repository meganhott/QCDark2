import numpy as np
import os
import json
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc

import input_parameters as parmt
from routines import logger, time_wrapper, makedir

#Constants
me = 0.51099895000e6 #eV
alpha = 1/137

def save_dft():
    dft_params = {
        'lattice_vectors': parmt.lattice_vectors,
        'atomloc': parmt.atomloc,
        'mybasis': parmt.mybasis,
        'effective_core_potential': parmt.effective_core_potential,
        'pseudo': parmt.pseudo,
        'precision': parmt.precision,
        'xcfunc': parmt.xcfunc,
        'k_grid': parmt.k_grid,
        'q_shift_dir': parmt.q_shift_dir,
        'q_shift': parmt.q_shift,
        'dft_instance': None
    }

    makedir('DFT_resources') #makes DFT_routines directory if it does not already exist
    dft_instances = os.listdir('DFT_resources')
    for d in dft_instances:
        dft_dict = json.load(open('DFT_resources/' + d + '/dft_params.txt', 'r'))
        if all(dft_dict[key] == dft_params[key] for key in dft_dict.keys() if key not in ['dft_instance']): #compare everything except dft_instance
            dft_params['dft_instance'] = d
            if len(os.listdir('DFT_resources/' + d)) < 2: #Something went wrong with previously calculated DFT, calculation should be started again
                logger.info('There is not a stored DFT cacluation for these input parameters, a new calculation will be performed and stored as {}.'.format(d))
                new_dft = True
            else:
                logger.info('DFT already calculated for these input parameters and stored as {}'.format(d))
                new_dft = False
            return new_dft, dft_params
    
    if dft_params['dft_instance'] is None:
        if len(dft_instances) == 0:
            dft_params['dft_instance'] = 'DFT_0'
        else:
            dft_params['dft_instance'] = 'DFT_' + str(max([int(d.split('_')[1]) for d in dft_instances]) + 1)
        logger.info('There is not a stored DFT cacluation for these input parameters, a new calculation will be performed and stored as {}.'.format(dft_params['dft_instance']))
        makedir('DFT_resources/' + dft_params['dft_instance'])
        json.dump(dft_params, open('DFT_resources/' + dft_params['dft_instance'] + '/dft_params.txt', 'w')) #save dft parameters
        new_dft = True

    return new_dft, dft_params

def list_saved_dft():
    dft_instances = os.listdir('DFT_resources')
    for d in dft_instances:
        dft_dict = json.load(open('DFT_resources/' + d + '/dft_params.txt', 'r'))
        print('DFT Instance: {}'.format(dft_dict['dft_instance']))
        print('\tLattice vectors: {}'.format(dft_dict['lattice_vectors']))
        print('\tAtom Locations: {}'.format(dft_dict['atomloc']))
        print('\tBasis: {}'.format(dft_dict['mybasis']))
        print('\tEffective Core Potential: {}'.format(dft_dict['effective_core_potential']))
        print('\tPseudopotential: {}'.format(dft_dict['pseudo']))
        print('\tPrecision: {}'.format(dft_dict['precision']))
        print('\tExchange Correlation Functional: {}'.format(dft_dict['xcfunc']))
        print('\tk-grid: {}'.format(dft_dict['k_grid']))
        print('\tq Shift Direction: {}'.format(dft_dict['q_shift_dir']))
        print('\tq shift: {}'.format(dft_dict['q_shift']))

def make_kpts(cell: pbcgto.cell.Cell, dft_params: dict, with_gamma: bool = True) -> pyscf.pbc.lib.kpts.KPoints:
    """
    Function to get the grid in reciprocal unit cell given k_grid density in input_parameters.py.
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Returns:
        k_grid: np.ndarray of shape (N, 3), k vectors generated from the cell.
    """
    dft_path = 'DFT_resources/' + dft_params['dft_instance']
    k_grid = parmt.k_grid
    kpts = cell.make_kpts(k_grid, wrap_around=True, with_gamma_point=with_gamma, space_group_symmetry=True)
    np.save(dft_path + '/k-pts_i', kpts.kpts)
    logger.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_i.npy', k_grid[0], k_grid[1], k_grid[2]))
    return kpts

@time_wrapper
def KS_electronic_structure(cell: pbcgto.cell.Cell, dft_params: dict) -> pbcdft.krks_ksymm.KsymAdaptedKRKS:
    """
    Function to do density functional theory. Solves the RKS if only one kpoint in kgrid, otherwise solves RKS at each k-point and constructs density matrix from integrating over 1BZ. 
    Inputs:
        cell:       pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
        dft_params: dict, DFT parameters
    Returns:
        kmf
    Saves:
        molecular energies, molecular coefficients and molecular occupation numbers.
    """
    dft_path = 'DFT_resources/' + dft_params['dft_instance']
    logger.info('Initial state calculation:')
    kpts = make_kpts(cell, dft_params, True)
    kmf = pbcdft.KRKS(cell, kpts).density_fit()
    kmf.xc = parmt.xcfunc
    kmf.kernel()
    if kmf.converged:
        np.save(dft_path + '/mo_en_i_dft', kpts.transform_mo_energy(kmf.mo_energy))
        np.save(dft_path + '/mo_coeff_i', kpts.transform_mo_coeff(kmf.mo_coeff))
        np.save(dft_path + '/mo_occ_i', kpts.transform_mo_occ(kmf.mo_occ))
    else:
        raise ValueError('DFT not converged. Might need to orthogonalize basis before continuing (Not Implemented).')
    logger.info('Electronic structure converged, KS energy is {:.2f} Hartrees.\n\tDFT data is stored to {}.'.format(kmf.e_tot, dft_path))
    return kmf

@time_wrapper
def KS_non_self_consistent_field(kmf: pbcdft.krks_ksymm.KsymAdaptedKRKS, dft_params: dict):
    """
    Non self consistent field calculation for final states.
    Inputs:
        kmf:        pyscf.pbc.dft.krks_ksymm.KsymAdaptedKRKS object
        dft_params: dict, DFT parameters
    """
    logger.info("Final State Calculation:")
    dft_path = 'DFT_resources/' + dft_params['dft_instance']
    q_shift_dir = np.array(dft_params['q_shift_dir'])
    q_shift = dft_params['q_shift']*q_shift_dir/np.linalg.norm(q_shift_dir)
    scaled_center = kmf.cell.get_scaled_kpts(q_shift)
    kpts = kmf.cell.make_kpts(dft_params['k_grid'], space_group_symmetry=True, wrap_around = True, scaled_center = scaled_center)
    np.save(dft_path + '/k-pts_f', kpts.kpts)
    logger.info("Selected q shift = {}".format(np.array2string(q_shift, precision = 5)))
    logger.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_f.npy', dft_params['k_grid'][0], dft_params['k_grid'][1], dft_params['k_grid'][2]))
    ek , ck = kmf.get_bands(kpts.kpts_ibz)
    ek = kpts.transform_mo_energy(ek)
    ck = kpts.transform_mo_coeff(ck)
    np.save(dft_path + '/mo_en_f_dft', ek)
    np.save(dft_path + '/mo_coeff_f', ck)
    logger.info('Non self consistent field equations solved for final state k-points. Data is stored to {}.'.format(dft_path))

@time_wrapper
def convert_to_eV_and_scissor(cell: pbcgto.cell.Cell, dft_params: dict):
    """
    Function converts energies from Ryd to eV - prints bandgap, if scissor - scissor corrects bandgap.
    Inputs:
        cell:                               pyscf.pbc.gto.cell.Cell object
        dft_params:                         dict, DFT parameters
    Reads:
        parmt.scissor_bandgap:              float, scissor energies in eV
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    Writes:
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    """
    occ_orb = cell.tot_electrons()//2
    dft_path = 'DFT_resources/' + dft_params['dft_instance']
    en_i = np.load(dft_path + '/mo_en_i_dft.npy')*alpha*alpha*me #convert to eV
    en_f = np.load(dft_path + '/mo_en_f_dft.npy')*alpha*alpha*me
    homo = max(en_i[:,:occ_orb].max(), en_f[:,:occ_orb].max())
    en_i, en_f = en_i - homo, en_f - homo
    homo = 0
    lumo = min(en_i[:,occ_orb:].min(), en_f[:,occ_orb:].min())
    logger.info("All energies converted to eV. Calculated Bandgap = {:.2f} eV.".format(lumo - homo))
    if parmt.scissor_bandgap is not None:
        if type(parmt.scissor_bandgap) != float:
            raise ValueError("Parameter scissor_bandgap in input_parameters.py must be either None or of type float.")
        correction = parmt.scissor_bandgap - lumo
        en_i[:,occ_orb:], en_f[:,occ_orb:] = en_i[:,occ_orb:] + correction, en_f[:,occ_orb:] + correction
        logger.info("Scissor Correction applied, new bandgap is {:.2f} eV.".format(parmt.scissor_bandgap))
    
    makedir(parmt.store + '/DFT')
    np.save(parmt.store + '/DFT/mo_en_i.npy', en_i)
    np.save(parmt.store + '/DFT/mo_en_f.npy', en_f)

    #store mo_coeff and k points in main file as well
    coeff_i = np.load(dft_path + '/mo_coeff_i.npy')
    coeff_f = np.load(dft_path + '/mo_coeff_f.npy')
    np.save(parmt.store + '/DFT/mo_coeff_i.npy', coeff_i)
    np.save(parmt.store + '/DFT/mo_coeff_f.npy', coeff_f)
    k_i = np.load(dft_path + '/k-pts_i.npy')
    k_f = np.load(dft_path + '/k-pts_f.npy')
    np.save(parmt.store + '/k-pts_i.npy', k_i)
    np.save(parmt.store + '/k-pts_f.npy', k_f)
    logger.info("Electronic structure energies updated in files.")

def get_band_indices(dft_params: dict):
    """
    Saves indices of lowest and highest valence and conduction bands to be used in calculations.
    """
    dft_path = 'DFT_resources/' + dft_params['dft_instance']
    mo_occ_i = np.load(dft_path + '/mo_occ_i.npy')

    if not (mo_occ_i == mo_occ_i[0]).all():
        raise NotImplementedError('Occupancy of bands was found to vary with k. Partially filled bands may cause issues determining occupied and unoccupied states, especially since k_f is different from k_i. Check {}/mo_occ_i.npy.'.format(dft_path))
    
    if mo_occ_i[0][0] != 2:
        raise NotImplementedError('Occupancy of filled bands is not 2. Spin-dependent DFT has not been implemented. Check {}/mo_occ_i.npy if you expect filled bands to have an occupancy of 2.'.format(dft_path))
    
    num_bands = mo_occ_i.shape[1]
    num_all_val = sum(mo_occ_i[0] != 0) #total number of occupied bands from dft calculation

    if parmt.numval == 'all':
        num_val = num_all_val
    elif parmt.numval > num_all_val:
        raise Exception('The specified number of valence bands to include ({}) is larger than the number of valence bands obtained from the DFT calculation ({}). Check input parameter "numval" and DFT output file {}/mo_occ_i.npy.'.format(parmt.numval, num_all_val, dft_path))
    else:
        num_val = parmt.numval
    
    if parmt.numcon == 'all':
        num_con = num_bands - num_all_val
    elif parmt.numcon > num_bands - num_all_val:
        raise Exception('The specified number of conduction bands to include ({}) is larger than the number of conduction bands obtained from the DFT calculation ({}). To increase the number of conduction bands, consider using a larger basis set. Check input parameters "numcon" and "mybasis", and DFT output file {}/mo_occ_i.npy.'.format(parmt.numcon, num_bands - num_all_val, dft_path))
    else:
        num_con = parmt.numcon

    ivaltop = num_all_val - 1 #index of the highest valence band included
    ivalbot = ivaltop - num_val + 1 #index of the lowest valence band included
    iconbot = ivaltop + 1 #index of the lowest conduction band included
    icontop = iconbot + num_con - 1 #index of the highest conduction band included

    np.save(parmt.store + '/bands.npy', np.array([ivalbot, ivaltop, iconbot, icontop]))