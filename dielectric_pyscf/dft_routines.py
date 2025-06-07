import numpy as np
import os
import json
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc
import pyscf.scf.addons # for basis sets with linear dependencies

import pyscf.lib

import dielectric_pyscf.input_parameters as parmt
from dielectric_pyscf.routines import logger, time_wrapper, makedir

# Constants
me = 0.51099895000e6 # eV
alpha = 1/137

def save_dft():
    """
    Checks saved DFT calculations and determines if a new calculation should be run.

    Returns:
        new_dft (bool):
            If True, a new DFT calculation will be run.
        dft_params (dict):
            DFT parameters specified in input_parameters.
    """
    dft_params = {
        'lattice_vectors': parmt.lattice_vectors,
        'atomloc': parmt.atomloc,
        'mybasis': parmt.mybasis,
        'orth': parmt.orth,
        'density_fitting': parmt.density_fitting,
        'effective_core_potential': parmt.effective_core_potential,
        'pseudo': parmt.pseudo,
        'precision': parmt.precision,
        'xcfunc': parmt.xcfunc,
        'k_grid': parmt.k_grid,
        'q_shift_dir': parmt.q_shift_dir,
        'q_shift': parmt.q_shift,
        'dft_instance': None
    }

    dft_path = parmt.DFT_resources_path + '/DFT_resources'
    makedir(dft_path) # makes DFT_routines directory if it does not already exist

    dft_instances = os.listdir(dft_path)
    for d in dft_instances:
        dft_dict = json.load(open(f'{dft_path}/{d}/dft_params.txt', 'r'))
        if all(dft_dict[key] == dft_params[key] for key in dft_dict.keys() if key not in ['dft_instance']): # compare everything except dft_instance
            dft_params['dft_instance'] = d
            dft_files = os.listdir(dft_path + '/' + d)
            if len(dft_files) < 4: # something went wrong with previously calculated DFT, calculation should be started again
                logger.info(f'There is not a stored DFT cacluation for these input parameters, a new calculation will be performed and stored as {d}.')
                new_dft = True
            elif ('mo_en_i_dft.npy' in dft_files) and ('mo_en_f_dft.npy' not in dft_files):
                logger.info(f'A SCF calculation has been performed for these input parameters as stored as {d} but must be redone for the NSCF calculation')
                new_dft = True
            else:
                logger.info(f'DFT already calculated for these input parameters and stored as {d}')
                new_dft = False
            return new_dft, dft_params
    
    if dft_params['dft_instance'] is None:
        if len(dft_instances) == 0:
            dft_params['dft_instance'] = 'DFT_0'
        else:
            dft_params['dft_instance'] = 'DFT_' + str(max([int(d.split('_')[1]) for d in dft_instances]) + 1)

        logger.info(f'There is not a stored DFT cacluation for these input parameters, a new calculation will be performed and stored as {dft_params["dft_instance"]}.')
        makedir(f'{dft_path}/{dft_params["dft_instance"]}')
        json.dump(dft_params, open(f'{dft_path}/{dft_params["dft_instance"]}/dft_params.txt', 'w')) # save dft parameters
        new_dft = True

    return new_dft, dft_params

def list_saved_dft(df=False):
    dft_path = parmt.DFT_resources_path + '/DFT_resources'
    dft_instances = os.listdir(dft_path)
    if df: #return pandas dataframe
        import pandas as pd
        #Work on making this look better
        for i, d in enumerate(dft_instances):
            dft_dict = json.load(open(f'{dft_path}/{d}/dft_params.txt', 'r'))
            dft_dict['lattice_vectors'] = str(dft_dict['lattice_vectors'])
            dft_dict['mybasis'] = str(dft_dict['mybasis'])
            dft_dict['k_grid'] = str(dft_dict['k_grid'])
            dft_dict['q_shift_dir'] = str(dft_dict['q_shift_dir'])

            if i == 0:
                df = pd.DataFrame(data=dft_dict)
            else:
                df = pd.concat([df, pd.DataFrame(data=dft_dict)], ignore_index=True)

    else: #print all dft params
        for d in dft_instances:
            dft_dict = json.load(open(f'{dft_path}/{d}/dft_params.txt', 'r'))
            print(f'DFT Instance: {dft_dict["dft_instance"]}')
            print(f'\tLattice vectors: {dft_dict["lattice_vectors"]}')
            print(f'\tAtom Locations: {dft_dict["atomloc"]}')
            print(f'\tBasis: {dft_dict["mybasis"]}')
            print(f'\tEffective Core Potential: {dft_dict["effective_core_potential"]}')
            print(f'\tPseudopotential: {dft_dict["pseudo"]}')
            print(f'\tPrecision: {dft_dict["precision"]}')
            print(f'\tExchange Correlation Functional: {dft_dict["xcfunc"]}')
            print(f'\tk-grid: {dft_dict["k_grid"]}'.format(dft_dict["k_grid"]))
            print(f'\tq Shift Direction: {dft_dict["q_shift_dir"]}')
            print(f'\tq shift: {dft_dict["q_shift"]}')

def make_kpts(cell: pbcgto.cell.Cell, dft_params: dict) -> pyscf.pbc.lib.kpts.KPoints:
    """
    Function to get the grid in reciprocal unit cell given k_grid density in input_parameters.py.

    Inputs:
        cell (pyscf.pbc.gto.cell.Cell):
            Initialized in build_cell_from_input routine
    Returns:
        k_i ((N_k, 3) np.ndarray): 
            Inital k-vectors
        k_f ((N_k, 3) np.ndarray): 
            Final k-vectors
    """
    # Save k_i
    k_grid = parmt.k_grid
    kpts_i = cell.make_kpts(k_grid, wrap_around=True) # wrap_around=True generates k_pts in 1BZ

    dft_path = parmt.DFT_resources_path + '/DFT_resources/' + dft_params['dft_instance']
    np.save(dft_path + '/k-pts_i', kpts_i)

    logger.info(f'{kpts_i.shape[0]} initial k vectors generated and stored to {parmt.store}/k-pts_i.npy given k-grid:\n\tnk_x = {k_grid[0]}, nk_y = {k_grid[1]}, nk_z = {k_grid[2]}.')

    # Save k_f
    q_shift_dir = np.array(dft_params['q_shift_dir'])
    q_shift = dft_params['q_shift'] * q_shift_dir / np.linalg.norm(q_shift_dir)

    logger.info(f'Selected q shift = {np.array2string(q_shift, precision = 5)}')

    scaled_center = cell.get_scaled_kpts(q_shift)

    kpts_f = cell.make_kpts(dft_params['k_grid'], wrap_around=True, scaled_center=scaled_center)
    np.save(dft_path + '/k-pts_f', kpts_f)

    logger.info(f'{kpts_f.shape[0]} final k vectors generated and stored to {parmt.store}/k-pts_f.npy given k-grid:\n\tnk_x = {dft_params["k_grid"][0]}, nk_y = {dft_params["k_grid"][1]}, nk_z = {dft_params["k_grid"][2]}.')

    # Using symmetry breaks many DFT calculations, no longer implemented
    """
    kpts_i = cell.make_kpts(k_grid, wrap_around=True, with_gamma_point=with_gamma, space_group_symmetry=space_group_symmetry)
    kpts_f = cell.make_kpts(dft_params['k_grid'], space_group_symmetry=space_group_symmetry, wrap_around = True, scaled_center = scaled_center)

    if space_group_symmetry:
        np.save(dft_path + '/k-pts_i', kpts_i.kpts)
        logger.info(f'{kpts_i.nkpts} initial k vectors generated, {kpts_i.nkpts_ibz} in irreducible BZ, and stored to {parmt.store}/k-pts_i.npy given k-grid:\n\tnk_x = {k_grid[0]}, nk_y = {k_grid[1]}, nk_z = {k_grid[2]}.')
        np.save(dft_path + '/k-pts_f', kpts_f.kpts)
        logger.info(f'{kpts_f.nkpts} final k vectors generated, {kpts_f.nkpts_ibz} in irreducible BZ, and stored to {parmt.store}/k-pts_f.npy given k-grid:\n\tnk_x = {dft_params["k_grid"][0]}, nk_y = {dft_params["k_grid"][1]}, nk_z = {dft_params["k_grid"][2]}.')
    """
    return kpts_i, kpts_f

@time_wrapper
def KS_electronic_structure(cell: pbcgto.cell.Cell, dft_params: dict, orth=parmt.orth, cderi_save_file=None, density_fitting=parmt.density_fitting) -> pbcdft.krks_ksymm.KsymAdaptedKRKS:
    """
    Function to do density functional theory. Performs restricted Kohn-Sham (RKS) DFT at each k-point and constructs density matrix from integrating over 1BZ. 

    Inputs:
        cell (pyscf.pbc.gto.cell.Cell):
            Initialized in build_cell_from_input routine, contains information about material
        dft_params (dict):
            DFT parameters
        CholOrth (bool):
            Determines if scf calculation will eliminate linear dependencies with Cholesky orthogonalization - useful for basis sets with diffuse functions
    Returns:
        kmf (pyscf.pbc.dft.krks.KRKS):
            SCF DFT object
    Saves:
        molecular energies, molecular coefficients and molecular occupation numbers.
    """
    dft_path = parmt.DFT_resources_path + '/DFT_resources/' + dft_params['dft_instance']
    logger.info('Initial state calculation:')
    kpts = make_kpts(cell, dft_params)[0]

    if density_fitting == 'GDF': # Gaussian density fitting
        kmf = pbcdft.KRKS(cell, kpts).density_fit()
    elif density_fitting == 'MDF': # Mixed density fitting
        kmf = pbcdft.KRKS(cell, kpts).mix_density_fit()
    elif density_fitting == 'RSDF': # Range separated density fitting
        kmf = pbcdft.KRKS(cell, kpts).rs_density_fit()
    elif density_fitting == 'FFTDF': # Fast Fourier transform density fitting. Do not use for all-electron calculations due to memory constraints!
        kmf = pbcdft.KRKS(cell, kpts)
    else:
        raise(ValueError('density_fitting must be "GDF" for Gaussian Density Fitting, "MDF" for Mixed Density Fitting, "RSDF" for Range-Separated Density Fitting, or "FFTDF" for Fast Fourier Transform Density Fitting. FFTDF is not supported for all-electron calculations due to memory constraints.'))

    kmf.xc = parmt.xcfunc
    kmf.chkfile = f'{dft_path}/kmf.chk' # save checkfile

    #kmf.exxdiv = exxdiv #None or ewald - may need to change to None for hybrid functionals? exxdiv by default

    if cderi_save_file is not None: # Saves density fitting matrix
        kmf.with_df._cderi_to_save = cderi_save_file
    #else:
    #    kmf.with_df._cderi_to_save = f'{parmt.store}/cderi.h5'

    if orth: #eventually catch linear dependency errors automatically and run this?
        logger.info('Molecular orbital basis will be orthogonalized in SCF calculation to remove linear dependence and improve convergence.')
        kmf = pyscf.scf.addons.remove_linear_dep_(kmf).run()
        # pyscf will still output warning even after this is applied but linear dependencies have been removed
    else:
        kmf.kernel()
    
    if kmf.converged:
        if orth:
            #remove highest bands/mo orbitals of some k-points so that all k-points have same number of bands
            N_MO_min = min([en_k.shape[0] for en_k in kmf.mo_energy])
            logger.info(f'Orthogonalized SCF resulted in {N_MO_min} bands/molecular orbitals from {cell.nao} atomic orbitals.')
            mo_energy = np.array([en_k[:N_MO_min] for en_k in kmf.mo_energy])
            mo_coeff = np.array([coeff_k[:,:N_MO_min] for coeff_k in kmf.mo_coeff])
            mo_occ = np.array([occ_k[:N_MO_min] for occ_k in kmf.mo_occ])
        else:
            #otherwise all k-points will have same number of MO
            mo_energy = kmf.mo_energy
            mo_coeff = kmf.mo_coeff
            mo_occ = kmf.mo_occ
    
        np.save(dft_path + '/mo_en_i_dft.npy', mo_energy)
        np.save(dft_path + '/mo_coeff_i.npy', mo_coeff)
        np.save(dft_path + '/mo_occ_i.npy', mo_occ)
    else:
        raise ValueError('DFT not converged. Might need to orthogonalize basis by setting CholOrth=True.')
    
    logger.info(f'Electronic structure converged, KS energy is {kmf.e_tot:.2f} Hartrees.\n\tDFT data is stored to {dft_path}.')

    return kmf

@time_wrapper
def KS_non_self_consistent_field(kmf: pbcdft.krks_ksymm.KsymAdaptedKRKS, dft_params: dict):
    """
    Non self consistent field calculation for final states.

    Inputs:
        kmf (pyscf.pbc.dft.krks.KRKS):
            SCF DFT object
        dft_params (dict):
            DFT parameters
    """
    logger.info('Final State Calculation:')
    kpts = make_kpts(kmf.cell, dft_params)[1]

    energy_k, coeff_k = kmf.get_bands(kpts)

    # Using symmetry breaks many DFT calculations, no longer implemented
    """
    enegy_k , coeff_k = kmf.get_bands(kpts.kpts_ibz)
    energy_k = kpts.transform_mo_energy(ek)
    coeff_k = kpts.transform_mo_coeff(ck)
    """

    dft_path = parmt.DFT_resources_path + '/DFT_resources/' + dft_params['dft_instance']
    np.save(dft_path + '/mo_en_f_dft.npy', energy_k)
    np.save(dft_path + '/mo_coeff_f.npy', coeff_k)

    logger.info(f'Non self consistent field equations solved for final state k-points. Data is stored to {dft_path}.')

@time_wrapper
def convert_to_eV_and_scissor(cell: pbcgto.cell.Cell, dft_params: dict):
    """
    Converts energies from Ryd to eV and prints bandgap. If scissor-corrected bandgap is specified in input_parameters, it is applied to MO energies.

    Inputs:
        cell (pyscf.pbc.gto.cell.Cell):
            Contains information about material
        dft_params (dict):
            DFT parameters
    """
    dft_path = parmt.DFT_resources_path + '/DFT_resources/' + dft_params['dft_instance']
    mo_en_i = np.load(dft_path + '/mo_en_i_dft.npy')*alpha*alpha*me # molecular orbital energies for initial k-points, convert to eV
    mo_en_f = np.load(dft_path + '/mo_en_f_dft.npy')*alpha*alpha*me # molecular orbital energies for final k-points, convert to eV

    occ_orb = cell.tot_electrons()//2 # number of occupied orbitals 
    homo = max(mo_en_i[:,:occ_orb].max(), mo_en_f[:,:occ_orb].max()) # energy of highest occupied MO
    mo_en_i, mo_en_f = mo_en_i - homo, mo_en_f - homo # shifting bands so HOMO is at 0 eV
    lumo = min(mo_en_i[:,occ_orb:].min(), mo_en_f[:,occ_orb:].min()) #energy of lowest unoccupied MO

    logger.info(f'All energies converted to eV. Calculated Bandgap = {lumo:.2f} eV.')

    # Apply scissor correction by shifting all conduction bands
    if parmt.scissor_bandgap is not None:
        if type(parmt.scissor_bandgap) != float:
            raise ValueError('Parameter scissor_bandgap in input_parameters.py must be either None or of type float.')
        
        correction = parmt.scissor_bandgap - lumo

        mo_en_i[:,occ_orb:], mo_en_f[:,occ_orb:] = mo_en_i[:,occ_orb:] + correction, mo_en_f[:,occ_orb:] + correction
        logger.info(f'Scissor Correction applied, new bandgap is {parmt.scissor_bandgap:.2f} eV.')
    
    makedir(parmt.store + '/DFT')
    np.save(parmt.store + '/DFT/mo_en_i.npy', mo_en_i)
    np.save(parmt.store + '/DFT/mo_en_f.npy', mo_en_f)

    # Store MO coefficients and k-points in main file as well
    mo_coeff_i = np.load(dft_path + '/mo_coeff_i.npy')
    mo_coeff_f = np.load(dft_path + '/mo_coeff_f.npy')
    np.save(parmt.store + '/DFT/mo_coeff_i.npy', mo_coeff_i)
    np.save(parmt.store + '/DFT/mo_coeff_f.npy', mo_coeff_f)

    k_i = np.load(dft_path + '/k-pts_i.npy')
    k_f = np.load(dft_path + '/k-pts_f.npy')
    np.save(parmt.store + '/k-pts_i.npy', k_i)
    np.save(parmt.store + '/k-pts_f.npy', k_f)

    logger.info('Electronic structure energies updated in files.')

def get_band_indices(dft_params: dict):
    """
    Saves indices of lowest and highest valence and conduction bands to be used in calculations.
    """
    dft_path = parmt.DFT_resources_path + '/DFT_resources/' + dft_params['dft_instance']
    mo_occ_i = np.load(dft_path + '/mo_occ_i.npy')

    if not (mo_occ_i == mo_occ_i[0]).all():
        raise NotImplementedError(f'Occupancy of bands was found to vary with k. Partially filled bands may cause issues determining occupied and unoccupied states, especially since k_f is different from k_i. Check {dft_path}/mo_occ_i.npy.')
    
    if mo_occ_i[0][0] != 2:
        raise NotImplementedError(f'Occupancy of filled bands is not 2. Spin-dependent DFT has not been implemented. Check {dft_path}/mo_occ_i.npy if you expect filled bands to have an occupancy of 2.')
    
    num_bands = mo_occ_i.shape[1]
    num_all_val = sum(mo_occ_i[0] != 0) # total number of occupied bands from dft calculation
    num_all_con = num_bands - num_all_val

    if parmt.numval == 'all':
        num_val = num_all_val
    elif parmt.numval == 'auto':
        # Remove valence bands that will not contribute to dielectric function up to energy parmt.E_max
        if parmt.numcon != 'auto':
            raise Exception('If you want to automatically exclude irrelevant conduction band valence bands, both "numval" and "numcon" must be set to "auto" in input_parameters.')
        else:
            mo_en_i = np.load(parmt.store + '/DFT/mo_en_i.npy')
            mo_en_f = np.load(parmt.store + '/DFT/mo_en_f.npy')
            mo_en = np.concatenate((mo_en_i, mo_en_f), axis=0)

            mo_en_min = np.min(mo_en, axis=0) # min and max energies of each band
            mo_en_max = np.max(mo_en, axis=0)
            dif = mo_en_min[None, num_all_val:] - mo_en_max[:num_all_val, None] #(val, cond) # minimum energy differences for each pair of valence and conduction bands

            locs = np.where(dif > parmt.E_max) # indices of (val, cond) pairs that exceed the maximum energy difference

            bands_to_discard = np.ones(num_bands, dtype='bool')
            for i in range(num_all_val):
                if np.count_nonzero(locs[0] == i) == num_all_con:
                    # If the transition energy to all conduction bands is larger than E_max, we can discard this valence band
                    bands_to_discard[i] = False
            for i in range(num_all_con):
                if np.count_nonzero(locs[1] == i) == num_all_val:
                    # If the transition energy to all valence bands is larger than E_max, we can discard this conduction band
                    bands_to_discard[i+num_all_val] = False

            # discard these bands and update number of inluded valence and conduction bands

            np.save(parmt.store + '/DFT/mo_en_i.npy', mo_en_i[:, bands_to_discard])
            np.save(parmt.store + '/DFT/mo_en_f.npy', mo_en_f[:, bands_to_discard])

            np.save(parmt.store + '/DFT/mo_coeff_i.npy', np.load(parmt.store + '/DFT/mo_coeff_i.npy')[:,:,bands_to_discard])
            np.save(parmt.store + '/DFT/mo_coeff_f.npy', np.load(parmt.store + '/DFT/mo_coeff_f.npy')[:,:,bands_to_discard])

            new_num_all_val = np.count_nonzero(bands_to_discard[:num_all_val])
            new_num_all_con = np.count_nonzero(bands_to_discard[num_all_val:])

            logger.info(f'\nThe following bands have been automatically excluded from calculations since valence to conduction energy differences are greater than E_max = {parmt.E_max} eV:\n{np.arange(num_bands)[np.invert(bands_to_discard)]}\nThe number of valence bands has been reduced from {num_all_val} to {new_num_all_val} and the number of conduction bands has been reduced from {num_all_con} to {new_num_all_con}.\n')

            num_all_val = num_val = new_num_all_val
            num_all_con = num_con = new_num_all_con

    elif parmt.numval > num_all_val:
        raise Exception(f'The specified number of valence bands to include ({parmt.numval}) is larger than the number of valence bands obtained from the DFT calculation ({num_all_val}). Check input parameter "numval" and DFT output file {dft_path}/mo_occ_i.npy.')
    else:
        num_val = parmt.numval
    
    if parmt.numcon == 'all':
        num_con = num_all_con
    elif parmt.numcon == 'auto':
        if parmt.numval != 'auto':
            raise Exception('If you want to automatically exclude irrelevant conduction band valence bands, both "numval" and "numcon" must be set to "auto" in input_parameters.')
        pass # already calculated if numval and numcon are both auto

    elif parmt.numcon > num_bands - num_all_val:
        raise Exception(f'The specified number of conduction bands to include ({parmt.numcon}) is larger than the number of conduction bands obtained from the DFT calculation ({num_bands - num_all_val}). To increase the number of conduction bands, consider using a larger basis set. Check input parameters "numcon" and "mybasis", and DFT output file {dft_path}/mo_occ_i.npy.')
    else:
        num_con = parmt.numcon

    ivaltop = num_all_val - 1 #index of the highest valence band included
    ivalbot = ivaltop - num_val + 1 #index of the lowest valence band included
    iconbot = ivaltop + 1 #index of the lowest conduction band included
    icontop = iconbot + num_con - 1 #index of the highest conduction band included

    np.save(parmt.store + '/bands.npy', np.array([ivalbot, ivaltop, iconbot, icontop]))