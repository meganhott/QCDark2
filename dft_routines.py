import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc

import input_parameters as parmt
from routines import logger, time_wrapper, makedir

#Constants
me = 0.51099895000e6 #eV
alpha = 1/137

def make_kpts(cell: pbcgto.cell.Cell, with_gamma: bool = True) -> pyscf.pbc.lib.kpts.KPoints:
    """
    Function to get the grid in reciprocal unit cell given k_grid density in input_parameters.py.
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
    Returns:
        k_grid: np.ndarray of shape (N, 3), k vectors generated from the cell.
    """
    k_grid = parmt.k_grid
    kpts = cell.make_kpts(k_grid, wrap_around=True, with_gamma_point=with_gamma, space_group_symmetry=True)
    np.save(parmt.store + '/k-pts_i', kpts.kpts)
    logger.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_i.npy', k_grid[0], k_grid[1], k_grid[2]))
    return kpts

@time_wrapper
def KS_electronic_structure(cell: pbcgto.cell.Cell) -> pbcdft.krks_ksymm.KsymAdaptedKRKS:
    """
    Function to do density functional theory. Solves the RKS if only one kpoint in kgrid, otherwise solves RKS at each k-point and constructs density matrix from integrating over 1BZ. 
    Inputs:
        cell:   pyscf.pbc.gto.cell.Cell object, initialized in build_cell_from_input routine.
        kpts:   np.ndarray consisting of k-points. 
                If None, kgrid is generated using parameters input in parmt.k_grid
    Returns:
        None
    Saves:
        molecular energies, molecular coefficients and molecular occupation numbers.
    """
    dft_path = parmt.store + '/DFT/'
    makedir(dft_path)
    logger.info('Initial state calculation:')
    kpts = make_kpts(cell, True)
    kmf = pbcdft.KRKS(cell, kpts).density_fit()
    kmf.xc = parmt.xcfunc
    kmf.kernel()
    if kmf.converged:
        np.save(dft_path + 'mo_en_i_dft', kpts.transform_mo_energy(kmf.mo_energy))
        np.save(dft_path + 'mo_coeff_i', kpts.transform_mo_coeff(kmf.mo_coeff))
        np.save(dft_path + 'mo_occ_i', kpts.transform_mo_occ(kmf.mo_occ))
    else:
        raise ValueError('DFT not converged. Might need to orthogonalize basis before continuing (Not Implemented).')
    logger.info('Electronic structure converged, KS energy is {:.2f} Hartrees.\n\tDFT data is stored to {}.'.format(kmf.e_tot, dft_path))
    return kmf

@time_wrapper
def KS_non_self_consistent_field(kmf: pbcdft.krks_ksymm.KsymAdaptedKRKS) -> None:
    """
    Non self consistent field calculation for final states.
    Inputs:
        kmf:    pyscf.pbc.dft.krks_ksymm.KsymAdaptedKRKS object
    Return:
        None
    """
    logger.info("Final State Calculation:")
    dft_path = parmt.store + '/DFT/'
    q_shift_dir = np.array(parmt.q_shift_dir)
    q_shift = parmt.q_shift*q_shift_dir/np.linalg.norm(q_shift_dir)
    scaled_center = kmf.cell.get_scaled_kpts(q_shift)
    kpts = kmf.cell.make_kpts(parmt.k_grid, space_group_symmetry=True, wrap_around = True, scaled_center = scaled_center)
    np.save(parmt.store + '/k-pts_f', kpts.kpts)
    logger.info("Selected q shift = {}".format(np.array2string(q_shift, precision = 5)))
    logger.info("{} k vectors generated, {} in irreducible BZ, and stored to \'{}\' given k-grid:\n\tnk_x = {}, nk_y = {}, nk_z = {}.".format(kpts.nkpts, kpts.nkpts_ibz, parmt.store + '/k-pts_f.npy', parmt.k_grid[0], parmt.k_grid[1], parmt.k_grid[2]))
    ek , ck = kmf.get_bands(kpts.kpts_ibz)
    ek = kpts.transform_mo_energy(ek)
    ck = kpts.transform_mo_coeff(ck)
    np.save(dft_path + 'mo_en_f_dft', ek)
    np.save(dft_path + 'mo_coeff_f', ck)
    logger.info('Non self consistent field equations solved for final state k-points. Data is stored to {}.'.format(dft_path))
    return None

@time_wrapper
def convert_to_eV_and_scissor(cell: pbcgto.cell.Cell) -> None:
    """
    Function converts energies from Ryd to eV - prints bandgap, if scissor - scissor corrects bandgap.
    Inputs:
        cell:                               pyscf.pbc.gto.cell.Cell object
    Reads:
        parmt.scissor_bandgap:              float, scissor energies in eV
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    Writes:
        parmt.store + '/DFT/mo_en_i.npy':   np.ndarray object, stored to disk
        parmt.store + '/DFT/mo_en_f.npy':   np.ndarray object, stored to disk
    """
    occ_orb = cell.tot_electrons()//2
    en_i = np.load(parmt.store + '/DFT/mo_en_i_dft.npy')*alpha*alpha*me
    en_f = np.load(parmt.store + '/DFT/mo_en_f_dft.npy')*alpha*alpha*me
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
    np.save(parmt.store + '/DFT/mo_en_i.npy', en_i)
    np.save(parmt.store + '/DFT/mo_en_f.npy', en_f)
    logger.info("Electronic structure energies updated in files.")
    return

def get_band_indices():
    """
    Saves indices of lowest and highest valence and conduction bands to be used in calculations.
    """
    dft_path = parmt.store + '/DFT/'
    mo_occ_i = np.load(dft_path + 'mo_occ_i.npy')

    if not (mo_occ_i == mo_occ_i[0]).all():
        raise NotImplementedError('Occupancy of bands was found to vary with k. Partially filled bands may cause issues determining occupied and unoccupied states, especially since k_f is different from k_i. Check mo_occ_i.npy.')
    
    if mo_occ_i[0][0] != 2:
        raise NotImplementedError('Occupancy of filled bands is not 2. Spin-dependent DFT has not been implemented. Check mo_occ_i.npy if you expect filled bands to have an occupancy of 2.')
    
    num_bands = mo_occ_i.shape[1]
    num_all_val = sum(mo_occ_i[0] != 0) #total number of occupied bands from dft calculation

    if parmt.numval == 'all':
        num_val = num_all_val
    elif parmt.numval > num_all_val:
        raise Exception(f'The specified number of valence bands to include ({parmt.numval}) is larger than the number of valence bands obtained from the DFT calculation ({num_all_val}). Check input parameter "numval" and DFT output file mo_occ_i.npy.')
    else:
        num_val = parmt.numval
    
    if parmt.numcon == 'all':
        num_con = num_bands - num_all_val
    elif parmt.numcon > num_bands - num_all_val:
        raise Exception(f'The specified number of conduction bands to include ({parmt.numcon}) is larger than the number of conduction bands obtained from the DFT calculation ({num_bands - num_all_val}). To increase the number of conduction bands, consider using a larger basis set. Check input parameters "numcon" and "mybasis", and DFT output file mo_occ_i.npy.')
    else:
        num_con = parmt.numcon

    ivaltop = num_all_val-1 #index of the highest valence band included
    ivalbot = ivaltop-num_val+1 #index of the lowest valence band included
    iconbot = ivaltop+1 #index of the lowest conduction band included
    icontop = iconbot+num_con-1 #index of the highest conduction band included

    np.save(parmt.store + '/bands.npy', np.array([ivalbot, ivaltop, iconbot, icontop]))
