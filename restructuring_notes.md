This describes the purpose of each major file and the functions contained in those files. The main functions are bolded while helper functions are normal text.

### binning.py
Contains functions related to binning epsilon
- spherical_to_cartesian (may not need this)
- cartesian_to_spherical
- construct_theta_bins
- construct_all_solid_angles
- **gen_bin_centers**
- **bin_eps_q**

### cartesian_moments.py
Contains functions for multiprocessing 1D overlaps. May want to move AO to different file
- fact2
- get_E_ijt
- primgauss_1D_overlaps_uR
- AO class

### dark_objects_routines.py
Contains functions that initialize dark_objects dict
- **build_cell_from_input**
- **construct_R_vectors**
- **get_cell_volume**
- **gen_G_vectors**
- gen_all_1D_prim_gauss
- **gen_prim_gauss_indices**
- **gen_all_atomic_orbitals**
- **get_basis_blocks**
- project_vectors_to_1BZ
- **get_1BZ_q_points**
- **primgauss_1D_overlaps**
- get_R_id
- **store_R_ids**

### dielectric_functions.py
The user should be able to obtain results by just running this file after modifying input_parameters.
- initialize_cell
- electronic_structure
- dielectric_RPA
- **main**

### dft_routines.py
Contains functions for performing dft calculations to obtain initial and final energies and wavefunctions with pyscf
- make_kpts
- KS_electronic_structure
- KS_non_self_consistent_field
- convert_to_eV_and_scissor
- get_band_indices

### epsilon_helper.py
Contains functions for non-lfe epsilon calculation which are parallelized over G-vectors
- get_3D_overlaps_blocks
- RPA_susceptibility
- RPA_dielectric
- RPA_dielectric_lfe (may move to own file later)

### epsilon_routines.py
Calls helper functions in epsilon_helper to calculate epsilon_RPA and calculates eps(|q|,E)
- initialize_RPA_dielectric
- get_energy_diff
- get_binned_epsilon
- **epsilon_r**

### input_parameters.py
User-input parameters for dielectric function calculation

### routines.py
General functions used by other modules. Also initializes logging
- time_wrapper
- makedir
- get_all_unique_nums_in_array
- get_all_unique_vectors_in_array - delete?
- load_unique_R

### utils.py
Compatibility checks and input parameter checks
- check_requirements
- patch
- check_inputs
