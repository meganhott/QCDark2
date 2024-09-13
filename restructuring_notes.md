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
- get_E_ijt
- primgauss_1D_overlaps_uR
- AO class

### dielectric_functions.py
The user should be able to obtain results by just running this file after modifying input_parameters.
- initialize_cell
- electronic_structure
- dielectric_RPA
- **main**

### epsilon_routines.py
Contains functions for non-lfe epsilon calculation which are parallelized over G-vectors
- get_3D_overlaps_blocks
- RPA_susceptibility
- **RPA_dielectric**
- **RPA_dielectric_lfe** (may move to own file later)

### input_parameters.py
User-input parameters for dielectric function calculation

### routines.py