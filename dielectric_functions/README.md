All hdf5 files contain RPA dielectric functions. Those in "composite" have LFEs included only up to q_{LFE} (see eq. (3.1)).

These files can be used to make `dark_matter_rates.df` (dielectric function) objects. See the example notebook. 

The "full names" of all the dielectric functions are provided below, which include information about their computation. Here is an example of how to read the names:

Ge_cc-pvtz_pbe_10k_7qlfe_qs0.0508919_8k_20qnolfe_qs0.01
    Ge: material - Germanium
    cc-pvtz: Gaussian basis set used in DFT calculation
    pbe: Exchange-correlation functional used in DFT calculation (all use PBE)
    10k_7qlfe_qs0.0508919: for momenta up to 7 ame, we used a 10x10x10 Monkhorst-Pack k-grid, a q-shift of 0.0508919 x (1,1,1) ame, and LFEs were included
    8k_20qnolfe_qs0.01: for momenta up to 20 ame (so 7 < q <= 20), we used an 8x8x8 k-grid, a q-shift of 0.01 x (1,1,1) ame, and LFEs were not included

composite:
diamond_cc-pvtz_pbe_8k_12qlfe_20qnolfe_qs0.10090636
GaAs_cc-pvtz_pbe_8k_7qlfe_qs0.08256101_20qnolfe_qs0.01
Ge_cc-pvtz_pbe_10k_7qlfe_qs0.0508919_8k_20qnolfe_qs0.01
SiC_cc-pvtz_pbe_8k_8qlfe_qs0.08256101_20qnolfe_qs0.01
Si_cc-pv(t+d)z_pbe_8k_8qlfe_25qnolfe_qs0.066286

lfe:
diamond_cc-pvtz_pbe_8k_12q_lfe_qs0.10090636.h5
GaAs_cc-pvtz_pbe_8k_7q_lfe_qs0.06367115.h5
Ge_cc-pvtz_pbe_10k_7q_lfe_qs0.0508919.h5
SiC_cc-pvtz_pbe_8k_8q_lfe_qs0.08256101.h5
Si_cc-pv(t+d)z_pbe_8k_8q_lfe_qs0.066286.h5

nolfe:
diamond_cc-pvtz_pbe_8k_20q_nolfe_qs0.10090636.h5
GaAs_cc-pvtz_pbe_8k_7qnolfe_qs0.06367115_20qnolfe_qs0.01.h5
Ge_cc-pvtz_pbe_10k_7qnolfe_qs0.0508919_8k_20qnolfe_qs0.01.h5
SiC_cc-pvtz_pbe_8k_8qnolfe_qs0.08256101_20qnolfe_qs0.01.h5
Si_cc-pv(t+d)z_pbe_8k_25q_nolfe_qs0.066286.h5

nolfe/low_momentum:
diamond_cc-pvtz_pbe_8k_12q_nolfe_qs0.10090636.h5
GaAs_cc-pvtz_pbe_8k_7q_nolfe_qs0.06367115.h5
Ge_cc-pvtz_pbe_10k_7q_nolfe_qs0.0508919.h5
SiC_cc-pvtz_pbe_8k_8q_nolfe_qs0.08256101.h5