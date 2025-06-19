
import numpy as np
from uncertainties import ufloat

# Integrated spectrum
n_re = 1.5  # 1.5*Re

min_SN = 4
min_fwhm, max_fwhm = 50, 500. # km / s
rad_bin = np.array([0, 0.5, 1.0, 1.5])
# rad_bin = np.array([0, 0.8, 1.6])
bin_size = np.diff(rad_bin)[0]

vel_shift = True   # Shifted based on rotational velocities to enhance SNR

# Dust dereddening when Balmer decrement is not applicable
integ_Av = ufloat(.38, .15)   # From all galaxies
if len(rad_bin) - 1 == 3:
    sp_res_Av = [ufloat(.62, .16), ufloat(.4, .2), ufloat(0., .2)]
else:
    sp_res_Av = [ufloat(.65, .19), ufloat(0, .2)]

# Emission line fitting
nch = 2048
dlambda_dict = {'IZ': 0.000151367217767984,
                'ZY': 0.000175292952917516, 'Y': 0.000175292952917516, 'YJ': 0.000175292952917516,
                'H':  0.000215820327866822, 'HK': 0.000215820327866822, 'K': 0.000215820327866822}
wl_range_dict = {'IZ': 0.779999971389771 + np.arange(nch) * dlambda_dict['IZ'],
                 'ZY': 1.00000000000000 + np.arange(nch) * dlambda_dict['ZY'],
                 'Y':  1.00000000000000 + np.arange(nch) * dlambda_dict['ZY'],
                 'YJ': 1.00000000000000 + np.arange(nch) * dlambda_dict['YJ'],
                 'H':  1.42499995231628 + np.arange(nch) * dlambda_dict['H'],
                 'HK': 1.42499995231628 + np.arange(nch) * dlambda_dict['H'],
                 'K':  1.42499995231628 + np.arange(nch) * dlambda_dict['H']}
band_ax_dict = {'IZ': 0,
                'ZY': 1, 'Y':  1, 'YJ': 2,
                'H':  3, 'HK': 3, 'K':  3}


emis_major_dict = {'IZ': r'[OII]$\lambda\lambda$3727,9',
                   'ZY': r'H$\gamma$', 'Y': r'[OIII]$\lambda$4363', 'YJ': r'H$\beta$',
                   'H':  r'H$\alpha$', 'HK': r'[SII]$\lambda$6717', 'K': r'[SII]$\lambda$6731'}
emis_minor1_dict = {'IZ': ' ',
                    'ZY': ' ', 'Y': ' ', 'YJ': r'[OIII]$\lambda$5007',
                    'H':  r'[NII]$\lambda$6584', 'HK': ' ', 'K': ' '}
emis_minor2_dict = {'IZ': ' ',
                    'ZY': ' ', 'Y': ' ', 'YJ': r'[OIII]$\lambda$4959',
                    'H':  r'[NII]$\lambda$6548', 'HK': ' ', 'K': ' '}

# in the unit of micro
spec_width_dict = {'IZ': (-0.0150, 0.0300),
                   'ZY': (-0.0100, 0.0100), 'Y':  (-0.0100, 0.0100), 'YJ': (-0.0400, 0.0500),
                   'H':  (-0.0250, 0.0250), 'HK': (-0.0250, 0.0250), 'K':  (-0.0250, 0.0250)}
fit_width_dict =  {'IZ': (-0.0040, 0.0040),
                   'ZY': (-0.0100, 0.0000), 'Y':  (-0.0000, 0.0100), 'YJ': (-0.0050, 0.0500),
                   'H':  (-0.0100, 0.0100), 'HK': (-0.0020, 0.0060), 'K':  (-0.0020, 0.0060)}

min_sky_weight = .8

# Stacking the spectrum
dlambda = 5e-5
std_wl_dict = {'IZ': np.arange(0.3650, 0.3800, dlambda), 'ZY': np.arange(0.4300, 0.4350, dlambda),
               'Y':  np.arange(0.4350, 0.4450, dlambda), 'YJ': np.arange(0.4515, 0.5115, dlambda),
               'H':  np.arange(0.6420, 0.6650, dlambda), 'HK': np.arange(0.6650, 0.6724, dlambda),
               'K':  np.arange(0.6724, 0.6800, dlambda)}

