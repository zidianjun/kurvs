from config import bin_size

from astropy.io import fits
import numpy as np

unit = 1e16  # erg/s/cm^2/micro
light_speed = 299792.458  # km / s

restframe_dict = {' ':                             np.nan,
                  r'[OII]$\lambda$3727':           0.3726032,
                  r'[OII]$\lambda$3729':           0.3728815,
                  r'[OII]$\lambda\lambda$3727,9':  0.3727424,
                  r'H$\gamma$':                    0.4340471,
                  r'[OIII]$\lambda$4363':          0.4363210,
                  r'H$\beta$':                     0.4861333,
                  r'[OIII]$\lambda$4959':          0.4958911,
                  r'[OIII]$\lambda$5007':          0.5006843,
                  r'[NII]$\lambda$6548':           0.6548050,
                  r'H$\alpha$':                    0.6562819,
                  r'[NII]$\lambda$6584':           0.6583460,
                  r'[SII]$\lambda$6717':           0.6716440,
                  r'[SII]$\lambda$6731':           0.6730810}


DPline_ratio = 3.
caseBHaHb = 2.86

instr_sig = 0.00015  # micro, from KMOS H-band spectrum.  bootstrap median: 0.0001584

arcsec = 4.848e-3  # 1 arcsec = 4.848e-6 rad = 4.848e-3 kpc / Mpc = 4.848 pc / Mpc
arcsec_per_pix = .1
KMOS_FWHM_arcsec = .6
KMOS_FWHM_pix = KMOS_FWHM_arcsec / arcsec_per_pix
KMOS_diameter = 3   # arcsec

HST_arcsec_per_pix = 60e-3
HST_KMOS_outer_size = int(KMOS_diameter / 2. / HST_arcsec_per_pix)
# # 1 pix in HST image is 60 mas, KMOS FoV is ~2.8 arcsec, 3/0.06=25
HST_KMOS_bin_pix_size = int(bin_size / HST_arcsec_per_pix)

JWST_arcsec_per_pix = 30e-3
JWST_KMOS_outer_size = int(KMOS_diameter / 2. / JWST_arcsec_per_pix)
# # 1 pix in JWST image is 60 mas, KMOS FoV is ~2.8 arcsec, 3/0.06=25
JWST_KMOS_bin_pix_size = int(bin_size / JWST_arcsec_per_pix)


Zsun = 8.69

