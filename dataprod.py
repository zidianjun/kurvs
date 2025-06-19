
from paths import *
import constant as c
from config import *
from specs import SpecFit
import utils as u
from diagnostics import MetDiag

from astropy.cosmology import Planck13 as cosmo
import numpy as np
from os.path import isfile
from scipy.interpolate import interp1d
import operator
ops = {'+': operator.add, '-': operator.sub}
import matplotlib as mpl
import matplotlib.pyplot as plt




class DataProd(object):

    def __init__(self, gal_name):
        self.gal_name = gal_name
        self.field = u.get_field(gal_name)
        if self._read_band('IZ'):
            cubeIZ = u.read_spec(gal_name, band='IZ')
            self.cubeIZ = u.sky_substract(cubeIZ)
        if self._read_band('YJ'):
            cubeYJ = u.read_spec(gal_name, band='YJ')
            self.cubeYJ = u.sky_substract(cubeYJ)
        if self._read_band('H'):
            cubeH = u.read_spec(gal_name, band='H')
            self.cubeH = u.sky_substract(cubeH)
        self.redshift, self.Re = u.read_phot(gal_name, 'redshift'), u.read_phot(self.gal_name, 're')
        self.kpc_per_arc = cosmo.angular_diameter_distance(self.redshift).value * c.arcsec
        self.kpc_per_pix = self.kpc_per_arc * c.arcsec_per_pix
        x = rad_bin[:-1] + .5 * np.diff(rad_bin)[0]
        self.x = x * self.kpc_per_arc
        

    def _read_band(self, band):
        return u.read_band(self.gal_name, band)

    def _get_cube(self, band):
        if not self._read_band(band):
            raise ValueError('%s does not have observations in %s band!' %(self.gal_name, band))
        else:
            return getattr(self, 'cube' + u.get_band(band))

    def get_integ_par(self, cube):
        shape = cube.shape
        cx, cy = (28, 28) if self.field == 'COSMOS' else (29, 29)
        return shape, cx, cy

    def get_integ_spec(self, band, mask=None):
        cube = self._get_cube(band)
        shift_map = 1 + u.read_kin(self.gal_name, cube.shape[-2:]) / c.light_speed
        if cube.shape[-2:] != shift_map.shape:
            raise ValueError("cube must have the same shape as shift_map!")
        shape, cx, cy = self.get_integ_par(cube)
        PA = u.read_phot(self.gal_name, 'pa')
        incl = u.read_phot(self.gal_name, 'inc_deg')
        rad_mat = u.depr_rad(cube.shape[1:], (cx, cy), PA=PA, incl=incl)
        rad_mat *= c.arcsec_per_pix

        r_min, r_max = (0, n_re * self.Re) if mask is None else mask
        # mask is a tuple of (r_min, r_max) in unit of arcsec
        spec = np.zeros(shape[0])
        for i in range(shape[1]):
            for j in range(shape[2]):
                if (rad_mat[i, j] >= r_min) & (rad_mat[i, j] < r_max):
                    if vel_shift:
                        shift = u.find_nearest(shift_map, i, j)
                        spec += np.nan_to_num(u.shifted(band, cube[:, i, j], shift))
                    else:
                        spec += np.nan_to_num(cube[:, i, j])
        return spec

    def get_stacked_spec(self, band, mask=None, report=False):
        if not self._read_band(band):
            return np.zeros(len(std_wl_dict[band])) * np.nan
        spec = self.get_integ_spec(band, mask=mask)
        spec = np.where(~np.isnan(spec), spec, 0.)
        rf_wl = wl_range_dict[band] / (1 + self.redshift)
        f = interp1d(rf_wl, spec * (1 + self.redshift))
        return f(std_wl_dict[band])


    def fit_integ_spec(self, band, mask=None, report=False):
        spec = self.get_integ_spec(band, mask=mask)
        return SpecFit(spec, band, self.redshift).spec_fit(report=report)


    def _set_xlim(self, ax, band, redshift):
        lambda_major = c.restframe_dict[emis_major_dict[band]] * (1. + redshift)
        x1, x2 = lambda_major + fit_width_dict[band][0], lambda_major + fit_width_dict[band][0]
        if band == 'IZ':
            if redshift == 0:
                x1, x2 = .368, .3775
            xlim = ax.set_xlim(x1, x2)
        if band == 'YJ':
            if redshift == 0:
                x1, x2 = .480, .505
            xlim = ax.set_xlim(x1, x2)
        if band == 'H':
            if redshift == 0:
                x1, x2 = .652, .661
            xlim = ax.set_xlim(x1, x2)
        return xlim


    def save_integ_info(self, integ_info):
        for band in ['IZ', 'YJ', 'H', 'HK', 'K']:
            if not self._read_band(band):
                if band == 'IZ':
                    integ_info.write("%s,-99,-99," %(self.gal_name))
                if band == 'YJ':
                    integ_info.write("-99,-99,-99,-99,")
                if band == 'H':
                    integ_info.write("-99,-99,-99,-99,-99,-99,-99,-99\n")
                continue

            f1, f2, noise = self.fit_integ_spec(band)[1:4]
            sn1 = -1 if f1 < 1e-2 or (f1 > .4 and band == 'YJ') else f1 / noise  # Hb is faint.
            sn2 = f2 / noise if f2 > 1e-2 else -1

            if band == 'IZ':
                integ_info.write("%s,%.2f,%.2f," %(self.gal_name, f1, sn1))
            if band == 'YJ':
                integ_info.write("%.2f,%.2f,%.2f,%.2f," %(f1, sn1, f2, sn2))
            if band == 'H':
                integ_info.write("%.2f,%.2f,%.2f,%.2f," %(f1, sn1, f2, sn2))
            if band == 'HK':
                integ_info.write("%.2f,%.2f," %(f1, sn1))
            if band == 'K':
                integ_info.write("%.2f,%.2f\n" %(f1, sn1))


    def plot_integ_spec(self, ax, band, mask=None, report=False):   # Defaulted at rest frame
        if not self._read_band(band):
            ax.tick_params(labelleft=False)
            return self._set_xlim(ax, band, redshift=0)
        wl_range = std_wl_dict[band]
        wl_major = c.restframe_dict[emis_major_dict[band]]
        wl_minor1 = c.restframe_dict[emis_minor1_dict[band]]
        wl_minor2 = c.restframe_dict[emis_minor2_dict[band]]

        spec, S1, S2, N, best_fit = self.fit_integ_spec(band, mask=mask, report=report)

        xlim = self._set_xlim(ax, band, redshift=0)
        window = (wl_range > xlim[0]) & (wl_range < xlim[1])
        ax.step(wl_range[window], spec[window], color='k', lw=.5)
        ax.axvline(x=wl_major, color='gray', ls='--', lw=.5)
        ax.axvline(x=wl_minor1, color='gray', ls='--', lw=.5)
        ax.axvline(x=wl_minor2, color='gray', ls='--', lw=.5)

        if (S1 / N > min_SN) or (S2 / N > min_SN):
            if self.gal_name == 'cdfs_29207':
                ax.set_ylim(-200, 500)
            else:
                ax.plot(wl_range[window], best_fit[window], color='#df1d27', lw=1)
        
        return S1 / N, S2 / N

    def met_diag_annuli(self, i):
        mask = (rad_bin[i], rad_bin[i+1])
        if not self._read_band('IZ'):
            I_OII, e_OII = 0, -1
        else:
            I_OII,  _,  e_OII =  self.fit_integ_spec('IZ', mask=mask)[1:4]
        if not self._read_band('YJ'):
            I_Hb, I_OIII, e_Hb = 0, 0, -1
        else:
            I_Hb, I_OIII, e_Hb = self.fit_integ_spec('YJ', mask=mask)[1:4]
        if not self._read_band('H'):
            I_Ha, I_NII, e_Ha = 0, 0, -1
            I_SII6717, e_SII6717, I_SII6731, e_SII6731 = 0, -1, 0, -1
        else:
            I_Ha, I_NII, e_Ha = self.fit_integ_spec('H',        mask=mask)[1:4]
            I_SII6717, _, e_SII6717 = self.fit_integ_spec('HK', mask=mask)[1:4]
            I_SII6731, _, e_SII6731 = self.fit_integ_spec('K',  mask=mask)[1:4]

        d = {r'[OII]$\lambda\lambda$3727,9': (I_OII, e_OII),
             r'H$\beta$': (I_Hb, e_Hb),
             r'[OIII]$\lambda$5007': (I_OIII, e_Hb),
             r'H$\alpha$': (I_Ha, e_Ha),
             r'[NII]$\lambda$6584': (I_NII, e_Ha),
             r'[SII]$\lambda$6717': (I_SII6717, e_SII6717),
             r'[SII]$\lambda$6731': (I_SII6731, e_SII6731)}
        return MetDiag(self.gal_name, d=d, defaulted_Av=sp_res_Av[i])


    def fit_grad_annuli(self, diag):
        length = len(rad_bin) - 1
        met_array, err_array, flag_array = np.zeros(length), np.zeros(length),  np.zeros(length)

        for i in range(length):
            met, err = getattr(self.met_diag_annuli(i), diag)()
            if np.isnan(met):
                continue
            met_array[i], err_array[i] = met, err

        good = (err_array != 0) & (err_array < np.inf) & (err_array > -np.inf)
        dZdr, e_dZdr = u.fit_grad(self.x[good], met_array[good], err_array[good])
        f = u.read_met_grad_factor(self.gal_name)
        # From Gillman et al. (2021)
        # The intrinsic gradient is steeper than the observed one by a factor of f.
        return self.x, met_array, err_array, dZdr * f, e_dZdr * f

    def plot_grad_annuli(self, ax, diag, color,
                         zorder=0, alpha=1., lw=.5, rad_info=None, plot_leg=False):
        length = len(rad_bin) - 1

        rad, met, err, dZdr, e_dZdr = self.fit_grad_annuli(diag)

        good = (err != 0) & (err < np.inf) & (err > -np.inf)
        lowSN = (err != 0) & (np.abs(err) == np.inf)

        ax.scatter(rad[good], met[good], color=color, alpha=alpha, s=50, label=diag, zorder=zorder)
        ax.plot(rad[good], met[good], color=color, alpha=alpha, lw=lw, zorder=zorder)
        ax.vlines(rad[good], met[good] - err[good], met[good] + err[good],
                  color=color, alpha=alpha, zorder=zorder)
        direction = '+' if np.nanmax(err) == np.inf else '-'
        opt = dict(connectionstyle='arc3, rad=0', color=color,
                   arrowstyle='simple, head_width=.25, head_length=.5, tail_width=.03')
        for x, y in zip(rad[lowSN], met[lowSN]):
            ax.scatter(x, y, color=color, alpha=alpha, s=20, label=diag, zorder=zorder)
            ax.annotate("", xy=(x, ops[direction](y, .3)), xytext=(x, y), arrowprops=opt)
        if plot_leg:
            ax.legend(loc='upper right')

        if rad_info is not None:
            rad_info.write(',%.3f,%.3f' %(dZdr, e_dZdr))
        if not np.isnan(dZdr):
            print(diag + ' gradient is %.3f +/- %.3f dex / kpc' %(dZdr, e_dZdr))

        return rad, met, err, dZdr, e_dZdr



# gal_name = 'cdfs_30450'
# ax = plt.subplot(111)
# DataProd(gal_name).fit_integ_spec('IZ', mask=(1., 1.5))
# ax.set_xlim(.2, 16)
# ax.set_ylim(7.7, 9.3)
# plt.show()

