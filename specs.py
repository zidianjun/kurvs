
from paths import *
from config import *
import constant as c
from utils import get_band

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings


def Gaussian(x, amp, center, sig):
    return amp / ((2 * np.pi) ** 0.5 * sig) * np.exp(-0.5 * (x - center) ** 2 / sig ** 2)

class SpecFit(object):
    # For the sake of using the global variable 'band'.
    def __init__(self, spec, band, redshift, rest_frame=False):
        self.spec_obs = spec
        if redshift == 0:
            self.spec_rest = spec
        else:
            spec_rest = np.where(~np.isnan(spec), spec, 0.)
            rf_wl = wl_range_dict[band] / (1 + redshift)
            f = interp1d(rf_wl, spec_rest * (1 + redshift))
            self.spec_rest = f(std_wl_dict[band])
        self.band, self.redshift = band, redshift


    def spec_fit_rest(self):
        lambda_major = c.restframe_dict[emis_major_dict[self.band]]
        lambda_minor = c.restframe_dict[emis_minor1_dict[self.band]]
        wavelength_range = std_wl_dict[self.band]
        if self.redshift == 0:
            weights = np.ones(len(wavelength_range))
        else:   # Convert the sky line weights to rest frame.
            f = interp1d(wl_range_dict[self.band] / (1 + self.redshift),
                  1. / (np.load(data_path + 'sky/KMOS_sky' + get_band(self.band) + '.npy') + 1) ** 2)
            weights = f(wavelength_range)
        
        spec_ind = ((wavelength_range > lambda_major + spec_width_dict[self.band][0]) &
                    (wavelength_range < lambda_major + spec_width_dict[self.band][1]))
        signal_ind = ((wavelength_range > lambda_major + fit_width_dict[self.band][0]) &
                      (wavelength_range < lambda_major + fit_width_dict[self.band][1]))
        x = wavelength_range[spec_ind & signal_ind]
        y = self.spec_rest[spec_ind & signal_ind]
        yn = self.spec_rest[spec_ind & ~signal_ind & (weights > min_sky_weight)]
        wn = weights[spec_ind & ~signal_ind & (weights > min_sky_weight)]

        cont = np.median(yn)  # Can be negative
        noise = np.sqrt(np.sum((yn - cont) ** 2 * wn) / np.sum(wn))

        if self.band in ['YJ', 'H']:
            popt, pcov = self.fit_multi_Gaussian_z0(x, y - cont, p0=[2e-2, 5e-3, 2e-4])
            Imajor, Iminor, sig = popt
        else:
            popt, pcov = self.fit_single_Gaussian_z0(x, y - cont, p0=[2e-2, 2e-4])
            Imajor, sig = popt
            Iminor = 0.

        sig_obs = np.sqrt(sig ** 2 + (c.instr_sig / (1 + self.redshift)) ** 2)
        dl = dlambda_dict[self.band] / (1 + self.redshift)
        Nch = 2.354 * sig_obs / dl
        Inoise = noise * np.sqrt(Nch) * dl

        if Imajor < 0 or Imajor > 2 or Iminor < 0 or abs(Inoise) < 1e-6:
            return wavelength_range, 0., 0., 999., sig, cont
        else:
            return wavelength_range, Imajor, Iminor, Inoise, sig, cont

    def spec_fit_obs(self):
        lambda_major = c.restframe_dict[emis_major_dict[self.band]]  * (1. + self.redshift)
        lambda_minor = c.restframe_dict[emis_minor1_dict[self.band]] * (1. + self.redshift)
        wavelength_range = wl_range_dict[self.band]
        weights = 1. / (np.load(data_path + 'sky/KMOS_sky' + get_band(self.band) + '.npy') + 1) ** 2

        spec_ind = ((wavelength_range > lambda_major + spec_width_dict[self.band][0]) &
                    (wavelength_range < lambda_major + spec_width_dict[self.band][1]))
        signal_ind = ((wavelength_range > lambda_major + fit_width_dict[self.band][0]) &
                      (wavelength_range < lambda_major + fit_width_dict[self.band][1]))
        x = wavelength_range[spec_ind & signal_ind]
        y = self.spec_obs[spec_ind & signal_ind]
        yn = self.spec_obs[spec_ind & ~signal_ind & (weights > min_sky_weight)]
        wn = weights[spec_ind & ~signal_ind & (weights > min_sky_weight)]

        cont = np.median(yn)  # Can be negative
        noise = np.sqrt(np.sum((yn - cont) ** 2 * wn) / np.sum(wn))

        if self.band in ['YJ', 'H']:
            popt, pcov = self.fit_multi_Gaussian(x, y - cont, p0=[self.redshift, 2e-2, 5e-3, 2e-4])
            z, Imajor, Iminor, sig = popt
        else:
            popt, pcov = self.fit_single_Gaussian(x, y - cont, p0=[self.redshift, 2e-2, 2e-4])
            z, Imajor, sig = popt
            Iminor = 0.

        sig_obs = np.sqrt(sig ** 2 + (c.instr_sig) ** 2)
        dl = dlambda_dict[self.band]
        Nch = 2.354 * sig_obs / dl
        Inoise = noise * np.sqrt(Nch) * dl

        if Imajor < 0 or Imajor > 2 or Iminor < 0 or abs(Inoise) < 1e-6:
            return wavelength_range, 0., 0., 999., sig, cont
        else:
            return wavelength_range, Imajor, Iminor, Inoise, sig, cont


    def spec_fit(self, report=False):

        wlr0, Imajor0, Iminor0, Inoise0, sig0, cont = self.spec_fit_rest()
        if self.redshift > 0:
            wlr1, Imajor1, Iminor1, Inoise1, sig1, _ = self.spec_fit_obs()
        else:
            Imajor1, Iminor1, Inoise1, best_fit = -np.inf, -np.inf, 999., np.zeros(len(self.spec_rest))

        if Imajor0 / Inoise0 > Imajor1 / Inoise1:
            Imajor, Iminor, Inoise, sig = Imajor0, Iminor0, Inoise0, sig0
            if report:
                print('%s flux = %.5f*10^{-16} erg/s/cm^2 (S/N = %.1f)'
                    %(emis_major_dict[self.band], Imajor, Imajor / Inoise))
                if emis_minor1_dict[self.band] != ' ':
                    print('%s flux = %.5f*10^{-16} erg/s/cm^2 (S/N = %.1f)'
                        %(emis_minor1_dict[self.band], Iminor, Iminor / Inoise))
                print('FWHM = %.1f km s^{-1}' %(sig * 2.354 * c.light_speed / (
                    c.restframe_dict[emis_major_dict[self.band]])))
                print('\n')
        else:
            Imajor, Iminor, Inoise, sig = Imajor1, Iminor1, Inoise1, sig0
            if report:
                print('%s flux = %.5f*10^{-16} erg/s/cm^2 (S/N = %.1f)'
                    %(emis_major_dict[self.band], Imajor, Imajor / Inoise))
                if emis_minor1_dict[self.band] != ' ':
                    print('%s flux = %.5f*10^{-16} erg/s/cm^2 (S/N = %.1f)'
                        %(emis_minor1_dict[self.band], Iminor, Iminor / Inoise))
                print('FWHM = %.1f km s^{-1}' %(sig * 2.354 * c.light_speed / (
                    c.restframe_dict[emis_major_dict[self.band]] * (1 + self.redshift))))
                print('\n')


        if Imajor < 0 or Imajor > 2 or Iminor < 0 or abs(Inoise) < 1e-6:
            return self.spec_rest - cont, 0., 0., 999., np.zeros(len(self.spec_rest))
        else:
            if self.band in ['YJ', 'H']:
                best_fit = self.multi_Gaussian_z0(wlr0, Imajor, Iminor, sig)
            else:
                best_fit = self.single_Gaussian_z0(wlr0, Imajor, sig)
            return self.spec_rest - cont, Imajor, Iminor, Inoise, best_fit


    def single_Gaussian(self, x, z, major, sig):
        sig_obs = np.sqrt(sig ** 2 + c.instr_sig ** 2)
        return Gaussian(x, major, c.restframe_dict[emis_major_dict[self.band]] * (1 + z), sig_obs)
    def fit_single_Gaussian(self, x, y, p0):
        z1, z2 = self.redshift*.99, self.redshift*1.01
        s1 = min_fwhm / 2.354 / c.light_speed * (
            c.restframe_dict[emis_major_dict[self.band]] * (1 + self.redshift))
        s2 = max_fwhm / 2.354 / c.light_speed * (
            c.restframe_dict[emis_major_dict[self.band]] * (1 + self.redshift))
        return curve_fit(self.single_Gaussian, x, y, p0, maxfev=40000,
            bounds=[(z1, 0, s1), (z2, 100, s2)])

    def multi_Gaussian(self, x, z, major, minor1, sig):
        sig_obs = np.sqrt(sig ** 2 + c.instr_sig ** 2)
        f1 = Gaussian(x, major, c.restframe_dict[emis_major_dict[self.band]] * (1 + z), sig_obs)
        f2 = Gaussian(x, minor1, c.restframe_dict[emis_minor1_dict[self.band]] * (1 + z), sig_obs)
        f3 = Gaussian(x, minor1 / c.DPline_ratio,
                      c.restframe_dict[emis_minor2_dict[self.band]] * (1 + z), sig_obs)
        return f1 + f2 + f3
    def fit_multi_Gaussian(self, x, y, p0):
        z1, z2 = self.redshift*.99, self.redshift*1.01
        s1 = min_fwhm / 2.354 / c.light_speed * (
            c.restframe_dict[emis_major_dict[self.band]] * (1 + self.redshift))
        s2 = max_fwhm / 2.354 / c.light_speed * (
            c.restframe_dict[emis_major_dict[self.band]] * (1 + self.redshift))
        return curve_fit(self.multi_Gaussian, x, y, p0, maxfev=40000,
            bounds=[(z1, 0, 0, s1), (z2, 100, 100, s2)])

    def single_Gaussian_z0(self, x, major, sig):
        sig_obs = np.sqrt(sig ** 2 + (c.instr_sig / (1 + self.redshift)) ** 2)
        return Gaussian(x, major, c.restframe_dict[emis_major_dict[self.band]], sig_obs)
    def fit_single_Gaussian_z0(self, x, y, p0):
        s1 = min_fwhm / 2.354 / c.light_speed * c.restframe_dict[emis_major_dict[self.band]]
        s2 = max_fwhm / 2.354 / c.light_speed * c.restframe_dict[emis_major_dict[self.band]]
        return curve_fit(self.single_Gaussian_z0, x, y, p0, maxfev=10000, bounds=[(0, s1), (100, s2)])

    def multi_Gaussian_z0(self, x, major, minor1, sig):
        sig_obs = np.sqrt(sig ** 2 + (c.instr_sig / (1 + self.redshift)) ** 2)
        f1 = Gaussian(x, major, c.restframe_dict[emis_major_dict[self.band]], sig_obs)
        f2 = Gaussian(x, minor1, c.restframe_dict[emis_minor1_dict[self.band]], sig_obs)
        f3 = Gaussian(x, minor1 / c.DPline_ratio, c.restframe_dict[emis_minor2_dict[self.band]], sig_obs)
        return f1 + f2 + f3
    def fit_multi_Gaussian_z0(self, x, y, p0):
        s1 = min_fwhm / 2.354 / c.light_speed * c.restframe_dict[emis_major_dict[self.band]]
        s2 = max_fwhm / 2.354 / c.light_speed * c.restframe_dict[emis_major_dict[self.band]]
        return curve_fit(self.multi_Gaussian_z0, x, y, p0, maxfev=10000, bounds=[(0, 0, s1), (100, 100, s2)])





