
from constant import Zsun, restframe_dict, caseBHaHb, DPline_ratio
from config import min_SN, integ_Av
from utils import read_line, read_Av

import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import pdb


def _kappa(wavelength):
    # From Battisti et al. (2022) at z ~ 1.3
    x = 1. / wavelength
    x0 = 1. / .2175
    gamma = 0.922
    return (3.80 * (-1.542 + 1.046 * x - 0.123 * x**2 + 0.0063 * x**3) + 3.26 + 
            0.83 * (x * gamma)**2 / ((x**2 - x0**2)**2 + (x * gamma)**2))

Rv = _kappa(.55)   # 3.225

# def _kappa(wavelength):
#     # From Calzetti et al. (2000)
#     x = 1 / wavelength
#     if wavelength >= .63:
#         return 2.659 * (-1.857 + 1.040*x) + Rv
#     else:
#         return 2.659 * (-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3) + Rv

# Rv = 4.05


def PPN2_M13N2_converter(array):
    l = len(array)
    res = np.zeros(l)
    for i in range(l):
        res[i] = np.roots((0.32, 1.26, 2.03, 9.37 - array[i]))[-1].real * 0.462 + 8.743
    return res

def M08N2_M13N2_converter(array):
    l = len(array)
    res = np.zeros(l)
    for i in range(l):
        x = array[i] - Zsun
        res[i] = (-.7732 + 1.2357*x - .2811*x**2 - .7201*x**3 - .3330*x**4) * 0.462 + 8.743
    return res

def _solve_met(line_ratio, pars, n=1000):
    a, b, c, d, e = pars
    res = []
    for i in range(n):
        y = np.random.normal(line_ratio.n, line_ratio.s)
        if y < 0 or not np.isreal(np.roots((a, b, c, d, e - np.log10(line_ratio.n)))[-1]):
            continue
        complex_root = np.roots((a, b, c, d, e - np.log10(y)))[-1]
        if complex_root.imag > 0:
            continue
        else:
            res.append(complex_root.real)
    return ufloat(np.median(res) + Zsun, np.std(res))


class MetDiag(object):
    def __init__(self, gal_name, d=None, defaulted_Av=integ_Av):
        self.gal_name = gal_name
        if d == None:
            self.d = {r'[OII]$\lambda\lambda$3727,9': read_line(gal_name, 'OII'),
                      r'H$\beta$': read_line(gal_name, 'Hb'),
                      r'[OIII]$\lambda$5007': read_line(gal_name, 'OIII5007'),
                      r'H$\alpha$': read_line(gal_name, 'Ha'),
                      r'[NII]$\lambda$6584': read_line(gal_name, 'NII6584'),
                      r'[SII]$\lambda$6717': read_line(gal_name, 'SII6717'),
                      r'[SII]$\lambda$6731': read_line(gal_name, 'SII6731')}
        else:
            self.d = d

        self.EBV = self.Balmer_EBV(defaulted_Av=defaulted_Av)
        self.Av = self.EBV * Rv


    def _u(self, label, lowSN=False):
        n, s = self.d[label]
        if lowSN:
            return ufloat(min_SN * s, s)
        else:
            return ufloat(n, s)

    def _bad_lines(self, label, min_SN=min_SN):
        n, s = self.d[label]
        return (np.nan_to_num(n) <= 0) or (np.nan_to_num(n / s) < min_SN)

    def _dered_f(self, wavelength, reddest_wavelength):
        kappa_a = _kappa(wavelength)
        kappa_b = _kappa(reddest_wavelength)
        return 10 ** (0.4 * (kappa_a - kappa_b) * self.EBV)

    def _ratio(self, up_list, down_list, lowSNu=False, lowSNd=False):
        reddest_wavelength = 0.
        for label in up_list + down_list:
            if restframe_dict[label] > reddest_wavelength:
                reddest_wavelength = restframe_dict[label]
        flux_up, flux_down = 0., 0.
        for label in up_list:
            flux_up  +=  self._u(label, lowSN=lowSNu) * self._dered_f(
                restframe_dict[label], reddest_wavelength)
        for label in down_list:
            flux_down += self._u(label, lowSN=lowSNd) * self._dered_f(
                restframe_dict[label], reddest_wavelength)
        return flux_up / flux_down

    def Balmer_EBV(self, defaulted_Av=integ_Av):
        if self._bad_lines(r'H$\alpha$') or self._bad_lines(r'H$\beta$') or \
            self._u(r'H$\alpha$') / self._u(r'H$\beta$') < 2.86 or \
            self._u(r'H$\alpha$') / self._u(r'H$\beta$') > 10:
            return defaulted_Av
        HaHb = self._u(r'H$\alpha$') / self._u(r'H$\beta$')
        return 1. / .4 * unp.log10(HaHb / caseBHaHb) / \
               (_kappa(restframe_dict[r'H$\beta$']) - _kappa(restframe_dict[r'H$\alpha$']))


    def PPN2(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        N2 = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$']))
        met = 9.37 + 2.03 * N2 + 1.26 * N2 ** 2 + 0.32 * N2 ** 3
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def M13N2(self):
        if self._bad_lines(r'[NII]$\lambda$6584', min_SN=0) or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        # N2 = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$']))
        # met = ufloat(8.743, 0.027) + ufloat(0.462, 0.024) * N2
        # return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)
        n, s = self.d[r'[NII]$\lambda$6584']
        lowSN = np.nan_to_num(n) > 0 and np.nan_to_num(n / s) < min_SN
        N2 = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'],
            lowSNu=lowSN)) * ufloat(1, 0)
        met = ufloat(8.743, 0.027) + ufloat(0.462, 0.024) * N2
        if lowSN:
            return (met.n, -np.inf)
        else:
            return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def B18N2(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        N2 = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$']))
        met = 8.98 + 0.63 * N2
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def PPO3N2(self):
        if self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'H$\beta$') or \
           self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        O3 = self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$'])
        N2 = self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'])
        O3N2 = unp.log10(O3 / N2)
        met = 8.73 - 0.32 * O3N2
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def M13O3N2(self):
        if self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'H$\beta$') or \
           self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        O3 = self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$'])
        N2 = self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'])
        O3N2 = unp.log10(O3 / N2)
        met = ufloat(8.533, 0.012) - ufloat(0.214, 0.012) * O3N2
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def B18O3N2(self):
        if self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'H$\beta$') or \
           self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$'):
            return (np.nan, np.nan)
        O3 = self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$'])
        N2 = self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'])
        O3N2 = unp.log10(O3 / N2)
        met = 9.05 - 0.47 * O3N2
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def D16(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$') or \
           self._bad_lines(r'[SII]$\lambda$6717') or self._bad_lines(r'[SII]$\lambda$6731'):
            return (np.nan, np.nan)
        y = (unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'[SII]$\lambda$6717', r'[SII]$\lambda$6731'])) +
            .264 * unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'])))
        met = 8.77 + y + 0.45 * (y + 0.3) ** 5
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)


    def Scal(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'[OIII]$\lambda$5007') or \
           self._bad_lines(r'H$\beta$') or \
           self._bad_lines(r'[SII]$\lambda$6717') or self._bad_lines(r'[SII]$\lambda$6731'):
            return (np.nan, np.nan)
        N2 = (1 + 1. / DPline_ratio) * self._ratio([r'[NII]$\lambda$6584'], [r'H$\beta$'])
        R3 = (1 + 1. / DPline_ratio) * self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$'])
        S2 = self._ratio([r'[SII]$\lambda$6717', r'[SII]$\lambda$6731'], [r'H$\beta$'])
        if unp.log10(N2) >= -0.6:
            met = +8.424 + 0.030 * unp.log10(R3/S2) + 0.751 * unp.log10(N2) + (
                  -0.349 + 0.182 * unp.log10(R3/S2) + 0.508 * unp.log10(N2)) * unp.log10(S2)
        else:
            met = +8.072 + 0.789 * unp.log10(R3/S2) + 0.726 * unp.log10(N2) + (
                  +1.069 - 0.170 * unp.log10(R3/S2) + 0.022 * unp.log10(N2)) * unp.log10(S2)
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def M08R23(self):
        if self._bad_lines(r'[OII]$\lambda\lambda$3727,9') or \
           self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'H$\beta$'):
            return (np.nan, np.nan)
        R23 = (self._ratio([r'[OII]$\lambda\lambda$3727,9', r'[OIII]$\lambda$5007'], [r'H$\beta$']) + 
               1. / DPline_ratio * self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$']))
        met = _solve_met(R23, [-.2524, -.6154, -.9401, -.7149, .7462])
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)

    def K19N2O2(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'[OII]$\lambda\lambda$3727,9', min_SN=0):
            return (np.nan, np.nan)
        # x = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'[OII]$\lambda\lambda$3727,9']))
        # y = -2.5
        # met = (9.4772 + 1.1797 * x + 0.5085 * y + 0.6879 * x * y + 0.2807 * x**2 +
        #        0.1612 * y**2 + 0.1187 * x * y**2 + 0.1200 * y * x**2 + 0.2293 * x**3 + 0.0164 * y**3)
        # return met if 8.3 < met.n < 9.3 else (np.nan, np.nan)
        n, s = self.d[r'[OII]$\lambda\lambda$3727,9']
        lowSN = np.nan_to_num(n) > 0 and np.nan_to_num(n / s) < min_SN
        x = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'[OII]$\lambda\lambda$3727,9'],
            lowSNd=lowSN)) * ufloat(1, 0)
        y = -2.5
        met = (9.4772 + 1.1797 * x + 0.5085 * y + 0.6879 * x * y + 0.2807 * x**2 +
               0.1612 * y**2 + 0.1187 * x * y**2 + 0.1200 * y * x**2 + 0.2293 * x**3 + 0.0164 * y**3)
        if lowSN:
            return (met.n, np.inf)
        else:
            return (met.n, met.s) if 8.3 < met.n < 9.3 else (np.nan, np.nan)

    def B18O32(self):
        if self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'[OII]$\lambda\lambda$3727,9'):
            return (np.nan, np.nan)
        met = 9.54 - 0.59 * (1 + 1. / DPline_ratio) * unp.log10(
            self._ratio([r'[OIII]$\lambda$5007'], [r'[OII]$\lambda\lambda$3727,9']))
        return (met.n, met.s) if 8.0 < met.n < 9.3 else (np.nan, np.nan)


    def BPT(self):
        if self._bad_lines(r'[NII]$\lambda$6584') or self._bad_lines(r'H$\alpha$') or \
           self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'H$\beta$', min_SN=0):
            return np.nan, np.nan, np.nan, np.nan
        n, s = self.d[r'H$\beta$']
        lowSN = np.nan_to_num(n) > 0 and np.nan_to_num(n / s) < min_SN
        y = unp.log10(self._ratio([r'[OIII]$\lambda$5007'], [r'H$\beta$'],
            lowSNd=lowSN)) * ufloat(1, 0)
        x = unp.log10(self._ratio([r'[NII]$\lambda$6584'], [r'H$\alpha$'])) * ufloat(1, 0)
        if lowSN:
            return x.n, x.s, y.n, np.inf
        else:
            return x.n, x.s, y.n, y.s


    def O32(self):
        if self._bad_lines(r'[OIII]$\lambda$5007') or self._bad_lines(r'[OII]$\lambda\lambda$3727,9'):
            return (np.nan, np.nan)
        x = unp.log10(self._ratio([r'[OIII]$\lambda$5007'], [r'[OII]$\lambda\lambda$3727,9']))
        y = 8.38
        ion = (13.768 + 9.4940 * x - 4.3223 * y - 2.3531 * x * y - 0.5769 * x**2 +
               0.2794 * y**2 + 0.1574 * x * y**2 + 0.0890 * y * x**2 + 0.0311 * x**3 + 0. * y**3)
        return ion

