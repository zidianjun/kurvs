
from paths import *
from config import rad_bin, wl_range_dict, min_SN, integ_Av
import constant as c

from astropy.io import fits
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, pearsonr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import warnings
from uncertainties import ufloat


#  Data reading

def get_band(band):
    if 'Y' in band:
        return 'YJ'
    elif 'K' in band:
        return 'H'
    else:
        return band

def get_field(gal_name):
    return 'CDFS' if 'cdfs' in gal_name or 'GS' in gal_name else 'COSMOS'

def read_list(field):
    return pd.read_csv(data_path + field + '_OBs' + '.csv').name

galaxy_list = np.concatenate([read_list('CDFS'), read_list('COSMOS')], axis=0)


def read_band(gal_name, band):
    df = pd.read_csv(data_path + get_field(gal_name) + '_OBs.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc(get_band(band))]

def read_phot(gal_name, par):
    df = pd.read_csv(data_path + get_field(gal_name) + '_morph_table.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc(par)]

def read_Av(gal_name, source):
    df = pd.read_csv(data_path + get_field(gal_name) + '_Av.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc('Av_' + source)]

def get_Av(source):
    return pd.read_csv(data_path + 'total_Av.csv')['Av_' + source]


def read_met_grad_factor(gal_name, q0=0.2):
    re, inc_deg = read_phot(gal_name, 're'), read_phot(gal_name, 'inc_deg')
    ba = np.sqrt(np.cos(inc_deg / 180 * np.pi) * (1 - q0**2) + q0**2)
    df = pd.read_csv(data_path + 'ref_data/Gillman21/ba' + str(int(ba * 10)) + '.csv')
    x, y = np.array(df.x), np.array(df.y)
    f = interp1d(x, y)
    if 0.6 / re > x[-1]:
        res = y[-1]
    elif 0.6 / re < x[0]:
        res = y[0]
    else:
        res = f(0.6 / re)
    return 1. / res

def read_integ(gal_name, label):
    df = pd.read_csv(output_path + get_field(gal_name) + '_integ_info.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc(label)]

def read_chem(gal_name, label):
    df = pd.read_csv(output_path + get_field(gal_name) + '_integ_met.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc(label)]

def read_FWHM(gal_name):
    df = pd.read_csv(output_path + 'HaFWHM.csv')
    for i, n in enumerate(galaxy_list):
        if gal_name == n:
            return np.array(df['HaFWHM'])[i]

def get_sdss_par(item):
    df_spec = pd.read_csv(data_path + 'ref_data/galSpecLine-dr8.csv')
    df_mass = pd.read_csv(data_path + 'ref_data/galSpecExtra-dr8.csv')
    ind = (df_spec.OIII_5007_FLUX / df_spec.OIII_5007_FLUX_ERR > min_SN)
    ind = ind & (df_spec.H_BETA_FLUX / df_spec.H_BETA_FLUX_ERR > min_SN)
    ind = ind & (df_spec.NII_6584_FLUX / df_spec.NII_6584_FLUX_ERR > min_SN)
    ind = ind & (df_spec.H_ALPHA_FLUX / df_spec.H_ALPHA_FLUX_ERR > min_SN)
    ind = ind & (df_mass.LGM_TOT_P50 > 0)
    df = df_spec if 'FLUX' in item else df_mass
    return np.array(df[ind][item])

def get_aurora_par():
    df = pd.read_csv(data_path + 'ref_data/ShapleyBPT.csv')
    return np.array(df.x), np.array(df.y)

def get_MZR(surname):
    df = pd.read_csv(data_path + 'ref_data/' + surname + 'MZR.csv')
    if surname in ['Belli', 'Henry', 'Jones', 'Wang', 'Wuyts']:
        return np.array(df['x']), np.array(df['y'])
    else:
        return np.log10(df['x']), np.array(df['y'])

def get_MGz(surname):
    df = pd.read_csv(data_path + 'ref_data/' + surname + 'MGz.csv')
    return np.array(df['x']), np.array(df['y'])

def get_MGM(surname):
    df = pd.read_csv(data_path + 'ref_data/' + surname + 'MGM.csv')
    return np.array(df['x']), np.array(df['y'])

def get_grad(label, bin_width=0.5):
    df = pd.read_csv(output_path + 'met_grads_radbin' + str(bin_width) + '.csv')
    return df[label]

def get_grad_Gillman():
    df = pd.read_csv(data_path + 'ref_data/GillmanMG.csv')
    return df['met_grad'], df['e_met_grad']

def get_OH_NO_Belfiore():
    inner = pd.read_csv(data_path + 'ref_data/BelfioreNO_inner.csv')
    outer = pd.read_csv(data_path + 'ref_data/BelfioreNO_outer.csv')
    return .5 * (inner['x'] + outer['x']), .5 * (inner['y'] + outer['y'])

def read_grad(gal_name, label, bin_width=0.5):
    df = pd.read_csv(output_path + 'met_grads_radbin' + str(bin_width) + '.csv')
    return df[df.name == gal_name].iloc[0, df.columns.get_loc(label)]

def read_spec(gal_name, band):
    tmp = fits.open(data_path + get_field(gal_name) + '/' + band + '/' + gal_name + '.fits')[0].data * c.unit
    if get_field(gal_name) == 'CDFS':
        if band == 'IZ':
            res = np.zeros((2048, 56, 56))
            res[:, 2:54, 2:54] = tmp
        if band == 'YJ':
            res = tmp.repeat(2, axis=-1).repeat(2, axis=-2) / 4.
        if band == 'H':
            res = tmp
    else:
        res = tmp
    return np.where(res == 0, np.nan, res)

def read_kin(gal_name, ref_shape):
    file_path = data_path + get_field(gal_name) + '/kinematics/' + \
                'vcen_ha_binmax_4_snrcut_5_' + gal_name + '_centred_01spax.fits'
    if os.path.isfile(file_path):
        return fits.open(file_path)[0].data
    else:
        print("%s's kinematics not found!" %(gal_name))
        return np.zeros(ref_shape)

def read_Vrot_sigma(gal_name):
    field = get_field(gal_name)
    df = pd.read_csv(data_path + get_field(gal_name) + '_kin_table.csv')
    Vrot = df[df.name == gal_name].iloc[0, df.columns.get_loc('v_34_int')]
    dVrot = df[df.name == gal_name].iloc[0, df.columns.get_loc('dv_34_int')]
    vel_disp = df[df.name == gal_name].iloc[0, df.columns.get_loc('sigma_0_int')]
    dvel_disp = df[df.name == gal_name].iloc[0, df.columns.get_loc('dsigma_0_int')]
    return Vrot, dVrot, vel_disp, dvel_disp

def get_sigma_model(vphi):
    model = pd.read_csv(data_path + 'model/Sharda21_vphi' + str(vphi) + '.csv')
    x, y = model.x, model.y
    return interp1d(x, y, fill_value=np.nan)

def get_Vrot_sigma_model(branch):
    model = pd.read_csv(data_path + 'model/Sharda21_Vrot_sigma_' + branch + '.csv')
    x, y = model.x, model.y
    return interp1d(x, y, fill_value=np.nan)

def read_line(gal_name, label):
    if label == 'Hb':
        I_Hb = read_integ(gal_name, 'I_Hb')
        e_Hb = read_integ(gal_name, 'I_Hb') / read_integ(gal_name, 'SN_Hb')
        if I_Hb <= 0 or e_Hb <= 0:
            e_Hb = read_integ(gal_name, 'I_OIII5007') / read_integ(gal_name, 'SN_OIII5007')
            if I_Hb <= 0 or e_Hb <= 0:
                I_Hb, e_Hb = np.nan, np.nan
        return I_Hb, e_Hb
    elif label == 'NII6584':
        I_NII = read_integ(gal_name, 'I_NII6584')
        e_NII = read_integ(gal_name, 'I_Ha') / read_integ(gal_name, 'SN_Ha')
        return I_NII, e_NII
    else:
        return (read_integ(gal_name, 'I_' + label),
                read_integ(gal_name, 'I_' + label) / read_integ(gal_name, 'SN_' + label))


def find_nearest(array, i, j, PA=0., incl=0.):
    if ~np.isnan(array[i, j]):
        return array[i, j]
    else:
        dist = depr_rad(array.shape, (j, i), PA=PA, incl=incl)
        dist = np.where(~np.isnan(array), dist, np.inf)
        return array.reshape(-1)[np.argmin(dist)]



def sky_substract(cube, ignore_r=1.5):   # 1.5 arcsec around the center belongs to targets
    ch, h, w = cube.shape
    half_h, half_w = int(h / 2), int(w / 2)
    sky = np.zeros(cube.shape) * np.nan
    for i in range(h):
        for j in range(w):
            if (i - half_h) ** 2 + (j - half_w) ** 2 > (ignore_r / c.arcsec_per_pix) ** 2:
                sky[:, i, j] = cube[:, i, j]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        telluric = np.nan_to_num(np.nanmedian(sky.reshape(ch, -1), axis=-1))
    res = cube.copy()
    for i in range(h):
        for j in range(w):
            res[:, i, j] = cube[:, i, j] - telluric
    return res



#  Spectrum fitting
def shifted(band, y, shift):
    x = wl_range_dict[band]
    f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    return f(x * shift)



def depr_rad(shape, cen_coord, PA=0., incl=0.):
    height, width = shape
    cx, cy = cen_coord
    cosi = np.cos(incl * np.pi / 180)
    theta = (PA + 90) * np.pi / 180  # PA=0 means north, but aligned to x axis.
    dep_mat = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta) / cosi, np.cos(theta) / cosi]])
    x0, y0 = np.meshgrid(range(width), range(height))
    x0, y0 = x0.reshape(-1), y0.reshape(-1)
    xy_mat = np.stack([x0 - cx, y0 - cy], axis=0)
    X, Y = np.dot(dep_mat, xy_mat)
    return np.sqrt(X ** 2 + Y ** 2).reshape(shape)

def bin_array(r, f, bin_size=2.5, err='std'):
    good = ~np.isnan(r) & ~np.isnan(f)
    stat = binned_statistic(r[good], f[good], statistic='mean', bins=rad_bin)
    bin_r = stat.bin_edges[:-1]
    bin_f = stat.statistic
    stat = binned_statistic(r[good], f[good], statistic='std', bins=rad_bin)
    if err == 'std':
        bin_u = stat.statistic
    else:
        num = np.sqrt(np.histogram(r[good], bins=rad_bin)[0])
        bin_u = stat.statistic / np.where(num > 0, num, np.nan)
    return bin_r, bin_f, bin_u



#   Fitting

def fit_grad(x, y, e, num=1000):
    if len(x) != len(y) or len(y) != len(e):
        raise ValueError("The lengths of all the input arrays must be the same!")
    if len(x) <= 1:
        return np.nan, np.nan
    else:
        par = np.zeros(num)
        for i in range(num):
            newy = np.random.normal(y, e)
            if len(x) == 2:
                par[i] = (newy[1] - newy[0]) / (x[1] - x[0])
            else:
                par[i] = curve_fit(lambda x, k, b: k * x + b, x, newy, p0=[-.1, 8.5])[0][0]
        return np.median(par), np.std(par)

def bpt_boundary(x, a, b, c):
    xx = np.where(x < b, x, np.nan)
    return a / (xx - b) + c

def linear(x, a, b):
    return a * x + b

def fit_linear(x, y, ex, ey, num=1000):
    if len(x) != len(y) or len(y) != len(ey):
        raise ValueError("The lengths of all the input arrays must be the same!")
    pars = np.zeros((num, 2))
    for i in range(num):
        newx = np.random.normal(x, ex)
        newy = np.random.normal(y, ey)
        pars[i] = curve_fit(linear, newx, newy, p0=[-.24, 2.80])[0]
    return np.median(pars, axis=0), np.std(pars, axis=0)



def mzr_func(x, z0, m0, beta):
    return z0 - np.log10(1 + (10 ** (x - m0)) ** -beta)


def fit_mzr(x, y, ex, ey, num=1000):
    if len(x) != len(y) or len(y) != len(ey):
        raise ValueError("The lengths of all the input arrays must be the same!")
    pars = np.zeros((num, 3))
    for i in range(num):
        newx = np.random.normal(x, ex)
        newy = np.random.normal(y, ey)
        pars[i] = curve_fit(mzr_func, newx, newy, p0=[8.8, np.min(x) + .5 * np.ptp(x), .81],
            bounds=[(8.5, np.min(x) + .33 * np.ptp(x), 0.2), (9.2, np.max(x) - .33 * np.ptp(x), 1.)])[0]
    return np.median(pars, axis=0), np.std(pars, axis=0)




def Pearson(x, y, ex, ey, num=1000):
    if len(x) != len(y) or len(y) != len(ey):
        raise ValueError("The lengths of all the input arrays must be the same!")
    pars = np.zeros(num)
    for i in range(num):
        newx = np.random.normal(x, ex)
        newy = np.random.normal(y, ey)
        pars[i] = pearsonr(newx, newy)[0]
    return np.median(pars), np.std(pars)





