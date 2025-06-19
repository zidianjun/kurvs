
from paths import *
from config import bin_size, rad_bin, n_re, std_wl_dict
from constant import arcsec_per_pix
import utils as u
from dataprod import DataProd
from diagnostics import MetDiag, M08N2_M13N2_converter, PPN2_M13N2_converter, Rv
from falsecolor import KMOSFoV
from specs import SpecFit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, binned_statistic
from uncertainties import ufloat
import uncertainties.unumpy as unp
import warnings


blue = '#367db7'
red = '#df1d27'
green = '#4aad4a'
orange = 'orange'
purple = 'mediumpurple'
turquoise = 'turquoise'

galaxy_list = np.concatenate([u.read_list('CDFS'), u.read_list('COSMOS')], axis=0)
Av_good_list = []
for gal_name in galaxy_list:
    if MetDiag(gal_name, defaulted_Av=0).EBV > 0:
        Av_good_list.append(gal_name)
# print(Av_good_list)


def _hist2d(xdata, ydata, x_range, y_range):
    x0, x1, nx = x_range
    y0, y1, ny = y_range
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny

    x, y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))

    count = np.zeros(nx * ny)
    for i in range(len(xdata)):
        ind_x = int((xdata[i] - x0) / dx - 1)
        ind_y = int((ydata[i] - y0) / dy - 1)
        ind = ind_y * nx + ind_x
        if 0 < ind < nx * ny:
            count[ind] += 1

    c = (np.where(count > 0, count, 1) / np.max(count))
    return (x, y, c.reshape((nx, ny)))


def write_integ_info(field):
    integ_info = open(output_path + field + '_integ_info.csv', 'w')
    integ_info.write('name,I_OII,SN_OII,I_Hb,SN_Hb,I_OIII5007,SN_OIII5007,' +
                     'I_Ha,SN_Ha,I_NII6584,SN_NII6584,I_SII6717,SN_SII6717,I_SII6731,SN_SII6731\n')
    integ_info.close()
    integ_info = open(output_path + field + '_integ_info.csv', 'a+')
    for gal_name in u.read_list(field):
        print(gal_name)
        dp = DataProd(gal_name)
        dp.save_integ_info(integ_info)
    integ_info.close()



def show_integ_info():

    for i, gal_name in enumerate(galaxy_list):
        p, q = i // 11, i % 11
        if q == 0:
            fig = plt.figure(figsize=(10, 11.5))
            plt.subplots_adjust(left=.05, bottom=.05, right=.95, top=.975, wspace=.8, hspace=0.)

        dp = DataProd(gal_name)
        ax = plt.subplot2grid((11, 7), (q, 0))
        KMOSFoV(gal_name, ax, area='integ')
        ax.set_ylabel('KURVS_' + str(i + 2) if i >= 36 else 'KURVS_' + str(i + 1))

        for j, band in zip([1, 2, 3], ['IZ', 'YJ', 'H']):
            ax = plt.subplot2grid((11, 7), (q, 2 * j - 1), colspan=2)
            dp.plot_integ_spec(ax, band, mask=(0, n_re * u.read_phot(gal_name, 're')))
            if q == 0:
                ax.set_title('$' + band + '$')
            if q < 10 and i != 42:
                ax.tick_params(labelbottom=False)
            else:
                if j == 2:
                    ax.set_xlabel(r'Rest-frame wavelength ($\mu$m)')
            if j == 3 and q == 5:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r'Flux intensity (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$)')

        if q == 10 or i == 42:
            # plt.show()
            plt.savefig(savefig_path + 'specs_sec' + str(p + 1) + '.pdf')


def write_integ_met(field):
    integ_info = open(output_path + field + '_integ_met.csv', 'w')
    integ_info.write('name,PPN2,e_PPN2,M13N2,e_M13N2,B18N2,e_B18N2,D16,e_D16,PPO3N2,e_PPO3N2,' +
                     'M13O3N2,e_M13O3N2,B18O3N2,e_B18O3N2,M08R23,e_M08R23,Scal,e_Scal,B18O32,e_B18O32,' +
                     'K19N2O2,e_K19N2O2\n')
    integ_info.close()
    integ_info = open(output_path + field + '_integ_met.csv', 'a+')

    for gal_name in u.read_list(field):
        print(gal_name)
        integ_info.write("%s" %(gal_name))
        for diag in ['PPN2', 'M13N2', 'B18N2', 'D16', 'PPO3N2', 'M13O3N2',
                     'B18O3N2', 'M08R23', 'Scal', 'B18O32', 'K19N2O2']:
            met, err = getattr(MetDiag(gal_name), diag)()
            integ_info.write(",%.2f,%.2f" %(met, err))
        integ_info.write("\n")

    integ_info.close()


def stacking_galaxy(l, mask=None):
    stackYJ = np.zeros(len(std_wl_dict['YJ']))
    stackH  =  np.zeros(len(std_wl_dict['H']))
    meanAvstar, totalMass = 0, 0
    for gal_name in l:
        m = 1. / 10 ** u.read_phot(gal_name, 'logMstar')
        yj = DataProd(gal_name).get_stacked_spec('YJ', mask=mask)
        h  = DataProd(gal_name).get_stacked_spec('H',  mask=mask)
        if np.sum(np.isnan(yj)) == 0 and np.sum(np.isnan(h)) == 0:
            stackYJ = (stackYJ * totalMass + yj * m) / (totalMass + m)
            stackH  =  (stackH * totalMass + h  * m) / (totalMass + m)
        meanAvstar += u.read_Av(gal_name, 'star')
        totalMass += m
    print('Averaged Av_star: %.2f' %(meanAvstar / len(l)))
    I_Hb, _, e_Hb = SpecFit(stackYJ, 'YJ', redshift=0.).spec_fit(report=True)[1:4]
    I_Ha, _, e_Ha  = SpecFit(stackH, 'H',  redshift=0.).spec_fit(report=True)[1:4]
    md = MetDiag('stacked', d={r'H$\beta$': (I_Hb, e_Hb), r'H$\alpha$': (I_Ha, e_Ha)})
    print('Averaged Av_gas: ', md.Balmer_EBV())


def Av_gas_star():
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    ax = plt.subplot(111)

    for gal_name in galaxy_list:
        Av = MetDiag(gal_name, defaulted_Av=0).Av
        if Av > 0:
            ax.scatter(u.read_Av(gal_name, 'star'), Av.n, fc='lightblue', ec='k', s=100, zorder=2)
            ax.vlines(u.read_Av(gal_name, 'star'), Av.n - Av.s, Av.n + Av.s, color='lightblue', zorder=1)

    ax.scatter(np.nan, np.nan, fc='lightblue', ec='k', s=100,
              label=('Six galaxies\n' + r'with H$\beta$ S/N $>4$'))
    
    meanAvstar, meanAvgas, meanAvgas_err = .90, .25, .13
    ax.scatter(meanAvstar, meanAvgas, fc=blue, ec='k', s=200, marker='s', zorder=2,
               label='Stacked spectrum of\nthe six galaxies')
    ax.vlines(meanAvstar, meanAvgas - meanAvgas_err, meanAvgas + meanAvgas_err, color=blue, zorder=1)
    # Av_gas from stacked spectrum vs. averaged Av_star of good galaxies
    meanAvstar, meanAvgas, meanAvgas_err = .99, .40, .25
    ax.scatter(meanAvstar, meanAvgas, fc=red, ec='k', s=200, marker='s', zorder=2,
               label='Stacked spectrum of\nall the galaxies')
    ax.vlines(meanAvstar, meanAvgas - meanAvgas_err, meanAvgas + meanAvgas_err, color=red, zorder=1)
    # Av_gas from stacked spectrum vs. averaged Av_star of all galaxies
    ax.axhline(y=0.42, ls='--', color=orange, zorder=0)
    ax.annotate(r'$z\in (1.0, 1.7)$'+'\n(Matharu+23)', xy=(2, .45), fontsize=15)
    ax.axhline(y=0.78, ls='--', color=orange, zorder=0)
    ax.annotate(r'$z\in (0.75, 1.5)$'+'\n(Domínguez+13)', xy=(2, .81), fontsize=15)
    ax.axhline(y=1.28, ls='--', color=orange, zorder=0)
    ax.annotate(r'$z\sim1.3$'+'\n(Battisti+22)', xy=(2, 1.31), fontsize=15)
    xx = np.arange(-2, 1.5, .001)
    ax.plot(xx, xx * (1.9 - 0.15 * xx), color=green, zorder=0,
            label=(r'$A_{V, \mathrm{gas}} = A_{V, \mathrm{star}}\times$' + '\n' +
                   r'$(1.9 - 0.15A_{V, \mathrm{star}})$' +
                    '\n(Wuyts+13)'))

    ax.set_xlim(.01, 2.9)
    ax.set_ylim(.01, 3.4)
    ax.set_xlabel(r'$A_{V, \mathrm{star}}$', fontsize=20)
    ax.set_ylabel(r'$A_{V, \mathrm{gas}}$', fontsize=20)
    ax.tick_params(labelsize=20, direction='in')
    ax.legend(loc='upper right', fontsize=15)

    # plt.show()
    plt.savefig(savefig_path + 'Av_gas_star.pdf')



def plot_basic_info():

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.15, bottom=.15, right=.95, top=.95)
    line0, label0 = [], []
    line1, label1 = [], []

    ax = plt.subplot(111)
    xx = np.arange(8, 12, .01)
    ms = xx - 9 - .5 + 1.5*np.log10(1+1.5) - 0.3*(np.where(
        xx - 9 - 0.36 - 2.5*np.log10(1+1.5) > 0, xx - 9 - 0.36 - 2.5*np.log10(1+1.5), 0))**2
    ax.plot(xx, ms, color='k', ls='-', zorder=1)
    ax.fill_between(xx, ms+np.log10(2), ms-np.log10(2), color='lightgray', zorder=0)
    for gal_name in galaxy_list:
        Mstar = u.read_phot(gal_name, 'logMstar')
        SFR = np.log10(u.read_phot(gal_name, 'SFR'))
        ax.scatter(Mstar, SFR, fc=red, ec='k', s=100, zorder=2)

    ax.set_xlim(9.2, 11.7)
    ax.set_ylim(-.1, 2.6)
    ax.set_xlabel(r'log($M_*$/M$_{\odot}$)', fontsize=20)
    ax.set_ylabel(r'log(SFR/M$_{\odot}$yr$^{-1}$)', fontsize=20)
    ax.tick_params(labelsize=20, direction='in')
    line0.append(ax.scatter(np.nan, np.nan, fc=red, ec='k', s=100))
    label0.append('KURVS sample')
    line1.append(ax.plot(np.nan, np.nan, color='k', ls='-')[0])
    label1.append('MS (Schreiber+15)\n' + r'$1.2<z<1.8$')
    ax.errorbar(11.4, .3, xerr=.2, yerr=.1, color='gray', capsize=4)
    legend0 = ax.legend(line0, label0, loc='upper left', fontsize=20)
    legend1 = ax.legend(line1, label1, loc='lower left', fontsize=20)
    plt.gca().add_artist(legend0)
    plt.gca().add_artist(legend1)

    # plt.show()
    plt.savefig(savefig_path + 'basic_info.pdf')




def plot_bpt_info():

    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=.10, bottom=.15, right=.98, top=.96, wspace=0)

    ax = plt.subplot(121)
    xx = np.arange(-3, 1, .01)

    x_array, y_array = np.zeros(len(galaxy_list)), np.zeros(len(galaxy_list))
    xerr_array, yerr_array = np.zeros(len(galaxy_list)), np.zeros(len(galaxy_list))
    for i, gal_name in enumerate(galaxy_list):
        x, xerr, y, yerr = MetDiag(gal_name).BPT()
        if np.isnan(x):
            continue
        x_array[i] = x
        y_array[i] = y
        xerr_array[i] = xerr
        yerr_array[i] = yerr
    normal = (yerr_array > 0) & (yerr_array < np.inf)
    lowSN = (yerr_array > 0) & ~normal

    ax.scatter(x_array[normal], y_array[normal], fc='gray', ec='k', s=100, zorder=3)
    ax.vlines(x_array[normal], y_array[normal] - yerr_array[normal], y_array[normal] + yerr_array[normal],
              color='gray', zorder=2)
    ax.hlines(y_array[normal], x_array[normal] - xerr_array[normal], x_array[normal] + xerr_array[normal],
              color='gray', zorder=2)
    ax.scatter(x_array[lowSN], y_array[lowSN], fc='lightgray', ec='gray', s=50, zorder=3)

    opt = dict(connectionstyle='arc3, rad=0', color='gray',
               arrowstyle='simple, head_width=.25, head_length=.5, tail_width=.03')
    for x, y in zip(x_array[lowSN], y_array[lowSN]):
        ax.annotate("", xy=(x, y+.15), xytext=(x, y+.01), arrowprops=opt)

    sdss_x = np.log10(u.get_sdss_par('NII_6584_FLUX') / u.get_sdss_par('H_ALPHA_FLUX'))
    sdss_y = np.log10(u.get_sdss_par('OIII_5007_FLUX') / u.get_sdss_par('H_BETA_FLUX'))
    sf = sdss_y < u.bpt_boundary(sdss_x, .61, .05, 1.30)
    x_grid, y_grid, c_grid = _hist2d(sdss_x[sf], sdss_y[sf],
                                     x_range=(-1.5, 0, 300), y_range=(-1, .8, 300))
    # ax.contour(x_grid, y_grid, c_grid, cmap=plt.cm.binary, zorder=0)
    ax.pcolor(x_grid, y_grid, c_grid, cmap=plt.cm.binary, rasterized=True, zorder=0)

    ax.plot(xx, u.bpt_boundary(xx, .67, .33, 1.13), color=blue,
            label='Best fit at $z\sim2.3$' + '\n(Steidel+14)', zorder=1)
    ax.scatter([-1.1, -.8, -.66, -.57, -.41], [.60, .42, .24, -.06, .05], color='cyan', s=50, ec='k',
               marker='s', alpha=.5, label='Binned data at $z\sim1.6$' + '\n(Zahid+14)', zorder=2)
    ax.vlines([-1.1, -.8, -.66, -.57, -.41], [.49, .33, .12, -.14, -.07], [.71, .52, .34, .03, .17],
              color='cyan', alpha=.5, zorder=1)
    ax.hlines([.60, .42, .24, -.06, .05], [-1.23, -.84, -.72, -.62, -.48], [-.97, -.74, -.60, -.53, -.34],
              color='cyan', alpha=.5, zorder=1)
    ax.plot(xx, u.bpt_boundary(xx, .67, .20, 1.12), color=green,
            label='Best fit at $z\sim2.3$' + '\n(Shapley+15)', zorder=1)
    ax.plot(xx, u.bpt_boundary(xx, .61, .22, 1.12), color=orange,
            label='Best fit at $z\sim2.3$' + '\n(Strom+17)', zorder=1)
    ax.plot(xx, u.bpt_boundary(xx, .61, .13, 1.09), color=red,
            label='Best fit at $z\sim1.6$' + '\n(Kashino+19)', zorder=1)
    ax.scatter(*u.get_aurora_par(), fc=purple, ec='face', s=10, zorder=2,
               label='AURORA at $z\in(1.4, 2.7)$' + '\n(Shapley+24)')

    ax.plot(xx, u.bpt_boundary(xx, .61, .05, 1.30), ls='--', color='gray', zorder=1)
    ax.plot(xx, u.bpt_boundary(xx, .61, .47, 1.19), ls='-.', color='gray', zorder=1)

    ax.set_xlim(-2.2, 0.2)
    ax.set_ylim(-1.8, 1.2)
    ax.set_xlabel(r'log([NII]$\lambda$6584/H$\alpha$)', fontsize=20)
    ax.set_ylabel(r'log([OIII]$\lambda$5007/H$\beta$)', fontsize=20)
    ax.tick_params(labelsize=20, direction='in')
    ax.legend(loc='lower left', fontsize=13)


    ax = plt.subplot(122)
    
    sdss_x = u.get_sdss_par('LGM_TOT_P50')
    sdss_y = np.log10(u.get_sdss_par('OIII_5007_FLUX') / u.get_sdss_par('H_BETA_FLUX'))
    x_grid, y_grid, c_grid = _hist2d(sdss_x[sf], sdss_y[sf],
                                     x_range=(8.3, 11.3, 300), y_range=(-1, .8, 300))
    # ax.contour(x_grid, y_grid, c_grid, cmap=plt.cm.binary, zorder=0)
    ax.pcolor(x_grid, y_grid, c_grid, cmap=plt.cm.binary, rasterized=True, zorder=0)

    Mstar_array = []
    for gal_name in galaxy_list:
        Mstar_array.append(u.read_phot(gal_name, 'logMstar'))
    Mstar_array = np.array(Mstar_array)

    ax.scatter(Mstar_array[normal], y_array[normal], fc='gray', ec='k', s=100, zorder=3)
    ax.vlines(Mstar_array[normal], y_array[normal] - yerr_array[normal],
              y_array[normal] + yerr_array[normal], color='gray', zorder=2)
    ax.hlines(y_array[normal], Mstar_array[normal] - .2, Mstar_array[normal] + .2, color='gray', zorder=2)
    ax.scatter(Mstar_array[lowSN], y_array[lowSN], fc='lightgray', ec='gray', s=50, zorder=2)
    for x, y in zip(Mstar_array[lowSN], y_array[lowSN]):
        ax.annotate("", xy=(x, y+.15), xytext=(x, y+.01), arrowprops=opt)

    plt.xlim(8.3, 11.9)
    plt.ylim(-1.8, 1.3)
    ax.set_xlabel(r'log($M_*$/M$_{\odot}$)', fontsize=20)
    ax.tick_params(labelleft=False, labelsize=20, direction='in')


    xx = np.arange(8.7, 11.2, .01)
    ax.plot(xx, -.290 * xx + 3.419, color=orange, zorder=0,
            label='Best fit at $z\sim2.3$' + '\n(Strom+17)')
    xx = np.arange(9.2, 11.4, .01)
    ax.plot(xx, -.540 * xx + 5.630, color=red, zorder=0,
            label='Best fit at $z\sim1.6$' + '\n(Kashino+19)')


    xx = np.arange(7.5, 12, .01)
    upper = np.where(xx < 10., .375/(xx-10.5)+1.14, 410.24-109.333*xx+9.71731*xx**2-0.288244*xx**3)
    ax.plot(xx, upper, ls='--', color='gray', zorder=0)
    xx = np.arange(8.6, 10, .01)
    ax.plot(xx, -.724 * xx + 6.812, ls=':', color='k', zorder=0,
        label='Best fit at $z=0$' + '\n(SDSS, ' + 'M$_*<10^{10}$M$_{\odot}$)')

    ax.legend(loc='lower left', fontsize=13)

    # plt.show()
    plt.savefig(savefig_path + 'bpt_info.pdf')



def plot_mzr_info():
    diag_list = ['M13N2', 'M08R23', 'K19N2O2']
    color_list = [red, turquoise, purple]

    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(left=.06, bottom=.10, right=.98, top=.96, wspace=0)
    xx = np.arange(0, 20, .001)

    for i, diag, color in zip(range(len(diag_list)), diag_list, color_list):
        ax = plt.subplot(1, len(diag_list), 1 + i)
        mass_l, mass_m, mass_h = [], [], []

        l = len(galaxy_list)
        mass_array, met_array, err_array = np.zeros(l), np.zeros(l), np.zeros(l)
        for j, gal_name in enumerate(galaxy_list):
            Mstar = u.read_phot(gal_name, 'logMstar')
            mass_array[j] = Mstar
            met = u.read_chem(gal_name, diag)
            err = u.read_chem(gal_name, 'e_' + diag)
            met_array[j] = met
            err_array[j] = err
            if Mstar < 10:
                mass_l.append(gal_name)
            elif Mstar < 10.5:
                mass_m.append(gal_name)
            else:
                mass_h.append(gal_name)

        ax.scatter(mass_array, met_array, fc=color, ec='k', s=100, zorder=4)
        ax.vlines(mass_array, met_array - err_array, met_array + err_array,
                  color=color, alpha=.5, zorder=3)

        x = mass_array[~np.isnan(met_array)]
        y = met_array[~np.isnan(met_array)]
        e = err_array[~np.isnan(met_array)]
        if diag == 'M08R23':
            x0, y0 = u.get_MZR('Henry')
            med, std = u.fit_mzr(np.concatenate([x, x0], axis=0), np.concatenate([y, y0], axis=0),
                .2, np.concatenate([e, np.ones(len(x0)) * np.nanmedian(e)], axis=0))
            ax.plot(xx, u.mzr_func(xx, *med), ls='--', lw=3, color=color, zorder=2)
        else:
            med, std = u.fit_mzr(x, y, .2, e)
            ax.plot(xx, u.mzr_func(xx, *med), ls='-', lw=3, color=color, zorder=2)[0]
        print(diag, np.sum(~np.isnan(met_array)), med, std)
        
        if i == 0:
            ax.plot(xx, u.mzr_func(xx, 8.61, 9.80, .81), ls=':', lw=2, color=red, zorder=1,
                    label='Zahid+14 ($1.4 < z < 1.7$)')[0]
            x, y = u.get_MZR('Wuyts')
            ax.scatter(x, M08N2_M13N2_converter(y), fc=red, ec='k', marker='^', zorder=1,
                       label='Wuyts+12 ($0.9<z<2.5$)')
            ax.scatter(*u.get_MZR('Yabe'), fc=red, ec='k', marker='v', zorder=1,
                       label='Yabe+15 ($z\sim1.4$)')
            ax.scatter(*u.get_MZR('Kashino'), fc=red, ec='k', marker='s', zorder=1,
                       label='Kashino+17 ($z\sim1.6$)')
            ax.scatter(*u.get_MZR('Topping'), fc=red, ec='k', marker='D', zorder=1,
                       label='Topping+21 ($z\sim1.5$)')
            abcde = [23.9049, -5.62784, 0.645142, -0.0235065, 0.09]
        elif i == 1:
            ax.scatter(*u.get_MZR('Henry'), fc=turquoise, ec='k', marker='>', zorder=1,
                       label='Henry+13 ($1.3<z<2.3$)')
            abcde =  [72.0142, -20.6826, 2.22124, -0.0783089, 0.13]
        else:
            abcde =  [28.0974, -7.23631, 0.850344, -0.0318315, 0.10]

        a, b, c, d, e = abcde
        xxx = np.arange(8.5, 11, .01)
        f = PPN2_M13N2_converter if i == 0 else lambda x: x
        ax.fill_between(xxx, f(a + b * xxx + c * xxx**2 + d * xxx**3) - e,
            f(a + b * xxx + c * xxx**2 + d * xxx**3) + e, color='lightgray', alpha=.4, zorder=0) 

        ax.set_xlim(7.9, 11.7)
        ax.set_ylim(7.95, 9.25)
        ax.set_xlabel(r'log($M_*$/M$_{\odot}$)', fontsize=15)
        if i == 0:
            ax.set_ylabel('12 + log(O/H)', fontsize=15)
            ax.tick_params(labelsize=15, direction='in')
        else:
            ax.tick_params(labelleft=False, labelsize=15, direction='in')
        ax.annotate(diag[3:], xy=(10.8, 8), color=color, fontsize=20)
        if i < 2:
            ax.legend(loc='upper left', fontsize=15)   

    # plt.show()
    plt.savefig(savefig_path + 'mzr_info.pdf')


def corner_diag():
    diag_list = ['M13N2', 'D16', 'M08R23', 'M13O3N2', 'K19N2O2']

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=.12, bottom=.12, right=.96, top=.96, hspace=0., wspace=0.)
    xx = np.arange(6, 12, 1)

    size = len(diag_list) - 1
    for i in range(1, size + 1):
        for j in range(size):
            if (i - 1) * size + j + 1 > (i - 1) * size + i:
                break
            ax = plt.subplot(size, size, (i - 1) * size + j + 1)
            ax.set_xlim(7.7, 9.7)
            ax.set_ylim(7.7, 9.7)
            if j == 0 and i == 1:
                ax.plot(xx, 2.3704 * xx - 11.452, color=red, ls=':', zorder=0)
            for gal_name in galaxy_list:
                x, ex = u.read_chem(gal_name, diag_list[j]), u.read_chem(gal_name, 'e_' + diag_list[j])
                y, ey = u.read_chem(gal_name, diag_list[i]), u.read_chem(gal_name, 'e_' + diag_list[i])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    ax.scatter(x, y, s=10, fc='gray', ec='k', zorder=2)
                    ax.vlines(x, y - ey, y + ey, color='gray', zorder=1)
                    ax.hlines(y, x - ex, x + ex, color='gray', zorder=1)
            ax.plot(xx, xx, ls=':', color='gray')
            xaxis = i == size
            yaxis = j == 0
            d1 = 'D16' if diag_list[j] == 'D16' else diag_list[j][3:]
            d2 = 'D16' if diag_list[i] == 'D16' else diag_list[i][3:]
            if xaxis:
                ax.set_xlabel('12 + log(O/H)\n[' + d1 + ']')
            if yaxis:
                ax.set_ylabel('12 + log(O/H)\n[' + d2 + ']')
            ax.tick_params(axis='both', direction='in', labelbottom=xaxis, labelleft=yaxis)

    # plt.show()
    plt.savefig(savefig_path + 'corner_diag.pdf')


def show_radial_info(gal_name):

    fig = plt.figure(figsize=(20, 5))
    plt.subplots_adjust(left=0.02, bottom=.15, right=.98, top=.95, wspace=3.5, hspace=0)

    dp = DataProd(gal_name)

    nrow = len(rad_bin) - 1
    ax = plt.subplot2grid((nrow, 12), (0, 0), colspan=3, rowspan=nrow)

    KMOSFoV(gal_name, ax, area='resolved', plot_scale=False)

    for i, band in enumerate(['IZ', 'YJ', 'H']):
        for j, letter in zip(range(len(rad_bin) - 1), ['A', 'B', 'C']):
            ax = plt.subplot2grid((nrow, 12), (j, 3 + 3 * i), colspan=3)
            dp.plot_integ_spec(ax, band, (rad_bin[j], rad_bin[j + 1]), report=True)
            # axis fraction
            ax.annotate(letter, xy=(.9, .8), xycoords='axes fraction', color='k', fontsize=15)
            if j < len(rad_bin) - 2:
                ax.tick_params(labelbottom=False, labelsize=15)
            if j == 0:
                ax.set_title('$' + band + '$')
            if i == 0 and j == 1:
                ax.set_ylabel(r'Flux intensity (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$)', fontsize=15)
            if j == len(rad_bin) - 2:
                ax.tick_params(labelsize=15)
                ax.set_xlabel(r'Rest-frame wavelength ($\mu$m)', fontsize=15)

    plt.show()
    # plt.savefig(savefig_path + 'specs_' + gal_name + '.pdf')


def radial_gradients(save_info=False):
    diag_list = ['K19N2O2', 'M08R23', 'M13N2']
    color_list = [purple, turquoise, red]


    if save_info:
        rad_info = open(output_path + 'met_grads_radbin' + str(round(rad_bin[1], 1)) + '.csv', 'w')
        rad_info.write('name')
        for diag in diag_list:
            rad_info.write(',dZdr_' + diag + ',e_dZdr_' + diag)
        rad_info.write('\n')
        rad_info.close()
        rad_info = open(output_path + 'met_grads_radbin' + str(round(rad_bin[1], 1)) + '.csv', 'a+')

    figsize = (10, 12)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=.08, bottom=.08, right=.98, top=.98, wspace=0., hspace=0.)
    
    l = len(galaxy_list)
    matrix = np.zeros((3, l, len(rad_bin) - 1))

    for i in range(l + 1):
        ax = plt.subplot(11, 4, i + 1)
        ax.set_xlim(.2, 16)
        ax.set_ylim(7.9, 9.5)

        if i < l:
            gal_name = galaxy_list[i]
            print(gal_name)
            if save_info:
                rad_info.write('%s' %(gal_name))
            dp = DataProd(gal_name)
            rad_info = rad_info if save_info else None
            for k, (diag, color) in enumerate(zip(diag_list, color_list)):
                matrix[k, i, :] = dp.plot_grad_annuli(ax, diag, color, zorder=k, rad_info=rad_info)[1]
            # The 16th and 84th percentiles in each radial bin.
            if len(rad_bin) == 4:
                K19N2O2_y1, K19N2O2_y2 = [8.48, 8.49, 8.37], [8.80, 8.74, 8.86]
                N2_y1, N2_y2      =      [8.37, 8.34, 8.46], [8.61, 8.59, 8.66]
            else:
                K19N2O2_y1, K19N2O2_y2 = [8.42, 8.30], [8.75, 8.81]
                N2_y1, N2_y2      =      [8.37, 8.41], [8.59, 8.66]
            ax.fill_between(dp.x, K19N2O2_y1, K19N2O2_y2, color=purple, alpha=.1)
            ax.fill_between(dp.x, N2_y1, N2_y2, color=red, alpha=.1)
            ax.annotate('KURVS ' + str(1 + int(i >= 36) + i), (.5, .1),
                        xycoords='axes fraction', fontsize=15, zorder=10)
        print('\n')
        if save_info:
            rad_info.write('\n')

        if i == 20:
            ax.set_ylabel('12 + log(O/H)', fontsize=15)
        else:
            if i % 4 == 0:
                ax.tick_params(labelsize=15)
            else:
                ax.tick_params(labelleft=False, labelsize=15)
        if i >= 40:
            ax.set_xlabel('Galactocentric\nradius (kpc)', fontsize=15)
        else:
            ax.tick_params(labelbottom=False, labelsize=15)
        if i == 43:
            for diag, color in zip(diag_list, color_list):
                ax.scatter(np.nan, np.nan, color=color, s=50, label=diag[3:])
            ax.legend(loc='upper right', fontsize=10)
            
    print('K19N2O2: ', np.nanpercentile(matrix[0], 16, axis=0), np.nanpercentile(matrix[0], 84, axis=0))
    print('M13N2: ',   np.nanpercentile(matrix[2], 16, axis=0), np.nanpercentile(matrix[2], 84, axis=0))
    # Get the 68% envelop of all the radial distributions.

    if save_info:
        rad_info.close()

    # plt.show()
    plt.savefig(savefig_path + 'met_grads_radbin' + str(round(rad_bin[1], 1)) + '.pdf')



def compare_gradients(between):

    if between == 'diagnostics':
        plt.figure(figsize=(8, 8))
        left, bottom, width, height, spacing = .18, .18, .6, .6, 0.

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(labelleft=False)
        lim = -.4, .4
        
        x, xerr = u.get_grad('dZdr_M13N2'), u.get_grad('e_dZdr_M13N2')
        y, yerr = u.get_grad('dZdr_K19N2O2'),  u.get_grad('e_dZdr_K19N2O2')
        ax_scatter.scatter(x, y, fc='lightblue', ec='k', s=100, zorder=2)
        ax_scatter.vlines(x, y - yerr, y + yerr, color='lightblue', zorder=1)
        ax_scatter.hlines(y, x - xerr, x + xerr, color='lightblue', zorder=1)

        common = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[common], y[common]
        print(u.Pearson(x, y, xerr[common], yerr[common]))
        ax_histx.hist(x.tolist(), bins=np.arange(-2, 2, .02), histtype='step', fill=True,
                      facecolor='w', edgecolor='lightblue', linewidth=3)
        ax_histx.axvline(x=0, ls=':', color='gray')
        ax_histx.set_xlim(lim)
        ax_histx.set_ylim(.1, 4.5)
        ax_histx.set_ylabel('$N$', fontsize=20)
        ax_histx.tick_params(axis='y', labelsize=20)
        ax_histy.hist(y.tolist(), bins=np.arange(-2, 2, .02), histtype='step', fill=False,
                      orientation='horizontal', facecolor='w', edgecolor='lightblue', linewidth=3)
        ax_histy.axhline(y=0, ls=':', color='gray')
        ax_histy.set_ylim(lim)
        ax_histy.set_xlim(.1, 4.5)
        ax_histy.set_xlabel('$N$', fontsize=20)
        ax_histy.tick_params(axis='x', labelsize=20)

        ax_scatter.set_xlim(lim)
        ax_scatter.set_ylim(lim)
        ax_scatter.set_xlabel(r'$\nabla$(O/H)$_{\mathrm{N2}}$ (dex kpc$^{-1}$)',  fontsize=20)
        ax_scatter.set_ylabel(r'$\nabla$(O/H)$_{\mathrm{N2O2}}$ (dex kpc$^{-1}$)',  fontsize=20)
        ax = ax_scatter
        ax.fill_between(np.arange(-1, 1, .01), np.arange(-1, 1, .01) - 0.05,
                        np.arange(-1, 1, .01) + 0.05, color='lightblue', alpha=.2)

    elif between == 'bin_width':
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=.18, bottom=.18, right=.98, top=.98)
        ax = plt.subplot(111)

        x, xerr = u.get_grad('dZdr_M13N2', bin_width=0.5), u.get_grad('e_dZdr_M13N2', bin_width=0.5)
        y, yerr = u.get_grad('dZdr_M13N2', bin_width=0.8), u.get_grad('e_dZdr_M13N2', bin_width=0.8)
        ax.scatter(x, y, fc=red, ec='k', s=100, zorder=3, label='N2')
        ax.vlines(x, y - yerr, y + yerr, color=red, zorder=2)
        ax.hlines(y, x - xerr, x + xerr, color=red, zorder=1)
        good = ~np.isnan(x) & ~np.isnan(y)
        x1, xerr1, y1, yerr1, good1 = x, xerr, y, yerr, good
        x, xerr = u.get_grad('dZdr_K19N2O2', bin_width=0.5), u.get_grad('e_dZdr_K19N2O2', bin_width=0.5)
        y, yerr = u.get_grad('dZdr_K19N2O2', bin_width=0.8), u.get_grad('e_dZdr_K19N2O2', bin_width=0.8)
        ax.scatter(x, y, fc=purple, ec='k', s=100, zorder=3, label='N2O2')
        ax.vlines(x, y - yerr, y + yerr, color=purple, zorder=2)
        ax.hlines(y, x - xerr, x + xerr, color=purple, zorder=1)
        good = ~np.isnan(x) & ~np.isnan(y)

        x = np.concatenate([x1, x])
        y = np.concatenate([y1, y])
        xerr = np.concatenate([xerr1, xerr])
        yerr = np.concatenate([yerr1, yerr])
        good = np.concatenate([good1, good])
        print(u.Pearson(x[good], y[good], xerr[good], yerr[good]))

        ax.set_xlim(-.25, .2)
        ax.set_ylim(-.25, .2)
        ax.set_xlabel(r'$\nabla$(O/H) (dex kpc$^{-1}$, $3\times0.5$")',  fontsize=20)
        ax.set_ylabel(r'$\nabla$(O/H) (dex kpc$^{-1}$, $2\times0.8$")',  fontsize=20)
        ax.legend(loc='upper left', fontsize=20)

    elif between == 'author':
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=.18, bottom=.18, right=.98, top=.98)
        ax = plt.subplot(111)

        x, xerr = u.get_grad('dZdr_M13N2')[:22], u.get_grad('e_dZdr_M13N2')[:22]
        y, yerr = u.get_grad_Gillman()
        good = ~np.isnan(x) & ~np.isnan(y)
        print(u.Pearson(x[good], y[good], xerr[good], yerr[good]))
        x, xerr, y, yerr = x[good], xerr[good], y[good], yerr[good]
        ax.scatter(x, y, fc=red, ec='k', s=100, zorder=2, label='S/N$>4$')
        ax.hlines(y, x - xerr, x + xerr, color=red, zorder=1)
        ax.vlines(x, y - yerr, y + yerr, color=red, zorder=1)


        ax.set_xlim(-.25, .2)
        ax.set_ylim(-.25, .2)
        ax.set_xlabel(r'$\nabla$(O/H)$_{\mathrm{N2}}$ (dex kpc$^{-1}$, this work)',  fontsize=20)
        ax.set_ylabel(r'$\nabla$(O/H)$_{\mathrm{N2}}$ (dex kpc$^{-1}$, Gillman et al. 2022)',  fontsize=20)
        ax.legend(loc='upper left', fontsize=20)

    else:
        raise ValueError("between must be 'diagnostics', 'bin_width', 'author', or 'SNR'!")
    
    ax.tick_params(labelsize=20)
    ax.plot(np.arange(-1, 1, .01), np.arange(-1, 1, .01), ls='--', color='gray', zorder=0)
    ax.axhline(y=0, ls=':', color='gray', zorder=0)
    ax.axvline(x=0, ls=':', color='gray', zorder=0)
    

    # plt.show()
    plt.savefig(savefig_path + 'met_grads_compared_between_' + between + '.pdf')


def grad_evol():

    fig = plt.figure(figsize=(8, 10))
    plt.gca().invert_xaxis()
    plt.subplots_adjust(left=.16, bottom=.1, right=.96, top=.96, wspace=0)

    ax = plt.subplot(111)

    m = []
    for gal_name in galaxy_list:
        if True:  # Try K19N2O2.
            y  =   u.read_grad(gal_name,   'dZdr_K19N2O2', bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_K19N2O2', bin_width=0.5)
            marker = 's'
        if np.isnan(y):  # If K19N2O2 is not available, try M08R23.
            y  =   u.read_grad(gal_name,   'dZdr_M08R23',  bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_M08R23',  bin_width=0.5)
        if np.isnan(y):  # If R23 is not available, try M13N2.
            y  =   u.read_grad(gal_name,   'dZdr_M13N2',   bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_M13N2',   bin_width=0.5)
            marker ='o'
        if np.isnan(y):  # If M13N2 is not available, skip.
            continue
        z = u.read_phot(gal_name, 'redshift')
        ax.scatter(z, y, fc='w', ec='k', marker='H', s=100, zorder=4)
        ax.vlines(z, y - yerr, y + yerr, color='gray', zorder=3)
        m.append(y)

    print(np.nanpercentile(m, 50), np.nanpercentile(m, 16), np.nanpercentile(m, 84))
    line0, label0, line1, label1, line2, label2 = [], [], [], [], [], []
    line0.append(ax.scatter(*u.get_MGz('Yuan'), fc=red, ec='k', marker='o', s=40, zorder=2))
    label0.append('Yuan+11')
    line0.append(ax.scatter(*u.get_MGz('Queyrel'), fc='darkorange', ec='k', marker='o', s=40, zorder=2))
    label0.append('Queyrel+12')
    line0.append(ax.scatter(*u.get_MGz('Swinbank'), fc='gold', ec='k', marker='o', s=40, zorder=2))
    label0.append('Swinbank+12')
    line0.append(ax.scatter(*u.get_MGz('Leethochawalit'), fc=green, ec='k', marker='o', s=40, zorder=2))
    label0.append('Leethochawalit+16')
    xx = np.arange(1.3, 1.66, 0.001)
    line0.append(ax.fill_between(xx, -0.03, 0.06, color=turquoise, alpha=.5, zorder=0))
    label0.append('Wuyts+16')
    ax.scatter(*u.get_MGz('Wang17'), fc=blue, ec='k', marker='D', s=40, zorder=2)
    line0.append(ax.scatter(*u.get_MGz('Wang19'), fc=blue, ec='k', marker='D', s=40, zorder=2))
    label0.append('Wang+17,19')
    line0.append(ax.scatter(*u.get_MGz('Curti'), fc=purple, ec='k', marker='D', s=40, zorder=2))
    label0.append('Curti+20')

    # line1.append(ax.plot(np.arange(0, 10, .001), 0.0253 * np.arange(0, 10, .001) - 0.0380,
    #              color='tomato', ls='-', lw=2, zorder=1)[0])
    # label1.append('Chiappini+01')
    line2.append(ax.fill_between(np.linspace(0, 10, 100), .01*np.ones(100), -.025*np.ones(100),
                 color='gray', alpha=.2, zorder=0))
    label2.append('Chiappini+01,\nMollá+19,\nEAGLE (Tissera+22)\n' +
                  'FIRE 2 (Bellardini+22,\nSun+24)')
    line1.append(ax.plot(*u.get_MGz('MUGS'), color='pink', ls='-', lw=2, zorder=1)[0])
    label1.append('MUGS (normal\nfeedback, Stinson+10)')
    line1.append(ax.fill_between(u.get_MGz('RaDES_lower')[0], u.get_MGz('RaDES_lower')[1],
                 u.get_MGz('RaDES_upper')[1], color='g', alpha=.2, zorder=0))
    label1.append('RaDES (Pilkington+12)')
    line1.append(ax.plot(*u.get_MGz('MAGICC'), color='pink', ls='--', lw=2, zorder=1)[0])
    label1.append('MAGICC (enhanced\nfeedback, Gibson+13)')
    line1.append(ax.axhline(y=-0.05, color='steelblue', ls='-', lw=2, zorder=1))
    label1.append('Mott+13 (variable SFE)')
    line1.append(ax.plot(*u.get_MGz('Mott'), color='steelblue', ls='--', lw=2, zorder=1)[0])
    label1.append('Mott+13 (constant SFE)')
    line1.append(ax.fill_between(np.arange(1.3, 1.4, 1e-4), -0.135, 0.015, color='m', alpha=.2, zorder=0))
    label1.append('FIRE (Ma+17)')
    # line1.append(ax.plot(np.arange(0, 10, .001), -0.008 * np.arange(0, 10, .001) - 0.0065,
    #              color='y', ls='-', lw=2, zorder=1)[0])
    # label1.append('Mollá+19')
    line1.append(ax.plot(np.arange(0, 10, .001), -.025 * np.arange(0, 10, .001) - 0.025,
                 color='magenta', ls='-', lw=2, alpha=.5, zorder=1)[0])
    label1.append('TNG50 (Hemler+21)')
    # line1.append(ax.axhline(y=0, color='slateblue', ls='-', lw=2, alpha=.5, zorder=1))
    # label1.append('EAGLE (Tissera+22)')
    # line1.append(ax.plot(*u.get_MGz('Bellardini'), color='seagreen', ls='-', lw=2, zorder=1)[0])
    # label1.append(r'FIRE 2 ($z<1.5$,' + '\nBellardini+22)')
    # line1.append(ax.plot(*u.get_MGz('Sun'), color='peru', ls='-', lw=2, zorder=1)[0])
    # label1.append(r'FIRE 2 ($z<3$, Sun+24)')
    line1.append(ax.plot(*u.get_MGz('Acharyya'), color='skyblue', ls='-', lw=2, zorder=1)[0])
    label1.append('FOGGIE (Acharyya+25)')

    line0.append(ax.scatter(np.nan, np.nan, fc='w', ec='k', marker='H', s=100))
    label0.append('This work')
    ax.axhline(y=0, color='gray', ls=':', zorder=0)
    ax.errorbar(1.0, .1, yerr=.04, color='gray', capsize=4)
    ax.set_xlim(1.83, .97)
    ax.set_ylim(-.48, .55)
    ax.set_xlabel('Redshift', fontsize=20)
    ax.set_ylabel(r'$\nabla$(O/H) (dex kpc$^{-1}$)', fontsize=20)
    ax.tick_params(labelsize=20)

    legend0 = ax.legend(line0, label0, loc='lower right', fontsize=14)
    legend1 = ax.legend(line1, label1, loc='upper right', fontsize=14)
    legend2 = ax.legend(line2, label2, loc='upper left',  fontsize=14)
    plt.gca().add_artist(legend0)
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # plt.show()
    plt.savefig(savefig_path + 'met_grad_evol.pdf')


def modelled_gradients():

    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=.12, bottom=.15, right=.98, top=.80, wspace=0)

    ax1 = plt.subplot(121)
    ax1.set_xscale('log')
    ax2 = plt.subplot(122)

    m_arr, y_arr = [], []
    for gal_name in galaxy_list:
        Vrot, dVrot, vel_disp, dvel_disp = u.read_Vrot_sigma(gal_name)
        Re = u.read_phot(gal_name, 're') * 1774 * 4.848e-3
        logMstar = u.read_phot(gal_name, 'logMstar')
        V = ufloat(Vrot, dVrot)
        sigma = ufloat(vel_disp, dvel_disp)
        V_sigma = V / sigma
        if True:   # Try K19N2O2.
            y  =   u.read_grad(gal_name,   'dZdr_K19N2O2', bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_K19N2O2', bin_width=0.5)
        if np.isnan(y):  # If K19N2O2 is not available, try M08R23.
            y  =   u.read_grad(gal_name,   'dZdr_M08R23',  bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_M08R23',  bin_width=0.5)
        if np.isnan(y):  # If M08R23 is not available, try M13N2.
            y  =   u.read_grad(gal_name,   'dZdr_M13N2',   bin_width=0.5)
            yerr = u.read_grad(gal_name, 'e_dZdr_M13N2',   bin_width=0.5)
        if np.isnan(y):  # If M13N2 is not available, skip.
            continue
        m_arr.append(logMstar)
        y_arr.append(y)
        im = ax1.scatter(V_sigma.n, y, c=Re, ec='k', marker='H', s=100,
                         cmap=plt.cm.RdYlBu_r, vmin=1, vmax=10, zorder=4)
        if 0.7 < V_sigma.n < 0.93 and -0.15 < y < -0.10:
            print(gal_name, V_sigma.n, y)
        ax1.vlines(V_sigma.n, y - yerr, y + yerr, color='gray', zorder=3)
        ax1.hlines(y, V_sigma.n - V_sigma.s, V_sigma.n + V_sigma.s, color='gray', zorder=3)
        im = ax2.scatter(logMstar, y, c=Re, ec='k', marker='H', s=100,
                         cmap=plt.cm.RdYlBu_r, vmin=1, vmax=10, zorder=4)
        ax2.vlines(logMstar, y - yerr, y + yerr, color='gray', zorder=2)
        ax2.hlines(y, logMstar - .2, logMstar + .2, color='gray', zorder=2)
    st = binned_statistic(m_arr, y_arr, bins=np.arange(9.2, 12, .4))
    ax2.scatter(st.bin_edges[:-1] + np.diff(st.bin_edges)[0], st.statistic, marker='D',
                s=500, fc='w', ec='k', lw=1.5, zorder=0)

    cax = fig.add_axes([0.12, 0.93, 0.86, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'$R_e$ (kpc)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax1.set_xlim(.2, 10)
    ax1.set_ylim(-.32, .24)
    ax1.tick_params(labelsize=20)
    ax1.set_xlabel(r'$V_{\phi} / \sigma$', fontsize=20)
    ax1.set_ylabel(r'$\nabla$(O/H) (dex kpc$^{-1}$)', fontsize=20)
    ax2.set_xlim(9.2, 11.8)
    ax2.set_ylim(-.32, .24)
    ax2.set_xlabel(r'log($M_*$/M$_{\odot}$)', fontsize=20)
    ax2.tick_params(labelleft=False, labelsize=20)


    xx = np.arange(.8, 10, .01)
    ax1.plot(xx, u.get_Vrot_sigma_model('upper')(xx), color='k', ls='--', zorder=2)
    ax1.plot(xx, u.get_Vrot_sigma_model('lower')(xx), color='k', ls='--', zorder=2)
    ax1.fill_between(xx, u.get_Vrot_sigma_model('lower')(xx), u.get_Vrot_sigma_model('upper')(xx),
        color='lightgray', zorder=1)
    ax1.fill_between(np.arange(1.5, 4, 1e-3), -1, 1, color='lightgray', alpha=.5, zorder=0)
    ax1.axhline(y=0, color='gray', ls=':', zorder=0)
    ax1.scatter(np.nan, np.nan, fc='w', ec='k', marker='H', s=100, label='This work')
    ax1.legend(loc='upper left', fontsize=15)

    ax2.axhline(y=0, color='gray', ls=':', zorder=0)
    ax2.plot(*u.get_MGM('Poetrodjojo'), color=blue, lw=2, zorder=1, label=r'$z\sim0$ (Poetrodjojo+21)')
    ax2.plot(*u.get_MGM('Sharda'), color=red, lw=2, zorder=1, label=r'$z=0$ (Sharda+21)')
    ax2.scatter(*u.get_MGM('Wuyts'), fc=green, ec='k', marker='s', zorder=1, s=200, label=r'$z=1.5$ (Wuyts+16)')
    ax2.fill_between(u.get_MGM('Hemler_lower')[0], u.get_MGM('Hemler_lower')[1],
                     u.get_MGM('Hemler_upper')[1], color=green, alpha=.3, zorder=1, label=r'$z\sim1.5$ (Hemler+21)')
    ax2.legend(loc='lower right', fontsize=15)

    # plt.show()
    plt.savefig(savefig_path + 'modelled_met_grads.pdf')





# plot_basic_info()

# show_integ_info()
# write_integ_info('CDFS')
# write_integ_info('COSMOS')
# plot_bpt_info()

# stacking_galaxy(Av_good_list)
# stacking_galaxy(galaxy_list)
# Av_gas_star()

# write_integ_met('CDFS')
# write_integ_met('COSMOS')
# plot_mzr_info()
# corner_diag()

# show_radial_info('cdfs_24904')
# show_radial_info('cdfs_26404')
# show_radial_info('cdfs_27318')
# show_radial_info('COS4_18969')
# show_radial_info('fmos_sf_strong_48_sha_lm')
# show_radial_info('cdfs_30732')
# show_radial_info('cosmos_128449')
# show_radial_info('COS4_11696')
# show_radial_info('cosmos_129830_sha_hm')


# stacking_galaxy(galaxy_list, mask=(rad_bin[0], rad_bin[1]))
# stacking_galaxy(galaxy_list, mask=(rad_bin[1], rad_bin[2]))
# stacking_galaxy(galaxy_list, mask=(rad_bin[2], rad_bin[3]))

# radial_gradients()
# compare_gradients(between='diagnostics')
# compare_gradients(between='bin_width')
# compare_gradients(between='author')

# grad_evol()
modelled_gradients()

