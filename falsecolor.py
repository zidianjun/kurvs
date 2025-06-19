
import constant
from config import rad_bin, bin_size, n_re
from paths import *
from utils import get_field, read_phot


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os


class KMOSFoV(object):
    def __init__(self, gal_name, ax, area='resolved', plot_scale=True, fontsize=15):
        self.gal_name = gal_name
        self.area = area
        self.field = get_field(gal_name)
        jwst_dir = data_path + 'images/JWST_col_images/' + self.gal_name + '.fits'
        hst_dir  =  data_path + 'images/HST_col_images/' + self.gal_name + '.fits'
        self.im_src, self.dir = ('JWST', jwst_dir) if os.path.isfile(jwst_dir) else ('HST', hst_dir)
        self.app = getattr(constant, self.im_src + '_arcsec_per_pix')
        self.Kbin = getattr(constant, self.im_src + '_KMOS_bin_pix_size')
        self.Kout = getattr(constant, self.im_src + '_KMOS_outer_size')
        img = self.get_local_im()
        h, w, c = img.shape
        ax.imshow(img, origin='lower', interpolation='none')
        self._rotated_ellipses(ax, fontsize=fontsize, plot_scale=plot_scale)
        ax.set_xlim(0, w - 1)
        ax.set_ylim(0, h - 1)
        ax.tick_params(labelbottom=False, labelleft=False, direction='in')

    def get_local_im(self):
        large_im = fits.open(self.dir, ignore_missing_simple=True)[0].data
        h, w = large_im.shape[:2]
        half_h, half_w = h // 2, w // 2
        search_rad = 20
        n = 2 if self.gal_name in ['cdfs_29589', 'cdfs_30561'] else 3
        search_area = np.sum(large_im[half_h-search_rad:half_h+search_rad,
            half_w-search_rad:half_w+search_rad, :n], axis=2)
        ind = np.argmax(search_area)
        if np.max(search_area) == 2.:
            ind = 2 * search_rad ** 2 + search_rad
        x1, y1 = ind % (2*search_rad), ind // (2*search_rad)
        cx, cy = half_w - search_rad + x1, half_h - search_rad + y1
        return large_im[cy-self.Kout:cy+self.Kout, cx-self.Kout:cx+self.Kout, :]


    def _rotated_ellipses(self, ax, fontsize=15, plot_scale=True):
        Re = read_phot(self.gal_name, 're')
        color, arcs, letters = ('c', [n_re * Re], ['']) if self.area == 'integ' else (
            'y', rad_bin[1:], ['A', 'B', 'C'])
        cosi = np.cos(read_phot(self.gal_name, 'inc_deg') * np.pi / 180)
        pa_deg = read_phot(self.gal_name, 'pa')
        for arc, letter in zip(arcs, letters):
            a = arc / self.app
            p = np.arange(0, 2*np.pi, 0.01)
            xpos = a*np.cos(p) + self.Kout
            ypos = a*cosi*np.sin(p) + self.Kout
            theta_deg = pa_deg + 90 + (pa_deg < 0) * 180
            theta = theta_deg * np.pi / 180
            # PA=0 means north, but PA+90 means 0 aligned to x axis.
            # However, rotational matrix is clockwised, thus requiring negative.
            dep_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            xpos, ypos = np.dot(dep_mat, np.stack([xpos - self.Kout, ypos - self.Kout], axis=0))
            ax.plot(xpos + self.Kout, ypos + self.Kout, color=color, ls='-')

            if self.area != 'integ':
                xy = ((a*np.cos(theta) + self.Kout)*1.03, (a*np.sin(theta) + self.Kout)*1.03)
                ax.annotate(letter, xy=xy, color='y', fontsize=fontsize)
                # if plot_scale:
                #     if self.im_src == 'HST':
                #         ax.hlines(6, 47, 47 - 2*self.Kbin, color=color)
                #     else:
                #         ax.hlines(10, 90, 90 - 2*self.Kbin, color=color)



