import os
import re
import sys
import math
import time
import string
import random

import warnings
from functools import partial

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from astropy import wcs
from astropy import units as u
from astropy.io import fits, ascii
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, setdiff, join
from astropy.stats import mad_std, biweight_location, gaussian_fwhm_to_sigma
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats

from photutils import detect_sources, deblend_sources
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture
    
from .io import logger
from .io import save_pickle, load_pickle, check_save_path
from .image import DF_pixel_scale, DF_raw_pixel_scale
from .plotting import LogNorm, AsinhNorm, colorbar

try:
    from reproject import reproject_interp
    reproject_install = True
except ImportError:
    warnings.warn("Package reproject is not installed. No rescaling available.")
    reproject_install = False

# default SE columns for cross_match
SE_COLUMNS = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
              "MAG_AUTO", "FLUX_AUTO", "FWHM_IMAGE", "MU_MAX", "FLAGS"]

### Baisc Funcs ###

def coord_Im2Array(X_IMAGE, Y_IMAGE, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE)-origin, 0)), int(max(round(X_IMAGE)-origin, 0))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    X_IMAGE, Y_IMAGE = y_arr+origin, x_arr+origin
    return X_IMAGE, Y_IMAGE

def fwhm_to_gamma(fwhm, beta):
    """ in arcsec """
    return fwhm / 2. / math.sqrt(2**(1./beta)-1)

def gamma_to_fwhm(gamma, beta):
    """ in arcsec """
    return gamma / fwhm_to_gamma(1, beta)

def Intensity2SB(Intensity, BKG, ZP, pixel_scale=DF_pixel_scale):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
    I = np.atleast_1d(np.copy(Intensity))
    I[np.isnan(I)] = BKG
    if np.any(I<=BKG):
        I[I<=BKG] = np.nan
    I_SB = -2.5*np.log10(I - BKG) + ZP + 2.5 * math.log10(pixel_scale**2)
    return I_SB

def SB2Intensity(SB, BKG, ZP, pixel_scale=DF_pixel_scale):
    """
    Convert surface brightness (mag/arcsec^2)to intensity given the
    background value, zero point and pixel scale.
    
    """
    SB = np.atleast_1d(SB)
    I = 10** ((SB - ZP - 2.5 * math.log10(pixel_scale**2))/ (-2.5)) + BKG
    return I

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def counter(i, number):
    if np.mod((i+1), number//4) == 0:
        logger.info("    - completed: %d/%d"%(i+1, number))


def round_good_fft(x):
    # Rounded PSF size to 2^k or 3*2^k
    a = 1 << int(x-1).bit_length()
    b = 3 << int(x-1).bit_length()-2
    if x>b:
        return a
    else:
        return min(a,b)

def calculate_psf_size(n0, theta_0, contrast=1e5, psf_scale=DF_pixel_scale,
                       min_psf_range=60, max_psf_range=720):
    A0 = theta_0**n0
    opt_psf_range = int((contrast * A0) ** (1./n0))
    psf_range = max(min_psf_range, min(opt_psf_range, max_psf_range))
    
    # full (image) PSF size in pixel
    psf_size = 2 * psf_range // psf_scale
    return round_good_fft(psf_size)
    
def compute_poisson_noise(data, n_frame=1, header=None, Gain=0.37):
    if header is not None:
        try:
            n_frame = np.int(header['NFRAMES'])
        except KeyError:
            n_frame = 1
    G_effective = Gain * n_frame # effecitve gain: e-/ADU
    std_poi = np.nanmedian(np.sqrt(data/G_effective))
    
    if np.isnan(std_poi):
        std_poi = None
        print("Sky Poisson Noise Unavailable.")
    else:
        print("Sky Poisson Noise: %.3f"%std_poi)
                                   
    return std_poi

def extract_bool_bitflags(bitflags, ind):
    from astropy.nddata.bitmask import interpret_bit_flags
    return np.array(["{0:016b}".format(0xFFFFFFFF & interpret_bit_flags(flag))[-ind]
                     for flag in np.atleast_1d(bitflags)]).astype(bool)


### Photometry Funcs ###

def background_stats(data, header, mask, bkg_keyname="BACKVAL", **kwargs):
    """ Check if background stored in header + short stats """
    from astropy.stats import sigma_clipped_stats
    from .io import find_keyword_header
    
    # Short estimate summary
    mean, med, std = sigma_clipped_stats(data, mask, **kwargs)
    logger.info("Background stats: mean = %.2f  med = %.2f  std = %.2f"%(mean, med, std))
    
    # check header key
    bkg = find_keyword_header(header, bkg_keyname)
    if bkg is None: bkg = med
    
    return bkg, std
    
def background_annulus(cen, data, mask,
                       r_in=240., r_out=360, draw=True,
                       **plot_kw):
    """ Extract local background value using annulus """
    data_ = data.copy()
    annulus_aperture = CircularAnnulus(cen, r_in=r_in, r_out=r_out)
    annulus_masks = annulus_aperture.to_mask(method='center')
    annulus_data = annulus_masks.multiply(data_)
    mask_ring = annulus_masks.data
    annulus_data_1d = annulus_data[mask_ring!=0]
    mask_1d = annulus_masks.multiply(mask)[mask_ring!=0]
    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d, mask=mask_1d)
    if draw:
        plt.imshow(np.ma.array(annulus_data, mask=mask_ring==0), **plot_kw)
        plt.show()
        
    return median_sigclip

def background_extraction(field, mask=None, return_rms=True,
                          b_size=64, f_size=3, n_iter=5, **kwargs):
    """ Extract background & rms image using SE estimator with mask """
    from photutils import Background2D, SExtractorBackground
    
    try:
        Bkg = Background2D(field, mask=mask,
                           bkg_estimator=SExtractorBackground(),
                           box_size=b_size, filter_size=f_size,
                           sigma_clip=SigmaClip(sigma=3., maxiters=n_iter),
                           **kwargs)
        back = Bkg.background
        back_rms = Bkg.background_rms
        
    except ValueError:
        img = field.copy()
        if mask is not None:
            img[mask] = np.nan
        back = np.nanmedian(field) * np.ones_like(field)
        back_rms = np.nanstd(field) * np.ones_like(field)
        
    if return_rms:
        return back, back_rms
    else:
        return back

def source_detection(data, sn=2.5, b_size=120,
                     k_size=3, fwhm=3, smooth=True, 
                     sub_background=True, mask=None):
    from astropy.convolution import Gaussian2DKernel
    from photutils import detect_sources, deblend_sources
    
    if sub_background:
        back, back_rms = background_extraction(data, b_size=b_size)
        threshold = back + (sn * back_rms)
    else:
        back = np.zeros_like(data)
        threshold = np.nanstd(data)
    if smooth:
        sigma = fwhm * gaussian_fwhm_to_sigma 
        kernel = Gaussian2DKernel(sigma, x_size=k_size, y_size=k_size)
        kernel.normalize()
    else:
        kernel=None
        
    segm_sm = detect_sources(data, threshold, npixels=5,
                            filter_kernel=kernel, mask=mask)

    data_ma = data.copy() - back
    data_ma[segm_sm.data!=0] = np.nan

    return data_ma, segm_sm


def flattened_linear(x, k, x0, y0):
    """ A linear function flattened at (x0,y0) of 1d array """
    return np.array(list(map(lambda x:k*x + (y0-k*x0) if x>=x0 else y0, x)))
    
def piecewise_linear(x, k1, k2, x0, y0):
    """ A piecewise linear function transitioned at (x0,y0) of 1d array """
    return np.array(list(map(lambda x:k1*x + (y0-k1*x0) if x>=x0 else k2*x + (y0-k2*x0), x)))

def iter_curve_fit(x_data, y_data, func, p0=None,
                   color=None, x_min=None, x_max=None,
                   x_lab='', y_lab='',c_lab='',
                   n_iter=3, k_std=5, draw=True,
                   fig=None, ax=None, **kwargs):
    """ Wrapper for iterative curve_fit """
    
    # min-max cutoff
    if x_min is None: x_min = x_data.min()
    if x_max is None: x_max = x_data.max()
    cut = (x_data>x_min) & (x_data<x_max)
    x_data = x_data[cut]
    y_data = y_data[cut]
    if color is not None: color = color[cut]
    
    # initialize
    x_test = np.linspace(x_min, x_max)
    clip = np.zeros_like(x_data, dtype='bool')
    
    # first curve_fit
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0, **kwargs)
    
    if draw:
        with plt.rc_context({"text.usetex": False}):
            if fig is None: fig = plt.figure()
            if ax is None: ax = fig.add_subplot(1,1,1)
        
    # Iterative sigma clip
    for i in range(n_iter):
        if draw: ax.plot(x_test, func(x_test, *popt),
                         color='r', lw=1, ls='--', alpha=0.2)
        
        x_clip, y_clip = x_data[~clip], y_data[~clip]
        popt, pcov = curve_fit(func, x_clip, y_clip, p0=p0, **kwargs)

        # compute residual and stddev
        res = y_data - func(x_data, *popt)
        std = mad_std(res)
        clip = res**2 > (k_std*std)**2
    
    # clip function
    clip_func = lambda x, y: (y - func(x, *popt))**2 > (k_std*std)**2
    
    if draw:
        s = ax.scatter(x_data, y_data, c=color,
                        s=10, cmap='viridis', alpha=0.4)
        ax.scatter(x_data[clip], y_data[clip], lw=2, s=25,
                    facecolors='none', edgecolors='orange', alpha=0.7)
        ax.plot(x_test, func(x_test, *popt), color='r')
        
        if plt.rcParams['text.usetex']:
            c_lab = c_lab.replace('_','$\_$')
            x_lab = x_lab.replace('_','$\_$')
            y_lab = y_lab.replace('_','$\_$')
        
        if color is not None:
            with plt.rc_context({"text.usetex": False}):
                fig.colorbar(s, label=c_lab)
        
        ax.set_xlim(x_min, x_max)
        invert = lambda lab: ('MAG' in lab) | ('MU' in lab)
        if invert(x_lab): ax.invert_xaxis()
        if invert(y_lab): ax.invert_yaxis()
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        
    return popt, pcov, clip_func
    
def identify_extended_source(SE_catalog, mag_limit=15, mag_saturate=13.5, draw=True):
    """ Empirically pick out (bright) extended sources in the SE_catalog.
        The catalog need to contain following columns:
        'MAG_AUTO', 'MU_MAX', 'ELLIPTICITY', 'CLASS_STAR' """
    
    bright = SE_catalog['MAG_AUTO'] < mag_limit
    SE_bright = SE_catalog[bright]
    if len(SE_bright)>0:
        x_data, y_data = SE_bright['MAG_AUTO'], SE_bright['MU_MAX']
    else:
        return SE_catalog, None
    
    MU_satur_0 = np.quantile(y_data, 0.001) # guess of saturated MU_MAX
    MAG_satur_0 = mag_saturate # guess of saturated MAG_AUTO
    
    # Fit a flattened linear
    logger.info("Fit an empirical relation to exclude extended sources...")
    popt, _, clip_func = iter_curve_fit(x_data, y_data, flattened_linear,
                                        p0=(1, MAG_satur_0, MU_satur_0),
                                        x_max=mag_limit, x_min=max(7,np.min(x_data)),
                                        draw=draw, c_lab='CLASS_STAR',
                                        color=SE_bright['CLASS_STAR'],
                                        x_lab='MAG_AUTO',y_lab='MU_MAX')
    if draw: plt.show()
    
    mag_saturate = popt[1]
    logger.info("Saturation occurs at mag = {:.2f}".format(mag_saturate))

    # pick outliers in the catalog
    outlier = clip_func(SE_catalog['MAG_AUTO'], SE_catalog['MU_MAX'])
    
    # identify bright extended sources by:
    # (1) elliptical object or CLASS_STAR<0.5
    # (2) brighter than mag_limit
    # (3) lie out of MU_MAX vs MAG_AUTO relation
    is_extend = (SE_catalog['ELLIPTICITY']>0.7) | (SE_catalog['CLASS_STAR']<0.5)
    is_extend = is_extend & bright & outlier
    
    SE_catalog_extend = SE_catalog[is_extend]
    
    if len(SE_catalog_extend)>0:
        SE_catalog_point = setdiff(SE_catalog, SE_catalog_extend)
        return SE_catalog_point, SE_catalog_extend, mag_saturate
    else:
        return SE_catalog, None, mag_saturate
    

def clean_isolated_stars(xx, yy, mask, star_pos, pad=0, dist_clean=60):
    """ Remove items of stars far away from mask """
    
    star_pos = star_pos + pad
    
    clean = np.zeros(len(star_pos), dtype=bool)
    for k, pos in enumerate(star_pos):
        rr = np.sqrt((xx-pos[0])**2+(yy-pos[1])**2)
        if np.min(rr[~mask]) > dist_clean:
            clean[k] = True
            
    return clean
        
def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                   color="steelblue", xunit="pix", yunit="Intensity",
                   seeing=2.5, pixel_scale=DF_pixel_scale, ZP=27.1,
                   sky_mean=884, sky_std=3, dr=1,
                   lw=2, alpha=0.7, markersize=5, I_shift=0,
                   core_undersample=False, figsize=None,
                   label=None, plot_line=False, mock=False,
                   plot=True, errorbar=False,
                   scatter=False, fill=False, use_annulus=False):
                   
    """
    Calculate 1d radial profile of a given star postage.
    """
    
    if mask is None:
        mask =  np.zeros_like(img, dtype=bool)
    if back is None:     
        back = np.ones_like(img) * sky_mean
    bkg_val = np.median(back)
    if cen is None:
        cen = (img.shape[1]-1)/2., (img.shape[0]-1)/2.
    
    if use_annulus:
        img[mask] = np.nan
    
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int(2 * seeing) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * bkg_val
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.00005 * z_diff[-1]))
    r_max = np.min([img.shape[0]//2, np.sqrt(n_pix_max/np.pi)])
    
    if xunit == "arcsec":
        r *= pixel_scale   # radius in arcsec
        r_core *= pixel_scale
        r_max *= pixel_scale
        d_r = dr * pixel_scale
    else:
        d_r = dr
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clip = lambda z: sigma_clip((z), sigma=5, maxiters=5)
        
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:  
            # for undersampled core, bin at int pixels
            bins_inner = np.unique(r[r<r_core]) - 1e-3
        else:
            n_bin_inner = int(min((r_core/d_r*2), 6))
            bins_inner = np.linspace(0, r_core-d_r, n_bin_inner) - 1e-3

        n_bin_outer = np.max([6, np.min([np.int(r_max/d_r/10), 50])])
        if r_max > (r_core+d_r):
            bins_outer = np.logspace(np.log10(r_core+d_r),
                                     np.log10(r_max+2*d_r), n_bin_outer)
        else:
            bins_outer = []
        bins = np.concatenate([bins_inner, bins_outer])
        _, bins = np.histogram(r, bins=bins)
    
    # Calculate binned 1d profile
    r_rbin = np.array([])
    z_rbin = np.array([])
    zerr_rbin = np.array([])
    
    for k, b in enumerate(bins[:-1]):
        r_in, r_out = bins[k], bins[k+1]
        in_bin = (r>=r_in) & (r<=r_out)
        if use_annulus:
            # Fractional ovelap w/ annulus
            annl = CircularAnnulus(cen, abs(r_in)/pixel_scale, r_out/pixel_scale)
            annl_ma = annl.to_mask()
            # Intensity by fractional mask
            z_ = annl_ma.multiply(img)
            
            zb = np.sum(z_[~np.isnan(z_)]) / annl.area
            zerr_b = sky_std / annl.area
            rb = np.mean(r[in_bin])
            
        else:
            z_clip = clip(z[~np.isnan(z) & in_bin])
            if np.ma.is_masked(z_clip):
                z_clip = z_clip.compressed()
            if len(z_clip)==0:
                continue

            zb = np.mean(z_clip)
            zstd_b = np.std(z_clip) if len(z_clip) > 10 else 0
            zerr_b = np.sqrt((zstd_b**2 + sky_std**2) / len(z_clip))
            rb = np.mean(r[in_bin])
           
        z_rbin = np.append(z_rbin, zb)
        zerr_rbin = np.append(zerr_rbin, zerr_b)
        r_rbin = np.append(r_rbin, rb)

    logzerr_rbin = 0.434 * abs( zerr_rbin / (z_rbin-sky_mean))
    
    if yunit == "SB":
        I_rbin = Intensity2SB(z_rbin, BKG=bkg_val,
                              ZP=ZP, pixel_scale=pixel_scale) + I_shift
    
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if yunit == "Intensity":  
            # plot radius in Intensity
            plt.plot(r_rbin, np.log10(z_rbin), "-o", color=color,
                     mec="k", lw=lw, ms=markersize, alpha=alpha, zorder=3, label=label) 
            if scatter:
                I = np.log10(z)
                
            if fill:
                plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                                 color=color, alpha=0.2, zorder=1)
            plt.ylabel("log Intensity")

        elif yunit == "SB":  
            # plot radius in Surface Brightness
            if mock is False:
                I_sky = -2.5*np.log10(sky_std) + ZP + 2.5 * math.log10(pixel_scale**2)

            p = plt.plot(r_rbin, I_rbin, "-o", mec="k",
                         lw=lw, ms=markersize, color=color,
                         alpha=alpha, zorder=3, label=label)
            if scatter:
                I = Intensity2SB(z, BKG=bkg_val,
                                 ZP=ZP, pixel_scale=pixel_scale) + I_shift
                
            if errorbar:
                Ierr_rbin_up = I_rbin - Intensity2SB(z_rbin+zerr_rbin, BKG=bkg_val,
                                 ZP=ZP, pixel_scale=pixel_scale) - I_shift
                Ierr_rbin_lo = Intensity2SB(z_rbin-zerr_rbin, BKG=bkg_val,
                                ZP=ZP, pixel_scale=pixel_scale) - I_rbin + I_shift
                lolims = np.isnan(Ierr_rbin_lo)
                uplims = np.isnan(Ierr_rbin_up)
                Ierr_rbin_lo[lolims] = 99
                Ierr_rbin_up[uplims] = np.nan
                plt.errorbar(r_rbin, I_rbin, yerr=[Ierr_rbin_up, Ierr_rbin_lo],
                             fmt='', ecolor=p[0].get_color(), capsize=2, alpha=0.5)
                
            plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
            plt.gca().invert_yaxis()
            plt.ylim(30,17)

        plt.xscale("log")
        plt.xlim(max(r_rbin[np.isfinite(r_rbin)][0]*0.8, pixel_scale*0.5),
                 r_rbin[np.isfinite(r_rbin)][-1]*1.2)
        if xunit == "arcsec":
            plt.xlabel("Radius [arcsec]")
        else:
            plt.xlabel("radius [pix]")
        
        if scatter:
            plt.scatter(r[r<3*r_core], I[r<3*r_core], color=color, 
                        s=markersize/2, alpha=alpha/2, zorder=1)
            plt.scatter(r[r>=3*r_core], I[r>=3*r_core], color=color,
                        s=markersize/5, alpha=alpha/10, zorder=1)
            

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dz_rbin = np.diff(np.log10(z_rbin))
        dz_cum = np.cumsum(dz_rbin)

        if plot_line:
            r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
            plt.axvline(r_satr,color="k",ls="--",alpha=0.9)
            plt.axvline(r_core,color="k",ls=":",alpha=0.9)
            plt.axhline(I_sky,color="gray",ls="-.",alpha=0.7)
        
    if yunit == "Intensity":
        return r_rbin, z_rbin, logzerr_rbin
    elif yunit == "SB": 
        return r_rbin, I_rbin, None

def make_psf_2D(n_s, theta_s,
                frac=0.3, beta=6.6, fwhm=6.,
                cutoff_param={"cutoff":False, "n_c":4, "theta_c":1200},
                psf_range=1200, pixel_scale=DF_pixel_scale, plot=False):
    
    """
    Make 2D PSF from parameters.
    
    Parameters
    ----------
    n_s : 1d list or array
        Power index of PSF aureole.
    theta_s : 1d list or array
        Transition radii of PSF aureole in arcsec.
    frac: float
        Fraction of aureole [0 - 1]
    beta : float
        Moffat beta
    fwhm : float
        Moffat fwhm in arcsec
    cutoff_param : dict, optional
        Parametets controlling the cutoff.
    psf_range : int, optional, default 1200
        Range of PSF. In arcsec.
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pix
    plot : bool, optional, default False
        Whether to plot the PSF image.
        
    Returns
    -------
    PSF : 2d array
        Image of the PSF. Normalized to 1.
    psf : eldflower.modeling.PSF_Model
        The PSF model object.
    
    """
    
    from .modeling import PSF_Model

    # Aureole Parameters
    params_mpow = {"frac":frac, "fwhm":fwhm, "beta":beta,
                   "n_s":np.atleast_1d(n_s), "theta_s":np.atleast_1d(theta_s)}
    
    params_mpow.update(cutoff_param)
    
    if cutoff_param["cutoff"]:
        psf_range = max(cutoff_param["theta_c"], psf_range)
    
    # Build PSF Model
    psf = PSF_Model(params=params_mpow, aureole_model='multi-power')

    # Build grid of image for drawing
    psf.pixelize(pixel_scale)

    # Generate core and aureole PSF
    psf_c = psf.generate_core()
    psf_e, psf_size = psf.generate_aureole(contrast=1e6,
                                           psf_range=psf_range,
                                           psf_scale=pixel_scale)

    # Plot Galasim 2D model extracted in 1D
    if plot: psf.plot1D(xunit='arcsec')
    
    # Center and grid
    size = int(np.floor(psf_range/pixel_scale) * 2) + 1
    cen = ((size-1)/2., (size-1)/2.)
    x_ = y_ = np.linspace(0,size-1,size)
    xx, yy = np.meshgrid(x_, y_)

    # Draw image of PSF normalized to 1
    PSF_aureole = psf.draw_aureole2D_in_real([cen], Flux=np.array([frac]))[0]
    PSF_core = psf.draw_core2D_in_real([cen], Flux=np.array([1-frac]))[0]
    PSF = PSF_core(xx,yy) + PSF_aureole(xx,yy)
    PSF = PSF/PSF.sum()
    
    return PSF, psf

def make_psf_1D(n_s, theta_s,
                frac=0.3, beta=6.6, fwhm=6.,
                cutoff_param={"cutoff":False, "n_c":4, "theta_c":1200},
                psf_range=1200, pixel_scale=DF_pixel_scale,
                dr=1, mag=0, ZP=0, plot=False):
    """
    Make 1D PSF profiles from parameters.
    
    Parameters
    ----------
    n_s : 1d list or array
        Power index of PSF aureole.
    theta_s : 1d list or array
        Transition radii of PSF aureole.
    frac: float
        Fraction of aureole [0 - 1]
    beta : float
        Moffat beta
    fwhm : float
        Moffat fwhm in arcsec
    cutoff_param : dict, optional
        Parametets controlling the cutoff.
    psf_range : int, optional, default 1200
        Range of PSF. In arcsec.
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pix
    dr : float, optional, default 0.5
        Parameters controlling the radial interval
    mag : float, optional, default 0.5
        Magnitude of the PSF.
    ZP : float, optional, default 0
        Zero point.
    plot : bool, optional, default False
        Whether to plot the 1D PSF profile.
        
    Returns
    -------
    r : 1d array
        Radius of the profile in arcsec
    I : 1d array
        Surface brightness in mag/arcsec^2
    D : 2d array
        Image of the PSF.
    
    """
    
    Amp = 10**((mag-ZP)/-2.5)
    if plot:
        print('Scaled 1D PSF to magnitude = ', mag)
        
    size = int(np.floor(psf_range/pixel_scale) * 2) + 1
    cen = ((size-1)/2., (size-1)/2.)
    
    D, psf = make_psf_2D(n_s, theta_s, frac, beta, fwhm,
                         cutoff_param=cutoff_param,
                         pixel_scale=pixel_scale,
                         psf_range=psf_range)
    D *= Amp
    r, I, _ = cal_profile_1d(D, cen=cen, mock=True,
                              ZP=ZP, sky_mean=0, sky_std=1e-9,
                              dr=dr, seeing=fwhm,
                              pixel_scale=pixel_scale,
                              xunit="arcsec", yunit="SB",
                              color="lightgreen",
                              lw=4, alpha=0.9, plot=plot,
                              core_undersample=True)
    if plot:
        plt.xlim(2, max(1e3, np.max(2*theta_s)))
        plt.ylim(24,4)
        for pos in theta_s:
            plt.axvline(pos, ls="--", color="k", alpha=0.3, zorder=0)

    return r, I, D

def calculate_fit_SB(psf, r=np.logspace(0.03,2.5,100), mags=[15,12,9], ZP=27.1):
    
    frac = psf.frac
        
    I_s = [10**((mag-ZP)/-2.5) for mag in mags]
    
    comp1 = psf.f_core1D(r)
    comp2 = psf.f_aureole1D(r)

    I_tot_s = [Intensity2SB(((1-frac) * comp1 + comp2 * frac) * I,
                            0, ZP, psf.pixel_scale) for I in I_s]
    return I_tot_s


### Class & Funcs for measuring scaling ###

def compute_Rnorm(image, mask_field, cen,
                  R=12, wid_ring=1, wid_cross=4,
                  mask_cross=True, display=False):
    """
    Compute the scaling factor using an annulus.
    Note the output values include the background level.
    
    Paramters
    ---------
    image : input image for measurement
    mask_field : mask map with masked pixels = 1.
    cen : center of the target in image coordiante
    R : radius of annulus in pix
    wid_ring : half-width of annulus in pix
    wid_cross : half-width of spike mask in pix
        
    Returns
    -------
    I_mean: mean value in the annulus
    I_med : median value in the annulus
    I_std : std value in the annulus
    I_flag : 0 good / 1 bad (available pixles < 5)
    
    """
    
    if image is None:
        return [np.nan] * 3 + [1]
    
    cen = (cen[0], cen[1])
    anl = CircularAnnulus([cen], R-wid_ring, R+wid_ring)
    anl_ma = anl.to_mask()[0].to_image(image.shape)
    in_ring = anl_ma > 0.5        # sky ring (R-wid, R+wid)
    mask = in_ring & (~mask_field) & (~np.isnan(image))
        # sky ring with other sources masked
    
    # Whether to mask the cross regions, important if R is small
    if mask_cross:
        yy, xx = np.indices(image.shape)
        rr = np.sqrt((xx-cen[0])**2+(yy-cen[1])**2)
        in_cross = ((abs(xx-cen[0])<wid_cross))|(abs(yy-cen[1])<wid_cross)
        mask = mask * (~in_cross)
    
    if len(image[mask]) < 5:
        return [np.nan] * 3 + [1]
    
    z_ = sigma_clip(image[mask], sigma=3, maxiters=5)
    z = z_.compressed()
        
    I_mean = np.average(z, weights=anl_ma[mask][~z_.mask])
    I_med, I_std = np.median(z), np.std(z)
    
    if display:
        L = min(100, int(mask.shape[0]))
        
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        ax1.imshow(mask, cmap="gray", alpha=0.7)
        ax1.imshow(mask_field, alpha=0.2)
        ax1.imshow(image, cmap='viridis', alpha=0.7,
                   norm=AsinhNorm(0.05, vmin=image.min(), vmax=I_med+50*I_std))
        ax1.plot(cen[0], cen[1], 'r+', ms=10)
        
        ax2.hist(z,alpha=0.7)
        
        # Label mean value
        plt.axvline(I_mean, color='k')
        plt.text(0.5, 0.9, "%.1f"%I_mean,
                 color='darkorange', ha='center', transform=ax2.transAxes)
        
        # Label 20% / 80% quantiles
        I_20 = np.quantile(z, 0.2)
        I_80 = np.quantile(z, 0.8)
        for I, x_txt in zip([I_20, I_80], [0.2, 0.8]):
            plt.axvline(I, color='k', ls="--")
            plt.text(x_txt, 0.9, "%.1f"%I, color='orange',
                     ha='center', transform=ax2.transAxes)
        
        ax1.set_xlim(cen[0]-L//4, cen[0]+L//4)
        ax1.set_ylim(cen[1]-L//4, cen[1]+L//4)
        
        plt.show()
        
    return I_mean, I_med, I_std, 0


def compute_Rnorm_batch(table_target,
                        image, seg_map, wcs,
                        r_scale=12, k_win=1,
                        wid_ring=0.5, wid_cross=4,
                        display=False, verbose=True):
                        
    """
    Compute scaling factors for objects in the table.
    Return an array with measurement and a dictionary containing maps and centers.
    
    Paramters
    ---------
    table_target : astropy.table.Table
        SExtractor table containing measurements of sources.
    image : 2d array
        Full image.
    seg_map : 2d array
        Full segmentation map used to mask nearby sources during the measurement.
    wcs_data : astropy.wcs.wcs
        WCS of image.
    r_scale : int, optional, default 12
        Radius in pixel at which the flux scaling is measured.
    k_win : int, optional, default 1
        Enlargement factor for extracting thumbnails.
    wid_ring : float, optional, default 0.5
        Half-width in pixel of ring used to measure the scaling.
    wid_cross : float, optional, default 4
        Half-width  in pixel of the spike mask when measuring the scaling.
        
    Returns
    -------
    res_norm : nd array
        A N x 5 array saving the measurements.
        [I_mean, I_med, I_std, I_sky, I_flag]
    res_thumb : dict
        A dictionary storing thumbnails, mask, background and center of object.
        
    """
    
    from .image import Thumb_Image

    # Initialize
    res_thumb = {}
    res_norm = np.empty((len(table_target), 5))
    
    # Iterate rows over the target table
    for i, row in enumerate(table_target):
        if verbose: counter(i, len(table_target))
        num, mag_auto = row['NUMBER'], row['MAG_AUTO']
        
        # For brighter sources, use a broader window
        if mag_auto <= 10.5:
            n_win = int(40 * k_win)
        elif 10.5 < mag_auto < 13.5:
            n_win = int(30 * k_win)
        elif 13.5 < mag_auto < 15:
            n_win = int(20 * k_win)
        else:
            n_win = int(10 * k_win)
        
        # Make thumbnail of the star and mask sources
        thumb = Thumb_Image(row, wcs, pixel_scale=DF_pixel_scale)
        thumb.extract_star(image, seg_map, n_win=n_win)
        
        # Measure the mean, med and std of intensity at r_scale
        thumb.compute_Rnorm(R=r_scale,
                            wid_ring=wid_ring,
                            wid_cross=wid_cross,
                            display=display)
                            
        I_flag = thumb.I_flag
        if (I_flag==1) & verbose: logger.debug("Errorenous measurement: #", num)
        
        # Store results as dict (might be bulky)
        res_thumb[num] = {"image":thumb.img_thumb,
                          "mask":thumb.star_ma,
                          "bkg":thumb.bkg,
                          "center":thumb.cen_star}
                          
        # Store measurements to array
        I_stats = ['I_mean', 'I_med', 'I_std', 'I_sky']
        res_norm[i] = np.array([getattr(thumb, attr) for attr in I_stats] + [I_flag])

    return res_norm, res_thumb

def measure_Rnorm_all(table,
                      bounds,
                      wcs_data,
                      image,
                      seg_map=None,
                      r_scale=12,
                      mag_limit=15,
                      mag_saturate=13.5,
                      mag_name='rmag_PS',
                      k_enlarge=1,
                      width_ring=0.5,
                      width_cross=4,
                      obj_name="", display=False,
                      save=True, dir_name='.',
                      read=False, verbose=True):
    """
    Measure intensity at r_scale for bright stars in table.

    Parameters
    ----------
    table : astropy.table.Table
        SExtractor table containing measurements of sources.
    bounds : 1d array or list
        Boundaries of the region in the image [Xmin, Ymin, Xmax, Ymax].
    wcs_data : astropy.wcs.wcs
        WCS of image.
    image : 2d array
        Full image.
    seg_map : 2d array, optional, default None
        Full segmentation map used to mask nearby sources during the measurement.
        If not given, it will be done locally by photutils.
    r_scale : int, optional, default 12
        Radius in pixel at which the flux scaling is measured.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured.
    mag_saturate : float, optional, default 13.5
        Estimate of magnitude at which the image is saturated.
    mag_name : str, optional, default 'rmag_PS'
        Column name of magnitude used in the table.
    k_enlarge : int, optional, default 1
        Enlargement factor for extracting thumbnails.
    width_ring : float, optional, default 0.5
        Half-width in pixel of ring used to measure the scaling.
    width_cross : float, optional, default 4
        Half-width  in pixel of the spike mask when measuring the scaling.
    obj_name : str, optional
        Object name used as prefix of saved output.
    save : bool, optional, default True
        Whether to save output table and thumbnails.
    dir_name : str, optional
        Path of saving. Use currrent one as default.
    read : bool, optional, default False
        Whether to read existed outputs if available.
    
    Returns
    -------
    table_norm : astropy.table.Table
        Table containing measurement results.
        
    res_thumb : dict
        A dictionary storing thumbnails, mask, background and center of object.
        'image' : image of the object
        'mask' : mask map from SExtractor with nearby sources masked (masked = 1)
        'bkg' : estimated local 2d background
        'center' : 0-based centroid of the object from SExtracror
        
    """
    
    msg = "Measure intensity at R = {0} ".format(r_scale)
    msg += "for catalog stars {0:s} < {1:.1f} in ".format(mag_name, mag_limit)
    msg += "{0}.".format(bounds)
    logger.info(msg)
    
    band = mag_name[0]
    range_str = 'X[{0:d}-{2:d}]Y[{1:d}-{3:d}]'.format(*bounds)
    
    fn_table_norm = os.path.join(dir_name, '%s-norm_%dpix_%smag%d_%s.txt'\
                                %(obj_name, r_scale, band, mag_limit, range_str))
    fn_res_thumb = os.path.join(dir_name, '%s-thumbnail_%smag%d_%s.pkl'\
                                %(obj_name, band, mag_limit, range_str))
    
    fn_psf_satck = os.path.join(dir_name, f'{obj_name}-{band}-psf_stack_{range_str}.fits')
    
    if read:
        table_norm = Table.read(fn_table_norm, format="ascii")
        res_thumb = load_pickle(fn_res_thumb)
        
    else:
        tab = table[table[mag_name]<mag_limit]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res_norm, res_thumb = compute_Rnorm_batch(tab, image,
                                                      seg_map, wcs_data,
                                                      r_scale=r_scale,
                                                      wid_ring=width_ring,
                                                      wid_cross=width_cross,
                                                      k_win=k_enlarge,
                                                      display=display,
                                                      verbose=verbose)
        
        keep_columns = ['NUMBER', 'MAG_AUTO', 'MAG_AUTO_corr', 'MU_MAX', mag_name] \
                        + [name for name in tab.colnames
                                if ('IMAGE' in name)|('CATALOG' in name)]
        for name in keep_columns:
            if name not in tab.colnames:
                keep_columns.remove(name)
        table_norm = tab[keep_columns].copy()
    
        for j, colname in enumerate(['Imean','Imed','Istd','Isky','Iflag']):
            if colname=='Iflag':
                col = res_norm[:,j].astype(int)
            else:
                col = np.around(res_norm[:,j], 5)
            table_norm[colname] = col
        
        if save:  # save star thumbnails
            check_save_path(dir_name, overwrite=True, verbose=False)
            save_pickle(res_thumb, fn_res_thumb, 'thumbnail result')
            table_norm.write(fn_table_norm, overwrite=True, format='ascii')
    
    # Stack non-saturated stars to obtain the inner PSF.
    tab_stack = tab[(tab['FLAGS']<=3) & (tab['MAG_AUTO']>mag_saturate)]
    psf_stack = stack_star_image(tab_stack, res_thumb, size=5*r_scale+1)
    
    if save:
        fits.writeto(fn_psf_satck, data=psf_stack, overwrite=True)
        logger.info(f"Saved stacked PSF to {fn_psf_satck}")
            
    return table_norm, res_thumb


### Stacking PSF functions ###

def resample_thumb(image, mask, center):
    """
    Shift and resample the thumb image & mask to have odd dimensions
    and center at the center pixel.
    
    Parameters
    ----------
    
    image : input image, 2d array
    mask : input mask, 2d bool array (masked =1)
    center : center of the target, array or turple
    
    Returns
    -------
    image_ : output image, 2d array
    mask_ : output mask, 2d bool array (masked =1)
    center_ : center of the target after the shift
        
    """

    from scipy.interpolate import RectBivariateSpline

    X_c, Y_c = center
    NY, NX = image.shape

    # original grid points
    Xp, Yp = np.linspace(0, NX-1, NX), np.linspace(0, NY-1, NY)

    rbspl = RectBivariateSpline(Xp, Yp, image, kx=3, ky=3)
    rbspl_ma = RectBivariateSpline(Xp, Yp, mask, kx=1, ky=1)

    # new NAXIS
    NY_ = NX_ = int(np.floor(image.shape[0]/2) * 2) - 3

    # shift grid points
    Xp_ = np.linspace(X_c - NX_//2, X_c + NX_//2, NX_)
    Yp_ = np.linspace(Y_c - NY_//2, Y_c + NY_//2, NY_)

    # resample image
    image_ = rbspl(Xp_, Yp_)
    mask_ = rbspl_ma(Xp_, Yp_) > 0.5

    center_ = np.array([X_c - Xp_[0], Y_c - Yp_[0]])
    return image_, mask_, center_

def stack_star_image(table_stack, res_thumb, size=61):
    """
    Stack images of stars in the table.
    
    Parameters
    ----------
    table_stack : astropy.table.Table
        SExtarctor table of stars to stack
    res_thumb : dict
        the dict containing the thumbnails, masks and centers
    size : int, optional, default 61
        Size of the stacked image in pixel, will be round to odd number.
    
    Returns
    -------
    image_stack : stacked image
    
    """
    from scipy.ndimage import binary_dilation
    
    size = int(size/2) * 2 + 1
    shape = (size, size)
    canvas = np.zeros(shape)
    footprint = np.zeros_like(canvas)
    
    i = 0
    logger.info("Stacking non-staurated stars to obtain the PSF core...")
    for num in table_stack['NUMBER']:

        # Read image, mask and center
        img_star = res_thumb[num]['image']
        mask_star = res_thumb[num]['mask']
        cen_star = res_thumb[num]['center']
        
        # enlarge mask
        for j in range(3):
            mask_star = binary_dilation(mask_star)
        
        shape_star = img_star.shape
        
        if shape_star[0]!=shape_star[1]: continue
           
        # meausre local background
        r_out = min(img_star.shape) * 0.8 //2
        r_in = r_out - 5
        bkg = background_annulus(cen_star, img_star, mask_star, r_in=r_in, r_out=r_out, draw=False)
        
        # resample thumbnail centroid to center
        img_star_, mask_star_, cen_star_ = resample_thumb(img_star, mask_star, cen_star)
        shape_star_ = img_star_.shape
        
        # remove nearby sources
        img_star_ = img_star_ - bkg
        img_star_[mask_star_] = 0
        
        # cutout
        if shape_star_[0] > size:
            dx = (shape_star_[0]-canvas.shape[0])//2
            dy = (shape_star_[1]-canvas.shape[1])//2
            cutout = img_star_[dx:-dx,dy:-dy]
        
            # add to canvas
            canvas += cutout
            footprint += (cutout!=0)
            i += 1
        
    image_stack = canvas/footprint
    logger.info("   - {:d} Stars used for stacking.".format(i))

    return image_stack

def make_global_stack_PSF(dir_name, bounds_list, obj_name, band, overwrite=True):
    """
    Combine the stacked PSF of all regions into one, skip if existed.
    
    Parameters
    ----------
    dir_name : str
        path containing the stacked PSF
    bounds_list : 2D int list / turple
        List of boundaries of regions to be fit (Nx4)
        [[X min, Y min, X max, Y max],[...],...]
    obj_name : str
        Object name
    band : str, 'g' 'G' 'r' or 'R'
        Filter name
    
    """
    
    fn_stack = os.path.join(dir_name, f'{obj_name}-{band}-PSF_stack.fits')
    
    if overwrite or (os.path.isfile(fn_stack)==False):
        for i, bounds in enumerate(bounds_list):
            range_str = 'X[{0:d}-{2:d}]Y[{1:d}-{3:d}]'.format(*bounds)
            fn = os.path.join(dir_name, f'{obj_name}-{band}-psf_stack_{range_str}.fits')
            image_psf = fits.getdata(fn)
            
            if i==0:
                image_stack = image_psf
            else:
                image_stack += image_psf
                
        image_stack = image_stack/image_stack.sum()
        
        if i>0:
            logger.info("Read & stack {:} PSF.".format(i+1))

        fits.writeto(fn_stack, data=image_stack, overwrite=True)
        logger.info("Saved stacked PSF as {:}".format(fn_stack))
    else:
        logger.info("{:} existed. Skip Stack.".format(fn_stack))
        
def montage_psf_image(image_psf, image_wide_psf, r=10, dr=0.5):
    """
    Montage the core of the stacked psf and the wing of the wide psf model.
    
    Parameters
    ----------
    image_psf : 2d array
        The image of the inner PSF.
    image_wide_psf : 2d array
        The image of the wide-angle PSF.
    r : int, optional, default 10
        Radius in pixel at which the PSF is montaged.
    dr : float, optional, default 0.5
        Width of annulus for measuring the scaling.
    
    Returns
    -------
    image_PSF : 2d array
        The image of the output PSF.
        
    """

    image_PSF = image_wide_psf.copy()

    # Wide PSF
    size = image_wide_psf.shape[0]
    cen = ((size-1)/2., (size-1)/2.)
    x_ = y_ = np.linspace(0,size-1,size)
    xx, yy = np.meshgrid(x_, y_)
    rr = np.sqrt((yy-cen[0])**2+(xx-cen[1])**2)

    I_wide = np.median(image_wide_psf[(rr<r+dr)&(rr>r-dr)])

    # Stacked PSF
    size_psf = image_psf.shape[0]
    cen_psf = ((size_psf-1)/2., (size_psf-1)/2.)
    x_psf = y_psf = np.linspace(0,size_psf-1,size_psf)
    xx_psf, yy_psf = np.meshgrid(x_psf, y_psf)
    rr_psf = np.sqrt((yy_psf-cen_psf[0])**2+(xx_psf-cen_psf[1])**2)

    I_psf = np.median(image_psf[(rr_psf<r+dr)&(rr_psf>r-dr)])

    # Montage
    image_PSF[rr<r] = image_psf[rr_psf<r]/ I_psf * I_wide
    image_PSF = image_PSF/image_PSF.sum()

    return image_PSF
    
    
def fit_psf_core_1D(image_psf,
                    frac=0.3, beta=6.6, fwhm=6.,
                    n_s=[3.3, 2.], theta_s=[5, 10**2.],
                    theta_out=30, d_theta=1.,
                    pixel_scale=2.5, beta_max=8.,
                    obj_name="",band="r",
                    draw=True, save=False, save_dir='./'):
    """
    Fit the core parameters from 1D profiles of the input 2D PSF.
    
    Parameters
    ----------
    image_psf : 2d array
        The image of the PSF.
    frac: float
        Fraction of aureole [0 - 1]
    beta : float
        Moffat beta
    fwhm : float
        Moffat fwhm in arcsec
    n_s : 1d list or array
        Power index of PSF aureole.
        Not required to be accurate except n0.
    theta_s : 1d list or array
        Transition radii of PSF aureole.
        Not required to be accurate.
    theta_out : float, optional, default 30
        Max radias in arcsec of the profile.
    d_theta : float, optional, default 1.
        Radial interval of the profile.
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pix
    beta_max : float, optional, default 8.
        Upper bound of the Moffat beta (lower bound is 1.1)
    obj_name : str
        Object name
    band : str, 'g' 'G' 'r' or 'R'
        Filter name
    draw : bool, optional, default True
        Whether to plot the fit.
    save : bool, optional, default True
        Whether to save the plot.
    save_dir : str, optional
        Path of saving plot, default current.
    """
    
    # Center of the input PSF image
    cen = ((image_psf.shape[1]-1)/2., (image_psf.shape[0]-1)/2.)

    # Set grid points
    rp = np.arange(1, theta_out+d_theta,d_theta)
        
    # Calculate 1D profile
    r_psf, I_psf, _ = cal_profile_1d(image_psf, cen=cen, dr=0.5,
                                     ZP=0, sky_mean=0, plot=False,
                                     pixel_scale=pixel_scale, seeing=3,
                                     xunit="arcsec", yunit="SB",
                                     core_undersample=True)

    # Interpolate at grid points
    Ip = np.interp(rp, r_psf, I_psf)

    # Guess and bounds for core params
    p0 = [frac, beta]
    bounds = [(0, 0.5), (1.1, beta_max)]
    
    logger.info("Fitting core parameters from stacked PSF...")
    # Define the target function for fitting the core
    def make_psf_core_1D(r_intp, frac, beta):
        r_1d, I_1d,_ = make_psf_1D(n_s=n_s, theta_s=theta_s,
                                   frac=frac, beta=beta, fwhm=fwhm,
                                   dr=0.5, pixel_scale=pixel_scale)
        I_intp = np.interp(r_intp, r_1d, I_1d)
        return I_intp
    
    # Fit the curve
    popt, pcov = curve_fit(make_psf_core_1D, rp, Ip, p0, bounds=bounds)
    frac, beta = popt
    frac_err, beta_err = np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])
    
    logger.info("   - frac = {:.3f}+/-{:.3f}".format(frac, frac_err))
    logger.info("   - beta = {:.3f}+/-{:.3f}".format(beta, beta_err))

    if draw:
        I_fit = make_psf_core_1D(rp, frac, beta)
        plt.plot(rp, I_fit, 'r-o', ms=5, label='Fit')
        plt.plot(rp, Ip, 'y-o', mfc='None', mec='y', label='Data')

        plt.ylim(I_fit.max()+0.5, I_fit.min()-0.5)
        plt.xlim(-2, theta_out * 1.1)
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("Surface Brightness")
        plt.legend()
        if save:
            fn_plot = f'Fit_core_{obj_name}-{band}.png'
            plt.savefig(os.path.join(save_dir, fn_plot))
        plt.show()

    return frac, beta


### Resampling functions ###

def transform_rescale(val, scale=0.5):
    """ transform coordinates after resampling """
    return (val-1) * scale + scale/2. + 0.5
    
def transform_table_coordinates(table, filename, scale=0.5):
    """ transform coordinates in a table and write to a new one """
    table_ = table.copy()
    
    # transform coordiantes for X/Y_IMAGE and A/B_IMAGE
    for coln in table_.colnames:
        if 'IMAGE' in coln:
            if ('X' in coln) | ('X' in coln):
                table_[coln] = transform_rescale(table[coln], scale)
            else:
                table_[coln] *= scale
                
    table_.write(filename, format='ascii', overwrite=True)
    

def downsample_wcs(wcs_input, scale=0.5):
    """ Downsample the input wcs along an axis using {CDELT, CRPIX} FITS convention """

    header = wcs_input.to_header()
    shape = wcs_input.pixel_shape
    
    if 'CDELT1' in header.keys(): cdname = 'CDELT'
    if 'CD1_1' in header.keys(): cdname = 'CD'
    
    for axis in [1, 2]:
        if cdname=='CDELT':
            cd = 'CDELT{0:d}'.format(axis)
        elif cdname == 'CD':
            cd = 'CD{0:d}_{0:d}'.format(axis)
        else:
            msg = 'Fits header has no CDELT or CD keywords!'
            logger.error(msg)
            raise KeyError(msg)
            
        cp = 'CRPIX{0:d}'.format(axis)
        na = 'NAXIS{0:d}'.format(axis)

        header[cp] = transform_rescale(header[cp], scale)
        header[cd] = header[cd]/scale
        header[na] = int(round(shape[axis-1]*scale))

    return wcs.WCS(header)

def write_downsample_fits(fn, fn_out, scale=0.5, order=3, wcs_out=None):
    """
    Write fits data downsampled by factor. Alternatively a target wcs can be provided.
    
    Parameters
    ----------
    fn: str
        full path of fits file
    fn_out: str
        full path of output fits file
    scale: int, optional, default 0.5
        scaling factor
    order: int, optional, default 3 ('bicubic')
        order of interpolation (see docs of reproject)
    wcs_out: wcs, optional, default None
        output target wcs. must have shape info.
        
    """
    if reproject_install == False:
        return None

    # read fits
    header = fits.getheader(fn)
    data = fits.getdata(fn)
    wcs_input = wcs.WCS(header)

    if (wcs_out is not None) & hasattr(wcs_out, 'pixel_shape'):
        # use input wcs and shape
        shape_out = wcs_out.pixel_shape
    else:
        # make new wcs and shape according to scale factor
        wcs_out = downsample_wcs(wcs_input, scale)
        shape_out = (int(data.shape[0]*scale), int(data.shape[1]*scale))

    # reproject the image by new wcs
    data_rp, _ = reproject_interp((data, wcs_input), wcs_out,
                                   shape_out=shape_out, order=3)

    # write new header
    header_out = wcs_out.to_header()
    for key in ['NFRAMES', 'BACKVAL', 'EXP_EFF', 'FILTNAM']:
        if key in header.keys():
            header_out[key] = header[key]
            
    header_out['RESCALE'] = scale
    
    # write new fits
    fits.writeto(fn_out, data_rp, header=header_out, overwrite=True)
    logger.info('Resampled image saved to: ', fn_out)

    return True

def downsample_segmentation(fn, fn_out, scale=0.5):
    """ Downsample segmentation and write to fits """
    from scipy.ndimage import zoom
    
    if os.path.isfile(fn):
        segm = fits.getdata(fn)
        segm_out = zoom(segm, zoom=0.5, order=1)
        fits.writeto(fn_out, segm_out, overwrite=True)
        
    else:
        pass
    
def process_resampling(fn, bounds, obj_name, band,
                       pixel_scale=DF_pixel_scale, r_scale=12,
                       mag_limit=15, dir_measure='./', work_dir='./',
                       factor=1, verbose=True):
                        
    from .image import ImageList
    
    # turn bounds_list into 2d array
    bounds = np.atleast_2d(bounds).astype(int)
    
    if factor!=1:
        if verbose:
            logger.info('Resampling by a factor of {0:.1g}...'.format(factor))
        
        scale = 1/factor
        
        fn_rp = "{0}_{2}.{1}".format(*os.path.basename(fn).rsplit('.', 1) + ['rp'])
        fn_rp = os.path.join(work_dir, fn_rp)
        
        bounds_rp = np.array([np.round(b_*scale) for b_ in bounds], dtype=int)
        
        # resample image if it does not exist
        if not os.path.exists(fn_rp):
            write_downsample_fits(fn, fn_rp, scale, order=3)
        
        # construct Image List for original image
        DF_Images = ImageList(fn, bounds, obj_name, band,
                              pixel_scale=pixel_scale)
        
        # read faint stars info and brightness measurement
        DF_Images.read_measurement_tables(dir_measure,
                                          r_scale=r_scale,
                                          mag_limit=mag_limit)
                                          
        # new quantities and names
        r_scale *= scale
        pixel_scale /= scale
        obj_name_rp = obj_name + '_rp'
        
        if verbose:
            logger.info('Transforming coordinates for measurement tables...')
            
        for Img, bound, bound_rp in zip(DF_Images, bounds, bounds_rp):
        
            # transform coordinates and write as new tables
            old_range = 'X[{0:d}-{2:d}]Y[{1:d}-{3:d}]'.format(*bound)
            new_range = 'X[{0:d}-{2:d}]Y[{1:d}-{3:d}]'.format(*bound_rp)
            
            table_faint, table_norm = Img.table_faint, Img.table_norm
            
            fn_catalog = os.path.join(dir_measure,
                        "%s-catalog_PS_%s_all.txt"%(obj_name_rp, band.lower()))
            fn_norm = os.path.join(dir_measure, "%s-norm_%dpix_%smag%s_%s.txt"\
                         %(obj_name_rp, r_scale, band.lower(), mag_limit, new_range))
                        
            transform_table_coordinates(table_faint, fn_catalog, scale)
            transform_table_coordinates(table_norm, fn_norm, scale)
            
            # reproject segmentation
            if verbose:
                logger.info('Resampling segmentation for bounds:', bound)
            fn_seg = os.path.join(dir_measure,
                        "%s-segm_%s_catalog_%s.fits"\
                        %(obj_name, band.lower(), old_range))
            fn_seg_out = os.path.join(dir_measure, "%s-segm_%s_catalog_%s.fits"\
                        %(obj_name_rp, band.lower(), new_range))
                        
            downsample_segmentation(fn_seg, fn_seg_out, scale)
        
    else:
        fn_rp, bounds_rp = fn, bounds
        
    return fn_rp, bounds_rp
    

### Catalog / Data Manipulation Helper ###
def id_generator(size=6, chars=None):
    if chars is None:
        chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))
    
def crop_catalog(cat, bounds, keys=("X_IMAGE", "Y_IMAGE"), sortby=None):
    Xmin, Ymin, Xmax, Ymax = bounds
    A, B = keys
    crop = (cat[A]>=Xmin) & (cat[A]<=Xmax) & (cat[B]>=Ymin) & (cat[B]<=Ymax)
    if sortby is not None:
        cat_crop = cat[crop]
        cat_crop.sort(keys=sortby)
        return cat_crop
    else:
        return cat[crop]
        
def crop_pad(image, pad):
    """ Crop the padding of the image """
    shape = image.shape
    return image[pad:shape[0]-pad, pad:shape[1]-pad]
    
def crop_image(data, bounds, wcs=None, draw=False, **kwargs):
    """ Crop the data (and segm map if given) with the given bouds.
        Note boundaries are in 1-based pixel coordianates. """
    
    Xmin, Ymin, Xmax, Ymax = bounds
    
    # X, Y image size
    nX, nY = (Xmax-Xmin, Ymax-Ymin)
    # center in 1-based pixel coordinates
    cen = (Xmin+(nX-1)/2., Ymin+(nY-1)/2.)
    
    # make cutout
    cutout = Cutout2D(data, cen, (nY, nX), wcs=wcs)
    
    if draw:
        from .plotting import draw_bounds
        draw_bounds(data, bounds, **kwargs)
    
    # also return cutout of wcs if given
    if wcs is None:
        return cutout.data
    else:
        return cutout.data, cutout.wcs

def transform_coords2pixel(table, wcs, name='',
                           RA_key="RAJ2000", DE_key="DEJ2000", origin=1):
    """ Transform the RA/DEC columns in the table into pixel coordinates given wcs"""
    coords = np.vstack([np.array(table[RA_key]), 
                        np.array(table[DE_key])]).T
    pos = wcs.wcs_world2pix(coords, origin)
    
    table.add_column(Column(np.around(pos[:,0], 4)*u.pix), name="X_CATALOG")
    table.add_column(Column(np.around(pos[:,1], 4)*u.pix), name="Y_CATALOG")
    table.add_column(Column(np.arange(len(table))+1, dtype=int), 
                     index=0, name="ID"+'_'+name)
    return table

def merge_catalog(SE_catalog, table_merge, sep=5 * u.arcsec,
                  RA_key="RAJ2000", DE_key="DEJ2000", keep_columns=None):
    """ Crossmatch and merge two catalogs by coordinates"""
    
    c_SE = SkyCoord(ra=SE_catalog["X_WORLD"], dec=SE_catalog["Y_WORLD"])
    c_tab = SkyCoord(ra=table_merge[RA_key], dec=table_merge[DE_key])
    
    idx, d2d, d3d = c_SE.match_to_catalog_sky(c_tab)
    match = d2d < sep 
    cat_SE_match = SE_catalog[match]
    cat_tab_match = table_merge[idx[match]]
    
    cat_tab_match.add_column(cat_SE_match["NUMBER"], index=0, name="NUMBER")
    cat_match = join(cat_SE_match, cat_tab_match, keys='NUMBER')
    
    if keep_columns is not None:
        cat_match.keep_columns(keep_columns)
    
    return cat_match

def read_measurement_table(dir_name, bounds0,
                           obj_name='', band='G',
                           pad=50, r_scale=12,
                           mag_limit=15):
    """ Read measurement tables from the directory """
    
    use_PS1_DR2 = True if 'PS2' in dir_name else False
    
    # Magnitude name
    b_name = band.lower()
    mag_name = b_name+'MeanPSFMag' if use_PS1_DR2 else b_name+'mag'
    
    # Clipped bounds
    patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = bounds0

    bounds = (patch_Xmin0+pad, patch_Ymin0+pad,
              patch_Xmax0-pad, patch_Ymax0-pad)

    ## Read measurement for faint stars from catalog
    # Faint star catalog name
    fname_catalog = os.path.join(dir_name, "%s-catalog_PS_%s_all.txt"%(obj_name, b_name))

    # Check if the file exist before read
    assert os.path.isfile(fname_catalog), f"Table {fname_catalog} does not exist!"
    
    logger.debug(f"Reading catalog {fname_catalog}.")
    table_catalog = Table.read(fname_catalog, format="ascii")
    mag_catalog = table_catalog[mag_name]

    # stars fainter than magnitude limit (fixed as background), > 22 is ignored
    table_faint = table_catalog[(mag_catalog>=mag_limit) & (mag_catalog<22)]
    table_faint = crop_catalog(table_faint,
                               keys=("X_CATALOG", "Y_CATALOG"),
                               bounds=bounds)
    
    ## Read measurement for bright stars
    # Catalog name
    range_str = "X[{:d}-{:d}]Y[{:d}-{:d}]"
    range_str = range_str.format(patch_Xmin0, patch_Xmax0, patch_Ymin0, patch_Ymax0)
    fname_norm = os.path.join(dir_name, "%s-norm_%dpix_%smag%s_%s.txt"\
                            %(obj_name, r_scale, b_name, mag_limit, range_str))
    # Check if the file exist before read
    assert os.path.isfile(fname_norm), f"Table {fname_norm} does not exist"
    
    logger.debug(f"Reading catalog {fname_norm}.")
    table_norm = Table.read(fname_norm, format="ascii")

    # Crop the catalog
    table_norm = crop_catalog(table_norm, bounds=bounds0)

    # Do not use flagged measurement
    Iflag = table_norm["Iflag"]
    table_norm = table_norm[Iflag==0]
    
    return table_faint, table_norm
    

def assign_star_props(ZP, sky_mean, image_shape, pos_ref,
                      table_norm, table_faint=None,
                      r_scale=12, mag_threshold=[13.5,12],
                      psf=None, keys='Imed', verbose=True, 
                      draw=True, save=False, save_dir='./'):
    """ Assign position and flux for faint and bright stars from tables. """
    
    from .modeling import Stars

    # Positions & Flux (estimate) of bright stars from measured norm
    star_pos = np.vstack([table_norm["X_CATALOG"],
                          table_norm["Y_CATALOG"]]).T - pos_ref
                         
    mag = table_norm['MAG_AUTO_corr'] if 'MAG_AUTO_corr' in table_norm.colnames else table_norm['MAG_AUTO']
    Flux = 10**((np.array(mag)-ZP)/(-2.5))

    # Estimate of brightness I at r_scale (I = Intensity - BKG) and flux
    z_norm = table_norm['Imed'].data - table_norm['Isky'].data
    z_norm[z_norm<=0] = min(1, z_norm[z_norm>0].min())
    
    # Convert and printout thresholds
    Flux_threshold = 10**((np.array(mag_threshold) - ZP) / (-2.5))
    
    if verbose:
        msg = "Magnitude Thresholds:  {0}, {1} mag"
        msg = msg.format(*mag_threshold)
        logger.info(msg)
        
        msg = "Flux Thresholds: {0}, {1} ADU"
        msg = msg.format(*np.around(Flux_threshold,2))
        logger.info(msg)
        
        try:
            SB_threshold = psf.Flux2SB(Flux_threshold, BKG=sky_mean, ZP=ZP, r=r_scale)
            msg = "Surface Brightness Thresholds: {0}, {1} mag/arcsec^2 "
            msg = msg.format(*np.around(SB_threshold,1))
            msg += "at {0} pix for sky = {1:.3f}".format(r_scale, sky_mean)
            logger.info(msg3)
            
        except:
            pass
            
    # Bright stars in model
    stars_bright = Stars(star_pos, Flux, Flux_threshold=Flux_threshold,
                         z_norm=z_norm, r_scale=r_scale, BKG=sky_mean)
    stars_bright = stars_bright.remove_outsider(image_shape, gap=[3*r_scale, r_scale])
    stars_bright._info()
    
    if (table_faint is not None) & ('MAG_AUTO_corr' in table_faint.colnames):
        table_faint['FLUX_AUTO_corr'] = 10**((table_faint['MAG_AUTO_corr']-ZP)/(-2.5))
        
        try:
            ma = table_faint['FLUX_AUTO_corr'].data.mask
        except AttributeError:
            ma = np.isnan(table_faint['FLUX_AUTO_corr'])

        # Positions & Flux of faint stars from catalog
        star_pos_faint = np.vstack([table_faint["X_CATALOG"].data[~ma],
                                    table_faint["Y_CATALOG"].data[~ma]]).T - pos_ref
        Flux_faint = np.array(table_faint['FLUX_AUTO_corr'].data[~ma])
        
        # Combine two samples, make sure they do not overlap
        star_pos = np.vstack([star_pos, star_pos_faint])
        Flux = np.concatenate([Flux, Flux_faint])
        
    stars_all = Stars(star_pos, Flux, Flux_threshold, BKG=sky_mean)

    if draw:
        stars_all.plot_flux_dist(label='All', color='plum')
        stars_bright.plot_flux_dist(label='Model', color='orange', ZP=ZP,
                                    save=save, save_dir=save_dir)
        plt.show()
        
    return stars_bright, stars_all
    
    
def interp_I0(r, I, r0, r1, r2):
    """ Interpolate I0 at r0 with I(r) between r1 and r2 """
    range_intp = (r>r1) & (r<r2)
    logI0 = np.interp(r0, r[(r>r1)&(r<r2)], np.log10(I[(r>r1)&(r<r2)]))
    return 10**logI0
    
def compute_mean_I(r, I, r1, r2):
    """ Compute mean I under I(r) between r1 and r2 """
    range_intg = (r>r1) & (r<r2)
    r_range = r[range_intg]
    return np.trapz(I[range_intg], r_range)/(r_range.max()-r_range.min())

def fit_n0(dir_measure, bounds,
           obj_name, band, BKG, ZP,
           pixel_scale=DF_pixel_scale,
           fit_range=[20,40], dr=0.1,
           N_fit=15, mag_max=13, mag_limit=15,
           I_norm=24, norm='intp',
           r_scale=12, sky_std=3,
           plot_brightest=True, draw=True,
           save=False, save_dir="./"):
           
    """
    Fit the first component of using bright stars.
    
    Parameters
    ----------
    dir_measure : str
        Directory storing the measurement
    bounds : 1d list, [Xmin, Ymin, Xmax, Ymax]
        Fitting boundary
    band : str, 'g' 'G' 'r' or 'R'
        Filter name
    obj_name : str
        Object name
    BKG : float
        Background value for profile measurement
    ZP : float
        Zero-point
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pix
    fit_range : 2-list, optional, default [20, 40]
        Range for fitting in arcsec
    dr : float, optional, default 0.2
        Profile step paramter
    N_fit : int, optional, default 15
        Number of stars used to fit n0
    mag_max : float, optional, default 13
        Max magnitude of stars used to fit n0
    I_norm : float, optional, default 24
        SB at which profiles are normed
    norm : 'intp' or 'intg', optional, default 'intg'
        Normalization method to scale profiles.
        Use mean value by 'intg', use interpolated value by 'intp'
    r_scale : int, optional, default 12
        Radius (in pix) at which the brightness is measured
        Default is 30" for Dragonfly.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured
    sky_std : float, optional, default 3
        Sky stddev (for display only)
    plot_brightest : bool, optional, default True
        Whether to draw profile of the brightest star
    draw : bool, optional, default True
        Whether to draw profiles and fit process
    save : bool, optional, default True
        Whether to save plot.
    save_dir : str, optional
        Full path of saving plot, default current.
        
    Returns
    -------
    n0 : float
        first power index
    d_n0 : float
        uncertainty of n0
    
    """

    from .modeling import log_linear

    Xmin, Ymin, Xmax, Ymax = bounds
    r1, r2 = fit_range
    r0 = r_scale*pixel_scale

    if  r1<r0<r2:
        # read result thumbnail and norm table
        b = band.lower()
        range_str = f'X[{Xmin}-{Xmax}]Y[{Ymin}-{Ymax}]'
        fn_res_thumb = os.path.join(dir_measure, f'{obj_name}-thumbnail_{b}mag{mag_limit}_{range_str}.pkl')
        fn_tab_norm = os.path.join(dir_measure, f'{obj_name}-norm_{r_scale}pix_{b}mag{mag_limit}_{range_str}.txt')
        res_thumb = load_pickle(fn_res_thumb)
        tab_norm = Table.read(fn_tab_norm, format='ascii')

        if draw:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
        else:
            fig, ax, ax_ins = None, None, None

        # r_rbin: r in arcsec, I_rbin: SB in mag/arcsec^2
        # I_r0: SB at r0, I_rbin: SB in mag/arcsec^2
        r_rbin_all, I_rbin_all = np.array([]), np.array([])
        I_r0_all, In_rbin_all = np.array([]), np.array([])

        tab_fit = tab_norm[tab_norm['MAG_AUTO_corr']<mag_max][:N_fit]
        if len(tab_fit)==0:
            msg = "No enought bright stars in this region. Include n0 in the fitting."
            logger.warning(msg)
            return None, None
            
        logger.info("Fit n0 with profiles of %d bright stars..."%(len(tab_fit)))
        for num in tab_fit['NUMBER']:
            res = res_thumb[num]
            img, ma, cen = res['image'], res['mask'], res['center']
            bkg = np.median(res_thumb[num]['bkg'])
            sky_mean = bkg if BKG is None else BKG
            
            # calculate 1d profile
            r_rbin, I_rbin, _ = cal_profile_1d(img, cen=cen, mask=ma,
                                               ZP=ZP, sky_mean=sky_mean, sky_std=sky_std,
                                               xunit="arcsec", yunit="SB",
                                               errorbar=False, dr=dr,
                                               pixel_scale=pixel_scale,
                                               core_undersample=False, plot=False)

            range_intp = (r_rbin>r1) & (r_rbin<r2)
            if len(r_rbin[range_intp]) > 5:
                if norm=="intp":
                    # interpolate I0 at r0, r in arcsec
                    I_r0 = interp_I0(r_rbin, I_rbin, r0, r1, r2)
                elif norm=="intg":
                    I_r0 = compute_mean_I(r_rbin, I_rbin, r1, r2)

                r_rbin_all = np.append(r_rbin_all, r_rbin)
                I_rbin_all = np.append(I_rbin_all, I_rbin)

                I_r0_all = np.append(I_r0_all, I_r0)
                In_rbin_all = np.append(In_rbin_all, I_rbin-I_r0+I_norm)

                if draw:
                    cal_profile_1d(img, cen=cen, mask=ma, dr=1,
                                   ZP=ZP, sky_mean=sky_mean, sky_std=2.8,
                                   xunit="arcsec", yunit="SB", errorbar=False,
                                   pixel_scale=pixel_scale,
                                   core_undersample=False, color='steelblue', lw=2,
                                   I_shift=I_norm-I_r0, markersize=0, alpha=0.2)

        if plot_brightest:
            num = list(res_thumb.keys())[0]
            img0, ma0, cen0 = res_thumb[num]['image'], res_thumb[num]['mask'], res_thumb[num]['center']
            cal_profile_1d(img0, cen=cen0, mask=ma0, dr=0.8,
                           ZP=ZP, sky_mean=BKG, sky_std=2.8,
                           xunit="arcsec", yunit="SB", errorbar=True,
                           pixel_scale=pixel_scale,
                           core_undersample=False, color='k', lw=3,
                           I_shift=I_norm-I_r0_all[0], markersize=8, alpha=0.9)

        if draw:
            ax.set_xlim(5.,4e2)
            ax.set_ylim(I_norm+6.5,I_norm-7.5)
            ax.axvspan(r1, r2, color='gold', alpha=0.1)
            ax.axvline(r0, ls='--',color='k', alpha=0.9, zorder=1)
            ax.set_xscale('log')
            
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_ins = inset_axes(ax, width="35%", height="35%",
                                bbox_to_anchor=(-0.02,-0.02, 1, 1),
                                bbox_transform=ax.transAxes)
            
    else:
        logger.warning("r0 is out of fit_range! n0 will be included in the fitting.")
        return None, None

    if norm=="intp":
        p_log_linear = partial(log_linear, x0=r0, y0=I_norm)
        popt, pcov, clip_func = iter_curve_fit(r_rbin_all, In_rbin_all, p_log_linear,
                                               x_min=r1, x_max=r2, p0=10,
                                               bounds=(3, 15), n_iter=3,
                                               x_lab='R [arcsec]', y_lab='MU [mag/arcsec2]',
                                               draw=draw, fig=fig, ax=ax_ins)
        r_n, I_n = r0, I_norm

    elif norm=="intg":
        p_log_linear = partial(log_linear)
        popt, pcov, clip_func = iter_curve_fit(r_rbin_all, In_rbin_all, p_log_linear,
                                               x_min=r1, x_max=r2, p0=(10, r0, I_norm),
                                               bounds=([3,r1,I_norm-1], [15,r2,I_norm+1]),
                                               x_lab='R [arcsec]', y_lab='MU [mag/arcsec2]',
                                               n_iter=3, draw=draw, fig=fig, ax=ax_ins)
        r_n, I_n = r0, p_log_linear(r0, *popt)

    if draw:
        ax.scatter(r_n, I_n, marker='*',color='r', s=300, zorder=4)
        
        ax_ins.set_ylim(I_norm+1.75, I_norm-2.25)
        ax_ins.axvline(r0, lw=2, ls='--', color='k', alpha=0.7, zorder=1)
        ax_ins.axvspan(r1, r2, color='gold', alpha=0.1)
        ax_ins.scatter(r_n, I_n, marker='*',color='r', s=200, zorder=4)
        ax_ins.tick_params(direction='in',labelsize=14)
        ax_ins.set_ylabel('')
        
        if save:
            fn_plot = f'Fit_n0_{range_str}_{obj_name}-{b}.png'
            plt.savefig(os.path.join(save_dir, fn_plot))
        plt.show()

    # I ~ klogr; m = -2.5logF => n = k/2.5
    n0, d_n0 = popt[0]/2.5, np.sqrt(pcov[0,0])/2.5
    logger.info("   - n0 = {:.4f}+/-{:.4f}".format(n0, d_n0))
    
    return n0, d_n0
    
    
## Add supplementary stars

def add_supplementary_atlas(tab, tab_atlas, SE_catalog,
                            sep=3*u.arcsec, mag_saturate=13):
    """ Add unmatched bright (saturated) stars using HLSP ATLAS catalog. """

    if len(tab['MAG_AUTO_corr']<mag_saturate)<1:
        return tab
        
    logger.info("Adding unmatched bright stars from HLSP ATLAS catalog.")
    
    # cross match SE catalog and ATLAS catalog
    coords_atlas = SkyCoord(tab_atlas['RA'], tab_atlas['Dec'], unit=u.deg)
    coords_SE = SkyCoord(SE_catalog['X_WORLD'],SE_catalog['Y_WORLD'])

    idx, d2d, _ = coords_SE.match_to_catalog_sky(coords_atlas)
    match = d2d < sep

    SE_catalog_match = SE_catalog[match]
    tab_atlas_match = tab_atlas[idx[match]]
    
    # add ATLAS mag to the table
    SE_catalog_match['gmag_atlas'] = tab_atlas_match['g']
    SE_catalog_match.sort('gmag_atlas')
    
    # add supplementary stars (bright stars failed to matched)
    cond_sup = (SE_catalog_match['MAG_AUTO']<mag_saturate) \
                & (SE_catalog_match['CLASS_STAR']>0.7)
    SE_catalog_sup = SE_catalog_match[cond_sup]
    num_SE_sup = np.setdiff1d(SE_catalog_sup['NUMBER'], tab['NUMBER'])
    
    # make a new table containing unmatched bright stars
    use_cols = SE_COLUMNS + ['gmag_atlas']
    tab_sup = Table(dtype=SE_catalog_sup[use_cols].dtype)
    for num in num_SE_sup:
        row = SE_catalog_sup[SE_catalog_sup['NUMBER']==num][0]
        tab_sup.add_row(row[use_cols])
        
    # add color term to MAG_AUTO
    CT = calculate_color_term(SE_catalog_match,
                              mag_range=[mag_saturate,18],
                              mag_name='gmag_atlas', draw=False)
    
    tab_sup['MAG_AUTO_corr'] = tab_sup['gmag_atlas'] + CT
    tab_sup.add_columns([tab_sup['X_IMAGE'], tab_sup['Y_IMAGE']],
                         names=['X_CATALOG', 'Y_CATALOG'])
                        
    # Join the two tables by common keys
    keys = set(tab.colnames).intersection(tab_sup.colnames)
    if len(tab_sup) > 0:
        tab_join = join(tab, tab_sup, keys=keys, join_type='outer')
        tab_join.sort('MAG_AUTO_corr')
        return tab_join
    else:
        return tab
    

def add_supplementary_SE_star(tab, SE_catatlog, mag_saturate=13, draw=True):
    """
    Add unmatched bright (saturated) stars in SE_catatlogas to tab.
    Magnitude is corrected by interpolation from other matched stars.
    
    """
    
    if len(tab['MAG_AUTO_corr']<mag_saturate)<5:
        return tab
        
    logger.info("Adding unmatched bright stars based on SE measurements...")
    
    # Empirical function to correct MAG_AUTO for saturation
    # Fit a sigma-clipped piecewise linear
    mag_lim = mag_saturate+2
    popt, _, clip_func = iter_curve_fit(tab['MAG_AUTO'], tab['MAG_AUTO_corr'],
                                        piecewise_linear, x_max=15, n_iter=5, k_std=10,
                                        p0=(1, 2, mag_saturate, mag_saturate),
                                        bounds=(0.9, [2, 4, mag_lim, mag_lim]),
                                        x_lab='MAG_AUTO', y_lab='MAG_AUTO_corr', draw=draw)

    # Empirical corrected magnitude
    f_corr = lambda x: piecewise_linear(x, *popt)
    mag_corr = f_corr(tab['MAG_AUTO'])
    
    # Remove rows with large magnitude offset
    loc_rm = np.where(abs(tab['MAG_AUTO_corr']-mag_corr)>2)
    if draw:
        plt.scatter(tab[loc_rm]['MAG_AUTO'], tab[loc_rm]['MAG_AUTO_corr'],
                    marker='s', s=40, facecolors='none', edgecolors='lime')
        plt.xlim(15, tab['MAG_AUTO'].min())
        plt.ylim(15, tab['MAG_AUTO_corr'].min())
        plt.show()
    tab.remove_rows(loc_rm[0])

    # Add supplementary stars (bright stars failed to matched)
    cond_sup = (SE_catatlog['MAG_AUTO']<mag_saturate) & (SE_catatlog['CLASS_STAR']>0.7)
    SE_cat_bright = SE_catatlog[cond_sup]
    num_SE_sup = np.setdiff1d(SE_cat_bright['NUMBER'], tab['NUMBER'])

    # make a new table containing all unmatched bright stars
    tab_sup = Table(dtype=SE_cat_bright.dtype)
    for num in num_SE_sup:
        row = SE_cat_bright[SE_cat_bright['NUMBER']==num][0]
        tab_sup.add_row(row)
    
    # add corrected MAG_AUTO
    tab_sup['MAG_AUTO_corr'] = f_corr(tab_sup['MAG_AUTO'])
    tab_sup.add_columns([tab_sup['X_IMAGE'], tab_sup['Y_IMAGE']],
                         names=['X_CATALOG', 'Y_CATALOG'])
                        
    # Join the two tables by common keys
    keys = set(tab.colnames).intersection(tab_sup.colnames)
    if len(tab_sup) > 0:
        tab_join = join(tab, tab_sup, keys=keys, join_type='outer')
        tab_join.sort('MAG_AUTO_corr')
        return tab_join
    else:
        return tab
    
    
def calculate_color_term(tab_target,
                         mag_range=[13,18], mag_name='gmag_PS',
                         verbose=True, draw=True):
    """
    Use non-saturated stars to calculate color term between SE MAG_AUTO
    and magnitude in the matched catalog .
    
    Parameters
    ----------
    tab_target : full matched source catlog
    
    mag_range : range of magnitude for stars to be used
    mag_name : column name of magnitude in tab_target 
    draw : whethert to draw a diagnostic plot of MAG_AUTO vs diff.
    
    Returns
    ----------
    CT : color correction term (SE - catlog)
    """
    
    mag = tab_target["MAG_AUTO"]
    mag_cat = tab_target[mag_name]
    d_mag = tab_target["MAG_AUTO"] - mag_cat
    
    use_range = (mag>mag_range[0])&(mag<mag_range[1])&(~np.isnan(mag_cat))
    d_mag = d_mag[use_range]
    mag = mag[use_range]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d_mag_clip = sigma_clip(d_mag, 3, maxiters=10)
        
    CT = biweight_location(d_mag_clip)
    
    if draw:
        plt.figure()
        plt.scatter(mag, d_mag, s=8, alpha=0.2, color='gray')
        plt.scatter(mag, d_mag_clip, s=6, alpha=0.3)
        plt.axhline(CT, color='k', alpha=0.7)
        plt.ylim(-3,3)
        plt.xlim(mag_range[0]-0.5, mag_range[1]+0.5)
        plt.xlabel("MAG_AUTO")
        plt.ylabel("MAG_AUTO - %s"%mag_name)
        plt.show()
        
    logger.info("Average Color Term [SE-%s] = %.5f"%(mag_name, CT))
        
    return np.around(CT,5)


def fit_empirical_aperture(tab_target, seg_map, mag_name='rmag_PS',
                           K=2, R_min=1, R_max=100,
                           mag_range=[11, 22], degree=2, draw=True):
    """
    Fit an empirical polynomial curve for log radius of aperture based on
    corrected magnitudes and segm map of SE. Radius is enlarged K times.
    
    Parameters
    ----------
    tab_target : full matched source catlog
    seg_map : training segm map
    
    mag_name : column name of magnitude in tab_target 
    mag_range : range of magnitude for stars to be used
    K : enlargement factor on the original segm map (default 2)
    R_min : minimum aperture size in pixel (default 1)
    R_max : maximum aperture size in pixel (default 100)
    degree : degree of polynomial (default 2)
    draw : whether to draw a diagnostic plot of log R vs mag
    
    Returns
    ----------
    estimate_radius : a function turns magnitude into log R
    
    """
    from photutils.segmentation import SegmentationImage
    
    msg = "Fitting {0}-order empirical relation for ".format(degree)
    msg += "apertures of catalog stars based on SExtarctor (X{0:.1f})".format(K)
    logger.info(msg)

    # Read from SE segm map
    segm_deb = SegmentationImage(seg_map)
    R_aper = (segm_deb.get_areas(tab_target["NUMBER"])/np.pi)**0.5
    tab_target['logR'] = np.log10(K * R_aper)
    
    mag_match = tab_target[mag_name]
    mag_match[np.isnan(mag_match)] = -1
    tab = tab_target[(mag_match>mag_range[0])&(mag_match<mag_range[1])]

    mag_all = tab[mag_name]
    logR = tab['logR']
    p_poly = np.polyfit(mag_all, logR, degree)
    f_poly = np.poly1d(p_poly)

    if draw:
        plt.scatter(tab_target[mag_name], tab_target['logR'], s=8, alpha=0.2, color='gray')
        plt.scatter(mag_all, logR, s=8, alpha=0.2, color='k')
        
    mag_ls = np.linspace(6,23)
    clip = np.zeros_like(mag_all, dtype='bool')
    
    for i in range(3):
        if draw: plt.plot(mag_ls, f_poly(mag_ls), lw=1, ls='--')
        mag, logr = mag_all[~clip], logR[~clip]

        p_poly = np.polyfit(mag, logr, degree)
        f_poly = np.poly1d(p_poly)

        dev = np.sqrt((logR-f_poly(mag_all))**2)
        clip = dev>3*np.mean(dev)
        
    if draw: 
        plt.plot(mag_ls, f_poly(mag_ls), lw=2, color='gold')

        plt.scatter(mag, logr, s=3, alpha=0.2, color='gold')

        plt.xlabel("magnitude (catalog)")
        plt.ylabel(r"$\log_{10}\,R$")
        plt.xlim(7,23)
        plt.ylim(0.15,2.2)
        plt.show()
    
    estimate_radius = lambda m: max(10**min(np.log10(R_max), f_poly(m)), R_min)
    
    return estimate_radius


def make_segm_from_catalog(catalog_star,
                           bounds, estimate_radius,
                           mag_name='rmag', mag_limit=22,
                           obj_name='', band='G',
                           ext_cat=None, draw=True,
                           save=False, dir_name='./Measure'):
    """
    Make segmentation map from star catalog. Aperture size used is based on SE semg map.
    
    Parameters
    ----------
    catalog_star : star catalog
    bounds : 1X4 1d array defining bounds of region
    estimate_radius : function of turning magnitude into log R
    
    mag_name : magnitude column name in catalog_star
    mag_limit : magnitude limit to add segmentation
    ext_cat : (bright) extended source catalog to mask
    draw : whether to draw the segm map
    save : whether to save the segm map as fits
    dir_name : path of saving
    
    Returns
    ----------
    seg_map : output segm map generated from catalog
    
    """
    
    Xmin, Ymin, Xmax, Ymax = bounds
    nX = Xmax - Xmin
    nY = Ymax - Ymin
    
    try:
        catalog = catalog_star[~catalog_star[mag_name].mask]
    except AttributeError:
        catalog = catalog_star[~np.isnan(catalog_star[mag_name])]
        
    catalog = catalog[catalog[mag_name]<mag_limit]
    msg = "Make segmentation map based on catalog {:s}: {:d} stars"
    msg = msg.format(mag_name, len(catalog))
    logger.info(msg)
    
    # Estimate mask radius
    R_est = np.array([estimate_radius(m) for m in catalog[mag_name]])
    
    # Generate object apertures
    apers = [CircularAperture((X_c-Xmin, Y_c-Ymin), r=r)
             for (X_c,Y_c, r) in zip(catalog['X_CATALOG'], catalog['Y_CATALOG'], R_est)]
    
    # Further mask for bright extended sources
    if ext_cat is not None:
        if len(ext_cat)>0:
            for (X_c,Y_c, a, b, theta) in zip(ext_cat['X_IMAGE'],
                                              ext_cat['Y_IMAGE'],
                                              ext_cat['A_IMAGE'],
                                              ext_cat['B_IMAGE'],
                                              ext_cat['THETA_IMAGE'],):
                pos = (X_c-Xmin, Y_c-Ymin)
                theta_ = np.mod(theta, 360) * np.pi/180
                aper = EllipticalAperture(pos, a*6, b*6, theta_)
                apers.append(aper)
            
    # Draw segment map generated from the catalog
    seg_map = np.zeros((nY, nX))
    
    # Segmentation k sorted by mag of source catalog
    for (k, aper) in enumerate(apers):
        star_ma = aper.to_mask(method='center').to_image((nY, nX))
        if star_ma is not None:
            seg_map[star_ma.astype(bool)] = k+2
    
    if draw:
        from .plotting import make_rand_cmap
        plt.figure(figsize=(6,6), dpi=100)
        plt.imshow(seg_map, vmin=1, cmap=make_rand_cmap(int(seg_map.max())))
        plt.show()
    
    # Save segmentation map built from catalog
    if save:
        check_save_path(dir_name, overwrite=True, verbose=False)
        hdu_seg = fits.PrimaryHDU(seg_map.astype(int))
        
        band = band.lower()
        range_str = f"X[{Xmin}-{Xmax}]Y[{Ymin}-{Ymax}]"
        fname = f"{obj_name}-segm_{band}_catalog_{range_str}.fits"
        filename = os.path.join(dir_name, fname)
        hdu_seg.writeto(filename, overwrite=True)
        logger.info(f"Saved segmentation map made from catalog as {filename}")
        
    return seg_map
    
def make_psf_from_fit(sampler, psf,
                      pixel_scale=DF_pixel_scale,
                      psf_range=None,
                      leg2d=False):
                      
    """
    
    Recostruct PSF from fit.
    
    Parameters
    ----------
    sampler : Sampler.sampler class
        The output sampler file (.res)
    psf : PSF_Model class
        Inherited PSF model.
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pixel
        
    Returns
    -------
    psf_fit : PSF_Model class
        Recostructed PSF.
    params : list
        Fitted parameters.
    """
    
    ct = sampler.container
    n_spline = ct.n_spline
    fit_sigma, fit_frac = ct.fit_sigma, ct.fit_frac
        
    params, _, _ = sampler.get_params_fit()
        
    K = 0
    if fit_frac: K += 1        
    if fit_sigma: K += 1
    
    if psf.aureole_model == "moffat":
        gamma1_fit, beta1_fit = params[:2]
        param_update = {'gamma1':gamma1_fit, 'beta1':beta1_fit}
        
    else:
        N_n = n_spline
        N_theta = n_spline - 1
        
        if psf.cutoff:
            try:
                n_c = psf.n_c
                theta_c = psf.theta_c
            except AttributeError:
                n_c = 4
                theta_c = 1200
                
        if psf.aureole_model == "power":
            n_fit = params[0]
            param_update = {'n':n_fit}

        elif psf.aureole_model == "multi-power":
            n_s_fit = params[:N_n]
            theta_s_fit = np.append(psf.theta_0, 10**params[N_n:N_n+N_theta])
            if psf.cutoff:
                n_s_fit = np.append(n_s_fit, n_c)
                theta_s_fit = np.append(theta_s_fit, theta_c)
                
            param_update = {'n0':n_s_fit[0], 'n_s':n_s_fit, 'theta_s':theta_s_fit}

    if fit_frac:
        frac = 10**params[-1]
        param_update['frac'] = frac
    
    # Make a new copy and update params
    psf_fit = psf.copy()
    psf_fit.update(param_update)

    mu_fit = params[-K-1]
    
    if fit_sigma:
        sigma_fit = 10**params[-K]
    else:
        sigma_fit = ct.std_est
        
    if ct.leg2d:
        psf_fit.A10, psf_fit.A01 = 10**params[-K-2], 10**params[-K-3]
        
    psf_fit.bkg, psf_fit.bkg_std  = mu_fit, sigma_fit
    
    _ = psf_fit.generate_core()
    _, _ = psf_fit.generate_aureole(psf_range=psf_range, psf_scale=pixel_scale)
    
    return psf_fit, params


def calculate_reduced_chi2(fit, data, uncertainty, dof=5):
    chi2_reduced = np.sum(((fit-data)/uncertainty)**2)/(len(data)-dof)
    logger.info("Reduced Chi^2 = %.5f"%chi2_reduced)


class MyError(Exception): 
    def __init__(self, message):  self.message = message 
    def __str__(self): return(repr(self.message))
    def __repr__(self): return 'MyError(%r)'%(str(self))

class InconvergenceError(MyError): 
    def __init__(self, message):  self.message = message 
    def __repr__(self):
        return 'InconvergenceError: %r'%self.message
