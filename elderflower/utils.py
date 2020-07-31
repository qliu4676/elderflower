import os
import re
import sys
import math
import time
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from astropy import wcs
from astropy import units as u
from astropy.io import fits, ascii
from astropy.table import Table, Column, join
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, median_absolute_deviation, gaussian_fwhm_to_sigma
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats

from photutils import detect_sources, deblend_sources
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture
from photutils.segmentation import SegmentationImage
    
from .plotting import LogNorm, AsinhNorm, colorbar

# default SE columns for cross_match
SE_COLUMNS = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
"MAG_AUTO", "FLUX_AUTO", "FWHM_IMAGE", "FLAGS"]

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

def Intensity2SB(I, BKG, ZP, pixel_scale=2.5):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
    I = np.atleast_1d(I)
    I[np.isnan(I)] = BKG
    if np.any(I<=BKG):
        I[I<=BKG] = np.nan
    I_SB = -2.5*np.log10(I - BKG) + ZP + 2.5 * math.log10(pixel_scale**2)
    return I_SB

def SB2Intensity(SB, BKG, ZP, pixel_scale=2.5):
    """ Convert surface brightness (mag/arcsec^2)to intensity given the background value, zero point and pixel scale """ 
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
        print("completed: %d/%d"%(i+1, number))


def round_good_fft(x):
    # Rounded PSF size to 2^k or 3*2^k
    a = 1 << int(x-1).bit_length()
    b = 3 << int(x-1).bit_length()-2
    if x>b:
        return a
    else:
        return min(a,b)

def calculate_psf_size(n0, theta_0, contrast=1e5, psf_scale=2.5,
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

def background_sub_SE(field, mask=None, b_size=64, f_size=3, n_iter=5):
    """ Subtract background using SE estimator with mask """ 
    from photutils import Background2D, SExtractorBackground, MedianBackground
    try:
        Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                           box_size=(b_size, b_size), filter_size=(f_size, f_size),
                           sigma_clip=SigmaClip(sigma=3., maxiters=n_iter))
        back = Bkg.background
        back_rms = Bkg.background_rms
    except ValueError:
        img = field.copy()
        if mask is not None:
            img[mask] = np.nan
        back, back_rms = np.nanmedian(field) * np.ones_like(field), np.nanstd(field) * np.ones_like(field)
    if mask is not None:
        back *= ~mask
        back_rms *= ~mask
    return back, back_rms

def display_background_sub(field, back):
    from .plotting import vmax_2sig, vmin_3mad
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    ax1.imshow(field, aspect="auto", cmap="gray", vmin=vmin_3mad(field), vmax=vmax_2sig(field),norm=LogNorm())
    im2 = ax2.imshow(back, aspect="auto", cmap='gray')
    colorbar(im2)
    ax3.imshow(field - back, aspect="auto", cmap='gray', vmin=0., vmax=vmax_2sig(field - back),norm=LogNorm())
    plt.tight_layout()

def source_detection(data, sn=2, b_size=120,
                     k_size=3, fwhm=3, smooth=True, 
                     sub_background=True, mask=None):
    from astropy.convolution import Gaussian2DKernel
    from photutils import detect_sources, deblend_sources
    
    if sub_background:
        back, back_rms = background_sub_SE(data, b_size=b_size)
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
    segm_sm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel, mask=mask)

    data_ma = data.copy() - back
    data_ma[segm_sm!=0] = np.nan

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
                n_iter=3, k_std=5, draw=True):
    """ Iterative curve_fit """
    
    if x_min is None: x_min = x_data.min()
    if x_max is None: x_max = x_data.max()
    x_test = np.linspace(x_min, x_max)
    clip = np.zeros_like(x_data, dtype='bool')

    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)

    # Iterative sigma clip
    for i in range(n_iter):
        if draw: plt.plot(x_test, func(x_test, *popt), color='gray', ls='--')
        
        x_clip, y_clip = x_data[~clip], y_data[~clip]
        popt, pcov = curve_fit(func, x_clip, y_clip, p0=p0)

        # compute residual and stddev
        res = y_data - func(x_data, *popt)
        std = mad_std(res)
        clip = res**2 > (k_std*std)**2
    
    # clip function
    clip_func = lambda x, y: (y - func(x, *popt))**2 > (k_std*std)**2
    
    if draw:
        s = plt.scatter(x_data, y_data, c=color,
                        s=5, cmap='viridis', alpha=0.4)
        plt.scatter(x_data[clip], y_data[clip], lw=2, s=20,
                    fc='none', ec='orange', alpha=0.7)
        plt.plot(x_test, func(x_test, *popt), color='r')
        
        if color is not None: plt.colorbar(s, label=c_lab)
        
        plt.xlim(x_max, x_min)
        plt.gca().invert_yaxis()
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        
    return popt, clip_func
    
def identify_extended_source(SE_catalog, mag_limit=15, mag_saturate=13, draw=True):
    """ Empirically pick out (bright) extended sources in the SE_catalog.
        The catalog need contain following columns:
        'MAG_AUTO', 'MU_MAX', 'ELLIPTICITY', 'CLASS_STAR' """
    
    bright = SE_catalog['MAG_AUTO'] < mag_limit
    SE_bright = SE_catalog[bright]
    x_data, y_data = SE_bright['MAG_AUTO'], SE_bright['MU_MAX']
    
    MU_saturate = np.quantile(y_data, 0.001) # guess of saturated MU_MAX
    MAG_saturate = mag_saturate # guess of saturated MAG_AUTO
    
    # Fit a flattened linear
    popt, clip_func = iter_curve_fit(x_data, y_data, flattened_linear,
                                    p0=(1, MAG_saturate, MU_saturate),
                                    x_max=mag_limit, c_lab='CLASS_STAR',
                                    color=SE_bright['CLASS_STAR'],
                                    x_lab='MAG_AUTO',y_lab='MU_MAX', draw=draw)

    # pick outliers in the catalog
    outlier = clip_func(SE_catalog['MAG_AUTO'], SE_catalog['MU_MAX'])
    
    # identify bright extended sources by:
    # (1) elliptical object or CLASS_STAR<0.5
    # (2) brighter than mag_limit
    # (3) lie out of MU_MAX vs MAG_AUTO relation
    is_extend = ((SE_catalog['ELLIPTICITY']>0.7)|(SE_catalog['CLASS_STAR']<0.5)) & bright & outlier
    
    return SE_catalog[is_extend]
    

def clean_isolated_stars(xx, yy, mask, star_pos, pad=0, dist_clean=60):
    
    star_pos = star_pos + pad
    
    clean = np.zeros(len(star_pos), dtype=bool)
    for k, pos in enumerate(star_pos):
        rr = np.sqrt((xx-pos[0])**2+(yy-pos[1])**2)
        if np.min(rr[~mask]) > dist_clean:
            clean[k] = True
            
    return clean
        
def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                   color="steelblue", xunit="pix", yunit="Intensity",
                   seeing=2.5, pixel_scale=2.5, ZP=27.1, sky_mean=884, sky_std=3, dr=1.5, 
                   lw=2, alpha=0.7, markersize=5, I_shift=0, figsize=None,
                   core_undersample=False, label=None, plot_line=False, mock=False,
                   plot=True, scatter=False, fill=False, errorbar=False, verbose=False):
    """Calculate 1d radial profile of a given star postage"""
    if mask is None:
        mask =  np.zeros_like(img, dtype=bool)
    if back is None:     
        back = np.ones_like(img) * sky_mean
    if cen is None:
        cen = (img.shape[1]-1)/2., (img.shape[0]-1)/2.
        
    yy, xx = np.indices((img.shape))
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int(3 * seeing/pixel_scale) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * np.median(back)
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.0001 * z_diff[-1]))
    r_max = np.sqrt(n_pix_max/np.pi)
    r_max = np.min([img.shape[0]//2, r_max])
    
    if verbose:
        print("Maximum R: %d (pix)"%np.int(r_max))    
    
    if xunit == "arcsec":
        r = r * pixel_scale   # radius in arcsec
        r_core = r_core * pixel_scale
        r_max = r_max * pixel_scale
        
    d_r = dr * pixel_scale if xunit == "arcsec" else dr
    
#     z = z[~np.isnan(z)]
    if mock:
        clip = lambda z: sigma_clip((z), sigma=5, maxiters=3)
    else:
        clip = lambda z: 10**sigma_clip(np.log10(z+1e-10), sigma=3, maxiters=5)
        
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:  
            # for undersampled core, bin in individual pixels 
            bins_inner = np.unique(r[r<r_core]) + 1e-3
        else: 
            bins_inner = np.linspace(0, r_core, int(min((r_core/d_r*2), 5))) - 1e-5

        n_bin_outer = np.max([7, np.min([np.int(r_max/d_r/10), 50])])
        if r_max > (r_core+d_r):
            bins_outer = np.logspace(np.log10(r_core+d_r), np.log10(r_max-d_r), n_bin_outer)
        else:
            bins_outer = []
        bins = np.concatenate([bins_inner, bins_outer])
        _, bins = np.histogram(r, bins=bins)
    
    # Calculate binned 1d profile
    r_rbin = np.array([])
    z_rbin = np.array([])
    zstd_rbin = np.array([])
    for k, b in enumerate(bins[:-1]):
        in_bin = (r>bins[k])&(r<bins[k+1])
        
        z_clip = clip(z[in_bin])
        if len(z_clip)==0:
            continue

        zb = np.mean(z_clip)
        zstd_b = np.std(z_clip)
        
        z_rbin = np.append(z_rbin, zb)
        zstd_rbin = np.append(zstd_rbin, zstd_b)
        r_rbin = np.append(r_rbin, np.mean(r[in_bin]))
        
        
    logzerr_rbin = 0.434 * abs( zstd_rbin / (z_rbin-sky_mean))
    
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
            I_rbin = Intensity2SB(I=z_rbin, BKG=np.median(back),
                                  ZP=ZP, pixel_scale=pixel_scale) + I_shift
            I_sky = -2.5*np.log10(sky_std) + ZP + 2.5 * math.log10(pixel_scale**2)

            plt.plot(r_rbin, I_rbin, "-o", mec="k",
                     lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label)   
            if scatter:
                I = Intensity2SB(I=z, BKG=np.median(back),
                                 ZP=ZP, pixel_scale=pixel_scale) + I_shift
                
            if errorbar:
                Ierr_rbin_up = I_rbin - Intensity2SB(I=z_rbin,
                                                     BKG=np.median(back)-sky_std,
                                                     ZP=ZP, pixel_scale=pixel_scale)
                Ierr_rbin_lo = Intensity2SB(I=z_rbin-sky_std,
                                            BKG=np.median(back)+sky_std,
                                            ZP=ZP, pixel_scale=pixel_scale) - I_rbin
                lolims = np.isnan(Ierr_rbin_lo)
                uplims = np.isnan(Ierr_rbin_up)
                Ierr_rbin_lo[lolims] = 4
                Ierr_rbin_up[uplims] = 4
                plt.errorbar(r_rbin, I_rbin, yerr=[Ierr_rbin_up, Ierr_rbin_lo],
                             fmt='', ecolor=color, capsize=2, alpha=0.5)
                
            plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
            plt.gca().invert_yaxis()
            plt.ylim(30,17)

        plt.xscale("log")
        plt.xlim(max(r_rbin[np.isfinite(r_rbin)][0]*0.8, 1e-1),r_rbin[np.isfinite(r_rbin)][-1]*1.2)
        plt.xlabel("R [arcsec]") if xunit == "arcsec" else plt.xlabel("r [pix]")
        
        if scatter:
            plt.scatter(r[r<3*r_core], I[r<3*r_core], color=color, 
                        s=markersize/2, alpha=alpha/2, zorder=1)
            plt.scatter(r[r>=3*r_core], I[r>=3*r_core], color=color,
                        s=markersize/5, alpha=alpha/10, zorder=1)
            

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
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

def calculate_fit_SB(psf, r=np.logspace(0.03,2.5,100), mags=[15,12,9], ZP=27.1):
    
    frac = psf.frac
        
    I_s = [10**((mag-ZP)/-2.5) for mag in mags]
    
    comp1 = psf.f_core1D(r)
    comp2 = psf.f_aureole1D(r)

    I_tot_s = [Intensity2SB(((1-frac) * comp1 + comp2 * frac) * I,
                            0, ZP, psf.pixel_scale) for I in I_s]
    return I_tot_s

### Funcs for measuring scaling ###

def get_star_pos(id, star_cat):
    """ Get the position of an object from the catalog"""
    
    X_c, Y_c = star_cat[id]["X_IMAGE"], star_cat[id]["Y_IMAGE"]
    return (X_c, Y_c)

def get_star_thumb(id, star_cat, wcs, data, seg_map, 
                   n_win=20, seeing=2.5, origin=1, verbose=True):
    """ Crop the data and segment map into thumbnails.
        Return thumbnail of image/segment/mask, and center of the star. """
    
    (X_c, Y_c) = get_star_pos(id, star_cat)    
    
    # define thumbnail size
    fwhm =  max(star_cat[id]["FWHM_IMAGE"], seeing)
    win_size = int(n_win * min(max(fwhm,2), 8))
    
    # calculate boundary
    X_min, X_max = max(1, X_c - win_size), min(data.shape[1], X_c + win_size)
    Y_min, Y_max = max(1, Y_c - win_size), min(data.shape[0], Y_c + win_size)
    x_min, y_min = coord_Im2Array(X_min, Y_min, origin)
    x_max, y_max = coord_Im2Array(X_max, Y_max, origin)
    
    if verbose:
        num = star_cat[id]["NUMBER"]
        print("NUMBER: ", num)
        print("X_c, Y_c: ", (X_c, Y_c))
        print("RA, DEC: ", (star_cat[id]["X_WORLD"], star_cat[id]["Y_WORLD"]))
        print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
        print("X_min, X_max, Y_min, Y_max: ", X_min, X_max, Y_min, Y_max)
    
    # crop image and segment map
    img_thumb = data[x_min:x_max, y_min:y_max].copy()
    if seg_map is None:
        seg_thumb = None
        mask_thumb = np.zeros_like(data, dtype=bool)
    else:
        seg_thumb = seg_map[x_min:x_max, y_min:y_max]
        mask_thumb = (seg_thumb!=0) # mask sources by 1
    
    # the center position is converted from world with wcs
    X_cen, Y_cen = wcs.wcs_world2pix(star_cat[id]["X_WORLD"], star_cat[id]["Y_WORLD"], origin)
    cen_star = X_cen - X_min, Y_cen - Y_min
    
    return (img_thumb, seg_thumb, mask_thumb), cen_star
    
def extract_star(id, star_cat, wcs, data, seg_map=None, 
                 seeing=2.5, sn_thre=2.5, n_win=25, n_dilation=1,
                 display_bg=False, display=True, verbose=False):
    
    """ Return the image thubnail, mask map, backgroud estimates, and center of star.
        Do a finer detection&deblending to remove faint undetected source."""
    
    from skimage import morphology
    thumb_list, cen_star = get_star_thumb(id, star_cat, wcs, data, seg_map,
                                          n_win=n_win, seeing=seeing, verbose=verbose)
    img_thumb, seg_thumb, mask_thumb = thumb_list
    
    # measure background, use a scalar value if the thumbnail is small 
    b_size = round(img_thumb.shape[0]//5/25)*25
    if img_thumb.shape[0] >= 50:
        back, back_rms = background_sub_SE(img_thumb, b_size=b_size)
    else:
        back, back_rms = (np.median(img_thumb[~mask_thumb])*np.ones_like(img_thumb), 
                            mad_std(img_thumb[~mask_thumb])*np.ones_like(img_thumb))
    if display_bg:
        # show background subtraction
        display_background_sub(img_thumb, back)  
            
    if seg_thumb is None:
        # the same thumbnail size
        fwhm = max([star_cat[id]["FWHM_IMAGE"], seeing])
        
        # do segmentation (a second time) to remove faint undetected stars using photutils
        sigma = seeing * gaussian_fwhm_to_sigma
        threshold = back + (sn_thre * back_rms)
        segm = detect_sources(img_thumb, threshold, npixels=5)

        # do deblending using photutils
        segm_deblend = deblend_sources(img_thumb, segm, npixels=5,
                                       nlevels=64, contrast=0.005)
    else:
        segm_deblend = SegmentationImage(seg_thumb)
        
    # the target star is at the center of the thumbnail
    star_lab = segm_deblend.data[int(cen_star[1]), int(cen_star[0])]
    star_ma = ~((segm_deblend.data==star_lab) | (segm_deblend.data==0)) # mask other source
        
    # dilation
    for i in range(n_dilation):
        star_ma = morphology.dilation(star_ma)
    
    if display:
        med_back = np.median(back)
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
        ax1.imshow(img_thumb, vmin=med_back-1, vmax=10000, norm=LogNorm(), cmap="viridis")
        ax1.set_title("star", fontsize=16)

        ax2.imshow(segm_deblend, cmap=segm_deblend.make_cmap(random_state=12345))
        ax2.set_title("segment", fontsize=16)

        img_thumb_ma = img_thumb.copy()
        img_thumb_ma[star_ma] = -1
        ax3.imshow(img_thumb_ma, cmap="viridis", norm=LogNorm(),
                   vmin=med_back-1, vmax=med_back+10*np.median(back_rms))
        ax3.set_title("extracted star", fontsize=16)
        plt.tight_layout()
    
    return img_thumb, star_ma, back, cen_star


def compute_Rnorm(image, mask_field, cen, R=12, wid=1, mask_cross=True, display=False):
    """ Compute (3 sigma-clipped) normalization using an annulus.
    Note the output values of normalization contain background.
    
    Paramters
    ----------
    image : input image for measurement
    mask_field : mask map with nearby sources masked as 1.
    cen : center of target
    R : radius of annulus
    wid : half-width of annulus
        
    Returns
    -------
    I_mean: mean value in the annulus
    I_med : median value in the annulus
    I_std : std value in the annulus
    I_flag : 0 good / 1 bad (available pixles < 5)
    
    """
    
    annulus_ma = CircularAnnulus([cen], R-wid, R+wid).to_mask()[0]      
    mask_ring = annulus_ma.to_image(image.shape) > 0.5    # sky ring (R-wid, R+wid)
    mask_clean = mask_ring & (~mask_field)                # sky ring with other sources masked
    
    # Whether to mask the cross regions, important if R is small
    if mask_cross:
        yy, xx = np.indices(image.shape)
        rr = np.sqrt((xx-cen[0])**2+(yy-cen[1])**2)
        cross = ((abs(xx-cen[0])<4)|(abs(yy-cen[1])<4))
        mask_clean = mask_clean * (~cross)
    
    if len(image[mask_clean]) < 5:
        return [np.nan] * 3 + [1]
    
    z = sigma_clip(np.log10(image[mask_clean]), sigma=2, maxiters=5)
    I_mean, I_med, I_std = 10**np.mean(z), 10**np.median(z.compressed()), np.std(10**z)

    if display:
        z = 10**z
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        ax1.imshow(mask_clean, cmap="gray", alpha=0.7)
        ax1.imshow(image, vmin=image.min(), vmax=I_med+50*I_std,
                   cmap='viridis', norm=AsinhNorm(), alpha=0.7)
        ax1.plot(cen[0], cen[1], 'r*', ms=10)
        
        ax2.hist(sigma_clip(z),alpha=0.7)
        
        # Label mean value
        plt.axvline(I_mean, color='k')
        plt.text(0.5, 0.9, "%.1f"%I_mean, color='darkorange', ha='center', transform=ax2.transAxes)
        
        # Label 20% / 80% quantiles
        I_20 = np.quantile(z.compressed(), 0.2)
        I_80 = np.quantile(z.compressed(), 0.8)
        for I, x_txt in zip([I_20, I_80], [0.2, 0.8]):
            plt.axvline(I, color='k', ls="--")
            plt.text(x_txt, 0.9, "%.1f"%I, color='orange',
                     ha='center', transform=ax2.transAxes)
        
    return I_mean, I_med, I_std, 0


def compute_Rnorm_batch(table_target, data, seg_map, wcs,
                        R=12, wid=1, return_full=False,
                        display=False, verbose=True):
    """ Combining the above functions. Compute for all object in table_target.
        Return an arry with measurement on the intensity and a dictionary containing maps and centers."""
    
    # Initialize
    res_thumb = {}    
    res_Rnorm = np.empty((len(table_target), 5))
    
    for i, (num, mag_auto) in enumerate(zip(table_target['NUMBER'], table_target['MAG_AUTO'])):
        if verbose: counter(i, len(table_target))
        ind = np.where(table_target['NUMBER']==num)[0][0]
        
        # For very bright sources, use a broader window
        n_win = 30 if mag_auto < 12 else 25
        img, ma, bkg, cen = extract_star(ind, table_target, wcs, data, seg_map,
                                         n_win=n_win, display_bg=False, display=False)
        
        res_thumb[num] = {"image":img, "mask":ma, "bkg":bkg, "center":cen}
        
        # Measure the mean, med and std of intensity at R
        I_mean, I_med, I_std, Iflag = compute_Rnorm(img, ma, cen, R=R, wid=wid, display=display)
        
        if (Iflag==1) & verbose: print ("Errorenous measurement: #", num)
        
        # Use the median value of background as the local background
        sky_mean = np.median(bkg)
        
        res_Rnorm[i] = np.array([I_mean, I_med, I_std, sky_mean, Iflag])
    
    return res_Rnorm, res_thumb

def measure_Rnorm_all(table, bounds,
                      wcs_data, image, seg_map=None, 
                      r_scale=12, width=1, mag_limit=15,
                      mag_name='rmag_PS', read=False,
                      obj_name="", save=True, dir_name='.',
                      display=False, verbose=True):
    """
    Measure normalization at r_scale for bright stars in table.
    If seg_map is not given, source detection will be run.

    Parameters
    ----------
    table : table containing list of sources
    bounds : 1X4 1d array defining the bound of region
    wcs_data : wcs
    image : image data
    
    seg_map : segm map used to mask nearby sources during the measurement. If not given a source detection will be done.
    r_scale : radius at which the flux scaling is measured (default: 10)
    width : half-width of ring used to measure the flux scaling at r_scale (default: 0.5 pix)
    mag_name : magnitude column name
    mag_limit : magnitude limit below which are measured
    read : whether to read existed outputs
    save : whether to save output table and thumbnails
    obj_name : object name used as prefix of saved output
    dir_name : path of saving 
    
    Returns
    ----------
    table_res_Rnorm : table containing measurement results
    res_thumb : thumbnails of image, mask, background and center of object, stored as dictionary
        
    """

    Xmin, Ymin = bounds[:2]
    
    table_Rnorm_name = os.path.join(dir_name, '%s-norm_%dpix_%smag%d_X%sY%s.txt'\
                                    %(obj_name, r_scale, mag_name[0], mag_limit, Xmin, Ymin))
    res_thumb_name = os.path.join(dir_name, '%s-thumbnail_%smag%d_X%sY%s'\
                                  %(obj_name, mag_name[0], mag_limit, Xmin, Ymin))
    if read:
        table_res_Rnorm = Table.read(table_Rnorm_name, format="ascii")
        res_thumb = load_thumbs(res_thumb_name)
        
    else:
        tab = table[table[mag_name]<mag_limit]
        res_Rnorm, res_thumb = compute_Rnorm_batch(tab, image, seg_map, wcs_data,
                                                   R=r_scale, wid=width,
                                                   return_full=True, display=display, verbose=verbose)
        
        
        keep_columns = ['NUMBER', 'MAG_AUTO', 'MAG_AUTO_corr', mag_name] \
                            + [s for s in tab.colnames if 'IMAGE' in s]
        table_res_Rnorm = tab[keep_columns].copy()
    
        for j, colname in enumerate(['Imean','Imed','Istd','Isky', 'Iflag']):
            if colname=='Iflag':
                col = res_Rnorm[:,j].astype(int)
            else:
                col = np.around(res_Rnorm[:,j], 5)
            table_res_Rnorm[colname] = col
        
        if save:
            check_save_path(dir_name, make_new=False, verbose=False)
            save_thumbs(res_thumb, res_thumb_name)
            table_res_Rnorm.write(table_Rnorm_name, overwrite=True, format='ascii')
            
    return table_res_Rnorm, res_thumb

### Catalog / Data Manipulation Helper ###
def id_generator(size=6, chars=None):
    import random
    import string

    if chars is None:
        chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def check_save_path(dir_name, make_new=True, verbose=True):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    elif make_new:
        if len(os.listdir(dir_name)) != 0:
            while os.path.exists(dir_name):
                dir_name = input("'%s' already existed. Enter a directory name for saving:"%dir_name)
            os.makedirs(dir_name)
    if verbose: print("Results will be saved in %s\n"%dir_name)
    

def find_keyword_header(header, keyword):
    """ Search keyword value in header (converting to float).
        Input a value by user if not found. """
    try:
        val = np.float(header[keyword])
     
    except KeyError:
        print("%s missing in header --->"%keyword)
        
        try:
            val = np.float(input("Input a value of %s :"%keyword))
        except ValueError:
            sys.exit("Invalid %s values!"%keyword) 
            
    return val
    
    
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

def crop_image(data, bounds, SE_seg_map=None, weight_map=None,
               sub_bounds=None, origin=1, color="w", draw=False):
    """ Crop the data (and segm map if given) with the given bouds. """
    from matplotlib import patches  
    Xmin, Ymin, Xmax, Ymax = bounds
    xmin, ymin = coord_Im2Array(Xmin, Ymin, origin)
    xmax, ymax = coord_Im2Array(Xmax, Ymax, origin)

    patch = np.copy(data[xmin:xmax, ymin:ymax])
    
    if SE_seg_map is None:
        seg_patch = None
    else:
        seg_patch = np.copy(SE_seg_map[xmin:xmax, ymin:ymax])
    
    if draw:
        if SE_seg_map is not None:
            sky = data[(SE_seg_map==0)]
        else:
            sky = sigma_clip(data, 3)
        sky_mean = np.mean(sky)
        sky_std = max(mad_std(sky[sky>sky_mean]),5)

        fig, ax = plt.subplots(figsize=(12,8))       
        plt.imshow(data, norm=AsinhNorm(a=0.1), cmap="gray_r",
                   vmin=sky_mean, vmax=sky_mean+10*sky_std, alpha=0.95)
        if weight_map is not None:
            plt.imshow(data*weight_map, norm=AsinhNorm(a=0.1), cmap="gray",
                       vmin=sky_mean, vmax=sky_mean+10*sky_std, alpha=0.3)
        
        width = Xmax-Xmin, Ymax-Ymin
        rect = patches.Rectangle((Xmin, Ymin), width[0], width[1],
                                 linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        plt.plot([Xmin+width[0]//2-360,Xmin+width[0]//2+360], [450,450],"whitesmoke",lw=3)
        plt.plot([Xmin+width[0]//2+360,Xmin+width[0]//2+360], [420,480],"whitesmoke",lw=3)
        plt.plot([Xmin+width[0]//2-360,Xmin+width[0]//2-360], [420,480],"whitesmoke",lw=3)
        plt.text(Xmin+width[0]//2, 220, r"$\bf 0.5\,deg$", color='whitesmoke', ha='center', fontsize=18)
        
        if sub_bounds is not None:
            for bounds in sub_bounds:
                Xmin, Ymin, Xmax, Ymax = bounds
                width = Xmax-Xmin, Ymax-Ymin
                rect = patches.Rectangle((Xmin, Ymin), width[0], width[1],
                                         linewidth=2.5, edgecolor='indianred', facecolor='none')
                ax.add_patch(rect)
                
        
        plt.show()
        
    if SE_seg_map is None:    
        return patch
    else:
        return patch, seg_patch

def query_vizier(catalog_name, radius, columns, column_filters, header=None, coord=None):
    """ Query catalog in Vizier database with the given catalog name,
    search radius and column names. If coords is not given, look for fits header """
    from astroquery.vizier import Vizier
    from astropy import units as u
    
    # Prepare for quearyinig Vizier with filters up to infinitely many rows. By default, this is 50.
    viz_filt = Vizier(columns=columns, column_filters=column_filters)
    viz_filt.ROW_LIMIT = -1
    
    if coord==None:
        RA, DEC = re.split(",", header['RADEC'])
        coord = SkyCoord(RA+" "+DEC , unit=(u.hourangle, u.deg))

    # Query!
    result = viz_filt.query_region(coord, radius=radius, 
                                   catalog=[catalog_name])
    return result

def transform_coords2pixel(table, wcs, name='', RA_key="RAJ2000", DE_key="DEJ2000", origin=1):
    """ Transform the RA/DEC columns in the table into pixel coordinates given wcs"""
    coords = np.vstack([np.array(table[RA_key]), 
                        np.array(table[DE_key])]).T
    pos = wcs.wcs_world2pix(coords, origin)
    table.add_column(Column(np.around(pos[:,0], 4)*u.pix), name='X_IMAGE'+'_'+name)
    table.add_column(Column(np.around(pos[:,1], 4)*u.pix), name='Y_IMAGE'+'_'+name)
    table.add_column(Column(np.arange(len(table))+1, dtype=int), 
                     index=0, name="ID"+'_'+name)
    return table

def merge_catalog(SE_catalog, table_merge, sep=5 * u.arcsec,
                  RA_key="RAJ2000", DE_key="DEJ2000", keep_columns=None):
    
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

def read_measurement_tables(dir_name, bounds0_list,
                            obj_name='', band='G',
                            pad=100, r_scale=12,
                            mag_limit=15):
    """ Read measurement tables from the directory """
    
    # Magnitude name
    b_name = band.lower()
    mag_name = b_name+'MeanPSFMag' if 'PS' in dir_name else b_name+'mag'
    
    tables_res_Rnorm = []
    tables_faint = []
    
    for bounds0 in np.atleast_2d(bounds0_list):
        # Clipped bounds
        patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = bounds0

        bounds = (patch_Xmin0+pad, patch_Ymin0+pad,
                        patch_Xmax0-pad, patch_Ymax0-pad)

        ## Read measurement for faint stars from catalog
        # Faint star catalog name
        fname_catalog = os.path.join(dir_name, "%s-catalog_PS_%s_all.txt"%(obj_name, b_name))

        # Check if the file exist
        if os.path.isfile(fname_catalog):
            table_catalog = Table.read(fname_catalog, format="ascii")
            mag_catalog = table_catalog[mag_name]
        else:
            sys.exit("Table %s does not exist. Exit."%fname_catalog)

        # stars fainter than magnitude limit (fixed as background), > 22 is ignored
        table_faint = table_catalog[(mag_catalog>=mag_limit) & (mag_catalog<22)]
        table_faint = crop_catalog(table_faint,
                                   keys=("X_IMAGE_PS", "Y_IMAGE_PS"),
                                   bounds=bounds)
        tables_faint += [table_faint]  
        
        ## Read measurement for bright stars
        # Catalog name
        fname_res_Rnorm = os.path.join(dir_name, "%s-norm_%dpix_%smag%s_X%dY%d.txt"\
                                       %(obj_name, r_scale, b_name,
                                         mag_limit, patch_Xmin0, patch_Ymin0))
        # Check if the file exist
        if os.path.isfile(fname_res_Rnorm):
            table_res_Rnorm = Table.read(fname_res_Rnorm, format="ascii")
        else:
            sys.exit("Table %s does not exist. Exit."%fname_res_Rnorm)

        # Crop the catalog
        table_res_Rnorm = crop_catalog(table_res_Rnorm, bounds=bounds0)

        # Do not use flagged measurement
        Iflag = table_res_Rnorm["Iflag"]
        tables_res_Rnorm += [table_res_Rnorm[Iflag==0]]
        
    return tables_faint, tables_res_Rnorm
    

def assign_star_props(table_faint, table_res_Rnorm, Image, 
                      r_scale=12, mag_threshold=[14,11],
                      psf=None, keys='Imed', verbose=True, 
                      draw=True, save=False, save_dir='./'):
    """ Assign position and flux for faint and bright stars from tables. """
    
    from .modeling import Stars
    
    # Image attributes
    ZP = Image.zp_val
    sky_mean = Image.bkg_val
    image_size = Image.image_size
    
    pos_ref = (Image.bounds[0], Image.bounds[1])
    
    table_faint['FLUX_AUTO_corr'] = 10**((table_faint["MAG_AUTO_corr"]-ZP)/(-2.5))
    
    try:
        ma = table_faint['FLUX_AUTO_corr'].data.mask
    except AttributeError:
        ma = np.isnan(table_faint['FLUX_AUTO_corr'])

    # Positions & Flux of faint stars from measured norm
    star_pos1 = np.vstack([table_faint['X_IMAGE_PS'].data[~ma],
                           table_faint['Y_IMAGE_PS'].data[~ma]]).T - pos_ref
    Flux1 = np.array(table_faint['FLUX_AUTO_corr'].data[~ma])

    # Positions & Flux (estimate) of bright stars from catalog
    star_pos2 = np.vstack([table_res_Rnorm['X_IMAGE_PS'],
                           table_res_Rnorm['Y_IMAGE_PS']]).T - pos_ref
    Flux2 = 10**((np.array(table_res_Rnorm["MAG_AUTO_corr"])-ZP)/(-2.5))

    # Estimate of brightness I at r_scale (I = Intensity - BKG) and flux
    z_norm = table_res_Rnorm['Imed'].data - sky_mean
    z_norm[z_norm<=0] = z_norm[z_norm>0].min()

    # Convert/printout thresholds
    Flux_threshold = 10**((np.array(mag_threshold) - ZP) / (-2.5))
    if verbose:
        print('Magnitude Thresholds:  {0}, {1} mag'.format(*mag_threshold))
        print("(<=> Flux Thresholds: {0}, {1} ADU)".format(*np.around(Flux_threshold,2)))
        try:
            SB_threshold = psf.Flux2SB(Flux_threshold, BKG=sky_mean, ZP=ZP, r=r_scale)
            print("(<=> Surface Brightness Thresholds: {0}, {1} mag/arcsec^2 at {2} pix)\n"\
                  .format(*np.around(SB_threshold,1),r_scale))
        except:
            pass

    # Combine two samples, make sure they do not overlap
    star_pos = np.vstack([star_pos1, star_pos2])
    Flux = np.concatenate([Flux1, Flux2])
    stars_all = Stars(star_pos, Flux, Flux_threshold=Flux_threshold)

    # Bright stars in model
    stars_0 = Stars(star_pos2, Flux2, Flux_threshold=Flux_threshold,
                    z_norm=z_norm, r_scale=r_scale, BKG=sky_mean, verbose=verbose)
    stars_0 = stars_0.remove_outsider(image_size, d=[36, 12])
    
    if draw:
        stars_all.plot_flux_dist(label='All', color='plum')
        stars_0.plot_flux_dist(label='Model', color='orange', ZP=ZP,
                              save=save, save_dir=save_dir)
        plt.show()
        
    return stars_0, stars_all

def cross_match(wcs_data, SE_catalog, bounds, radius=None, 
                pixel_scale=2.5, mag_limit=15, sep=4*u.arcsec,
                clean_catalog=True, mag_name='rmag',
                catalog={'Pan-STARRS': 'II/349/ps1'},
                columns={'Pan-STARRS': ['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
                                        'objID', 'Qual', 'gmag', 'e_gmag', 'rmag', 'e_rmag']},
                column_filters={'Pan-STARRS': {'rmag':'{0} .. {1}'.format(5, 23)}},
                magnitude_name={'Pan-STARRS':['rmag','gmag']},
                verbose=True):
    """ 
        Cross match SExtractor catalog with Vizier Online catalog.
        
        'URAT': 'I/329/urat1'
                magnitude_name: "rmag"
                columns: ['RAJ2000', 'DEJ2000', 'mfa', 'gmag', 'e_gmag', 'rmag', 'e_rmag']
                column_filters: {'mfa':'=1', 'rmag':'{0} .. {1}'.format(8, 18)}
                
        'USNO': 'I/252/out'
                magnitude_name: "Rmag"
                columns: ['RAJ2000', 'DEJ2000', 'Bmag', 'Rmag']
                column_filters: {"Rmag":'{0} .. {1}'.format(5, 15)}
                
    """
    from astropy.table import join, vstack
    
    cen = (bounds[2]+bounds[0])/2., (bounds[3]+bounds[1])/2.
    coord_cen = wcs_data.pixel_to_world(cen[0], cen[1])
    
    if radius is None:
        L = math.sqrt((cen[0]-bounds[0])**2 + (cen[1]-bounds[1])**2)
        radius = L * pixel_scale * u.arcsec
        
    print("Search", np.around(radius.to(u.deg), 3), "around:")
    print(coord_cen)
    
    for j, (cat_name, table_name) in enumerate(catalog.items()):
        # Query from Vizier
        result = query_vizier(catalog_name=cat_name,
                              radius=radius,
                              columns=columns[cat_name],
                              column_filters=column_filters[cat_name],
                              coord=coord_cen)

        Cat_full = result[table_name]
        
        if len(cat_name) > 4:
            c_name = cat_name[0] + cat_name[-1]
        else:
            c_name = cat_name
        
        m_name = np.atleast_1d(mag_name)[j]
        
        # Transform catalog wcs coordinate into pixel postion
        Cat_full = transform_coords2pixel(Cat_full, wcs_data, name=c_name)
        
        # Crop catalog and sort by the catalog magnitude
        Cat_crop = crop_catalog(Cat_full, bounds, sortby=m_name,
                                keys=("X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name))
        
        # catalog magnitude
        mag_cat = Cat_crop[m_name]
        
        # Screen out bright stars (mainly for cleaning duplicate source in catalog)
        Cat_bright = Cat_crop[(np.less(mag_cat, mag_limit,
                                       where=~np.isnan(mag_cat))) & ~np.isnan(mag_cat)]
        mag_cat.mask[np.isnan(mag_cat)] = True
        
        if clean_catalog:
            # Clean duplicate items in the catalog
            c_bright = SkyCoord(Cat_bright['RAJ2000'], Cat_bright['DEJ2000'], unit=u.deg)
            c_catalog = SkyCoord(Cat_crop['RAJ2000'], Cat_crop['DEJ2000'], unit=u.deg)
            idxc, idxcatalog, d2d, d3d = c_catalog.search_around_sky(c_bright, sep)
            inds_c, counts = np.unique(idxc, return_counts=True)
            
            row_duplicate = np.array([], dtype=int)
            
            # Use the measurement with min error in RA/DEC
            for i in inds_c[counts>1]:
                obj_duplicate = Cat_crop[idxcatalog][idxc==i]
#                 obj_duplicate.pprint(max_lines=-1, max_width=-1)
                
                # Remove detection without magnitude measurement
                mag_obj = obj_duplicate[m_name]
                obj_duplicate = obj_duplicate[~np.isnan(mag_obj)]
                
                # Use the detection with the best astrometry
                e2_coord = obj_duplicate["e_RAJ2000"]**2 + obj_duplicate["e_DEJ2000"]**2
                min_e2_coord = np.nanmin(e2_coord)
                
                for ID in obj_duplicate[e2_coord>min_e2_coord]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
            
            Cat_crop.remove_rows(np.unique(row_duplicate))
            #Cat_bright = Cat_crop[mag_cat<mag_limit]
            
        for m_name in magnitude_name[cat_name]:
            mag = Cat_crop[m_name]
            if verbose:
                print("%s %s:  %.3f ~ %.3f"%(cat_name, m_name, mag.min(), mag.max()))

        # Merge Catalog
        keep_columns = SE_COLUMNS + ["ID"+'_'+c_name] + magnitude_name[cat_name] + \
                                    ["X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name]
        tab_match = merge_catalog(SE_catalog, Cat_crop, sep=sep,
                                  keep_columns=keep_columns)
        tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                         keep_columns=keep_columns)
        
        # Rename columns
        for m_name in magnitude_name[cat_name]:
            tab_match[m_name].name = m_name+'_'+c_name
            tab_match_bright[m_name].name = m_name+'_'+c_name
        
        # Join tables
        if j==0:
            tab_target_all = tab_match
            tab_target = tab_match_bright
        else:
            tab_target_all = join(tab_target_all, tab_match, keys=SE_COLUMNS,
                                 join_type='left', metadata_conflicts='silent')
            tab_target = join(tab_target, tab_match_bright, keys=SE_COLUMNS,
                              join_type='left', metadata_conflicts='silent')
            
    # Sort matched catalog by SE MAG_AUTO
    tab_target.sort('MAG_AUTO')
    tab_target_all.sort('MAG_AUTO')

    mag_all = tab_target_all[mag_name+'_'+c_name]
    mag = tab_target[mag_name+'_'+c_name]
    if verbose:
        print("Matched stars with %s %s:  %.3f ~ %.3f"\
              %(cat_name, mag_name, mag_all.min(), mag_all.max()))
        print("Matched bright stars with %s %s:  %.3f ~ %.3f"\
              %(cat_name, mag_name, mag.min(), mag.max()))
    
    return tab_target, tab_target_all, Cat_crop

def cross_match_PS1_DR2(wcs_data, SE_catalog, bounds,
                        band='g', radius=None, clean_catalog=True,
                        pixel_scale=2.5, mag_limit=15, sep=2.5*u.arcsec,
                        verbose=True):
    """
    Use PANSTARRS DR2 API to do cross-match with the SE source catalog. 
    Note this could be (much) slower compared to cross-match using Vizier.
    
    Parameters
    ----------
    wcs_data : wcs of data
    
    SE_catalog : SE source catalog
    
    bounds : Nx4 2d / 1d array defining the cross-match region(s) [Xmin, Ymin, Xmax, Ymax]
    
    radius : radius (in astropy unit) of search to PS-1 catalog. 
            If not given, use the half diagonal length of the region.
            
    clean_catalog : whether to clean the matched catalog. (default True)
            The PS-1 catalog contains duplicate items on a single source with different
            measurements. If True, duplicate items of bright sources will be cleaned by 
            removing those with large coordinate errors and pick the items with most 
            detections in that band .
            
    mag_limit : magnitude threshould defining bright stars.
    
    sep : maximum separation (in astropy unit) for crossmatch with SE.
    
    Returns
    ----------
    tab_target : table containing matched bright sources with SE source catalog
    tab_target_all : table containing matched all sources with SE source catalog
    catalog_star : PS-1 catalog of all sources in the region(s)
        
    """
    from astropy.table import join, vstack
    from astropy.nddata.bitmask import interpret_bit_flags
    from .panstarrs import ps1cone
    
    band = band.lower()
    mag_name = band + 'MeanPSFMag'
    c_name = 'PS'
    
    for j, bounds in enumerate(np.atleast_2d(bounds)):
        cen = (bounds[2]+bounds[0])/2., (bounds[3]+bounds[1])/2.
        coord_cen = wcs_data.pixel_to_world(cen[0], cen[1])
        ra, dec = coord_cen.ra.value, coord_cen.dec.value

        if radius is None:
            L = math.sqrt((cen[0]-bounds[0])**2 + (cen[1]-bounds[1])**2)
            radius = (L * pixel_scale * u.arcsec).to(u.deg)

        print("Search", np.around(radius, 3), "around:")
        print(coord_cen)
        
        #### Query PANSTARRS start ####
        constraints = {'nDetections.gt':1, band+'MeanPSFMag.lt':23}

        # strip blanks and weed out blank and commented-out values
        columns = """raMean,decMean,raMeanErr,decMeanErr,nDetections,ng,nr,
                    gMeanPSFMag,gMeanPSFMagErr,gFlags,rMeanPSFMag,rMeanPSFMagErr,rFlags""".split(',')
        columns = [x.strip() for x in columns]
        columns = [x for x in columns if x and not x.startswith('#')]
        results = ps1cone(ra, dec, radius.value, release='dr2', columns=columns, **constraints)
        
        Cat_full = ascii.read(results)
        for filter in 'gr':
            col = filter+'MeanPSFMag'
            Cat_full[col].format = ".4f"
            Cat_full[col][Cat_full[col] == -999.0] = np.nan
        for coord in ['ra','dec']:
            Cat_full[coord+'MeanErr'].format = ".5f"
            
        #### Query PANSTARRS end ####

        Cat_full.sort(mag_name)
        Cat_full['raMean'].unit = u.deg
        Cat_full['decMean'].unit = u.deg
        Cat_full = transform_coords2pixel(Cat_full, wcs_data, name=c_name,
                                          RA_key="raMean", DE_key="decMean")
        
        # Crop catalog and sort by the catalog magnitude
        Cat_crop = crop_catalog(Cat_full, bounds, sortby=mag_name,
                                keys=("X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name))
        
        # Remove detection without magnitude
        has_mag = ~np.isnan(Cat_crop[mag_name])
        Cat_crop = Cat_crop[has_mag]
        
        # Pick out bright stars
        mag_cat = Cat_crop[mag_name]
        Cat_bright = Cat_crop[mag_cat<mag_limit]
        
        if clean_catalog:
            # A first crossmatch with bright stars in catalog for cleaning
            tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                             RA_key="raMean", DE_key="decMean")
            tab_match_bright.sort(mag_name)
            
            # Clean duplicate items in the catalog
            c_bright = SkyCoord(tab_match_bright['X_WORLD'],
                                tab_match_bright['Y_WORLD'], unit=u.deg)
            c_catalog = SkyCoord(Cat_crop['raMean'],
                                 Cat_crop['decMean'], unit=u.deg)
            idxc, idxcatalog, d2d, d3d = \
                        c_catalog.search_around_sky(c_bright, sep)
            inds_c, counts = np.unique(idxc, return_counts=True)
            
            row_duplicate = np.array([], dtype=int)
            
            # Use the measurement following some criteria
            for i in inds_c[counts>1]:
                obj_dup = Cat_crop[idxcatalog][idxc==i]
                obj_dup['sep'] = d2d[idxc==i]
                #obj_dup.pprint(max_lines=-1, max_width=-1)
                
                # Use the detection with mag
                mag_obj_dup = obj_dup[mag_name]
                obj_dup = obj_dup[~np.isnan(mag_obj_dup)]
                                
                # Use the closest match
                good = (obj_dup['sep'] == min(obj_dup['sep']))
                
### Extra Criteria             
#                 # Coordinate error of detection
#                 err2_coord = obj_dup["raMeanErr"]**2 + \
#                             obj_dup["decMeanErr"]**2
                
#                 # Use the detection with the best astrometry
#                 min_e2_coord = np.nanmin(err2_coord)
#                 good = (err2_coord == min_e2_coord)
                
#                 # Use the detection with PSF mag err
#                 has_err_mag = obj_dup[mag_name+'Err'] > 0   
                
#                 # Use the detection > 0
#                 n_det = obj_dup['n'+band]
#                 has_n_det = n_det > 0
                
#                 # Use photometry not from tycho in measurement
#                 use_tycho_phot =  extract_bool_bitflags(obj_dup[band+'Flags'], 7)
                
#                 good = has_err_mag & has_n_det & (~use_tycho_phot)
###
                    
                # Add rows to be removed
                for ID in obj_dup[~good]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
                
                obj_dup = obj_dup[good]
                
                if len(obj_dup)<=1:
                    continue
                
                # Use brightest detection
                mag = obj_dup[mag_name]
                for ID in obj_dup[mag>min(mag)]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
            
            # Remove rows
            Cat_crop.remove_rows(np.unique(row_duplicate))
            
            # Subset catalog containing bright stars
            Cat_bright = Cat_crop[Cat_crop[mag_name]<mag_limit]
        
        # Merge Catalog
        SE_columns = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
                      "MAG_AUTO", "FLUX_AUTO", "FWHM_IMAGE"]
        keep_columns = SE_columns + ["ID"+'_'+c_name] + columns + \
                                    ["X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name]
        tab_match = merge_catalog(SE_catalog, Cat_crop, sep=sep,
                                  RA_key="raMean", DE_key="decMean", keep_columns=keep_columns)
        tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                         RA_key="raMean", DE_key="decMean", keep_columns=keep_columns)
        
        if j==0:
            tab_target_all = tab_match
            tab_target = tab_match_bright
            catalog_star = Cat_crop
            
        else:
            tab_target_all = vstack([tab_target_all, tab_match], join_type='exact')
            tab_target = vstack([tab_target, tab_match_bright], join_type='exact')
            catalog_star = vstack([catalog_star, Cat_crop], join_type='exact')
            
    # Sort matched catalog by matched magnitude
    tab_target.sort(mag_name)
    tab_target_all.sort(mag_name)
    
    if verbose:
        print("Matched stars with PANSTARRS DR2 %s:  %.3f ~ %.3f"\
              %(mag_name, np.nanmin(tab_target_all[mag_name]),
                np.nanmax(tab_target_all[mag_name])))
        print("Matched bright stars with PANSTARRS DR2 %s:  %.3f ~ %.3f"\
              %(mag_name, np.nanmin(tab_target[mag_name]),
                np.nanmax(tab_target[mag_name])))
    
    return tab_target, tab_target_all, catalog_star
        

def add_supplementary_SE_star(tab, SE_catatlog, mag_saturate=13, draw=True):
    """ Add unmatched bright (saturated) stars in SE_catatlogas to tab.
        Magnitude is corrected by interpolation from other matched stars """
    
    print("Mannually add unmatched bright stars to the catalog.")
    
    # Empirical function to correct MAG_AUTO for saturation
    # Fit a sigma-clipped piecewise linear
    popt, clip_func = iter_curve_fit(tab['MAG_AUTO'], tab['MAG_AUTO_corr'],
                                    piecewise_linear, x_max=15,
                                    p0=(1, 2, mag_saturate, mag_saturate),
                                    x_lab='MAG_AUTO', y_lab='MAG_AUTO_corr', draw=draw)

    # Empirical corrected magnitude
    f_corr = lambda x: piecewise_linear(x, *popt)
    mag_corr = f_corr(tab['MAG_AUTO'])
    
    # Remove rows with uncanny matched magnitude
    loc_rm = np.where(abs(tab['MAG_AUTO_corr']-mag_corr)>2)
    if draw: plt.scatter(tab[loc_rm]['MAG_AUTO'], tab[loc_rm]['MAG_AUTO_corr'],
                         marker='s', s=40, fc='none', ec='lime'); plt.show()
    tab.remove_rows(loc_rm[0])

    # Below are new lines to add back supplementary (unmatched bright) stars
    cond_sup = (SE_catatlog['MAG_AUTO']<mag_saturate) & (SE_catatlog['CLASS_STAR']>0.5)
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
                        names=['X_IMAGE_PS', 'Y_IMAGE_PS'])
                        
    # Join the two tables by common keys
    keys = set(tab.colnames).intersection(tab_sup.colnames)
    tab_join = join(tab, tab_sup, keys=keys, join_type='outer')
    
    return tab_join
    
    
def calculate_color_term(tab_target, mag_range=[13,18], mag_name='gmag_PS', draw=True):
    """
    Use non-saturated stars to calculate Color Correction between SE MAG_AUTO and magnitude in the matched catalog . 
    
    Parameters
    ----------
    tab_target : full matched source catlog
    
    mag_range : range of magnitude for stars to be used
    mag_name : column name of magnitude in tab_target 
    draw : whethert to draw a plot showing MAG_AUTO vs diff.
    
    Returns
    ----------
    CT : color correction term (SE - catlog)
    """
    
    mag = tab_target["MAG_AUTO"]
    mag_cat = tab_target[mag_name]
    d_mag = tab_target["MAG_AUTO"] - mag_cat
    
    d_mag = d_mag[(mag>mag_range[0])&(mag<mag_range[1])&(~np.isnan(mag_cat))]
    mag = mag[(mag>mag_range[0])&(mag<mag_range[1])&(~np.isnan(mag_cat))]

    d_mag_clip = sigma_clip(d_mag, 3, maxiters=10)
    CT = np.mean(d_mag_clip)
    print('\nAverage Color Term [SE-catalog] = %.5f'%CT)
    
    if draw:
        plt.scatter(mag, d_mag, s=8, alpha=0.2, color='gray')
        plt.scatter(mag, d_mag_clip, s=6, alpha=0.3)
        plt.axhline(CT, color='k', alpha=0.7)
        plt.ylim(-3,3)
        plt.xlim(mag_range[0]-0.5, mag_range[1]+0.5)
        plt.xlabel("MAG_AUTO (SE)")
        plt.ylabel("MAG_AUTO - %s"%mag_name)
        plt.show()
        
    return np.around(CT,5)

def fit_empirical_aperture(tab_target, seg_map, mag_name='rmag_PS',
                           mag_range=[13, 22], K=2, degree=3, draw=True):
    """
    Fit an empirical polynomial curve for log radius of aperture based on corrected magnitudes and segm map of SE. Radius is enlarged K times.
    
    Parameters
    ----------
    tab_target : full matched source catlog
    seg_map : training segm map
    
    mag_name : column name of magnitude in tab_target 
    mag_range : range of magnitude for stars to be used
    K : enlargement factor on the original segm map
    degree : degree of polynomial (default 3)
    draw : whether to draw log R vs mag
    
    Returns
    ----------
    estimate_radius : a function turns magnitude into log R
    
    """
    
    
    print("\nFit %d-order empirical relation of aperture radii for catalog stars based on SE (X%.1f)"%(degree, K))

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

        plt.xlabel("%s (catalog)"%mag_name)
        plt.ylabel(r"$\log_{10}\,R$")
        plt.xlim(7,23)
        plt.ylim(0.15,2.2)
        plt.show()
    
    estimate_radius = lambda m: max(10**min(2, f_poly(m)), 2)
    
    return estimate_radius


def make_segm_from_catalog(catalog_star, bounds, estimate_radius,
                           mag_name='rmag', cat_name='PS', obj_name='', band='G',
                           extend_cat=None, draw=True, save=False, dir_name='./Measure'):
    """
    Make segmentation map from star catalog. Aperture size used is based on SE semg map.
    
    Parameters
    ----------
    catalog_star : star catalog
    bounds : 1X4 1d array defining bounds of region
    estimate_radius : function of turning magnitude into log R
    
    mag_name : magnitude column name in catalog_star
    cat_name : suffix of star catalog used
    extend_cat : (bright) extended source catalog to mask
    draw : whether to draw the output segm map
    save : whether to save the segm map as fits
    dir_name : path of saving
    
    Returns
    ----------
    seg_map : output segm map generated from catalog
    
    """
    
    image_size = bounds[2] - bounds[0]
    Xmin, Ymin = bounds[:2]
    X_key, Y_key = 'X_IMAGE'+'_'+cat_name, 'Y_IMAGE'+'_'+cat_name
    
    try:
        catalog = catalog_star[~catalog_star[mag_name].mask]
    except AttributeError:
        catalog = catalog_star[~np.isnan(catalog_star[mag_name])]
    print("\nMake segmentation map based on catalog %s: %d stars"%(mag_name, len(catalog)))
    
    # Estimate mask radius
    R_est = np.array([estimate_radius(m) for m in catalog[mag_name]])
    
    # Generate object apertures
    apers = [CircularAperture((X_c-Xmin, Y_c-Ymin), r=r)
             for (X_c,Y_c, r) in zip(catalog[X_key], catalog[Y_key], R_est)]
    
    # Further mask for bright extended sources
    if extend_cat is not None:
        if len(extend_cat)>0:
            for (X_c,Y_c, r) in zip(extend_cat['X_IMAGE'],
                                    extend_cat['Y_IMAGE'],
                                    extend_cat['A_IMAGE']*3):
                apers.append(CircularAperture((X_c-Xmin, Y_c-Ymin), r=r))
            
    # Draw segment map generated from the catalog
    seg_map = np.zeros((image_size, image_size))
    
    # Segmentation k sorted by mag of source catalog
    for (k, aper) in enumerate(apers):
        star_ma = aper.to_mask(method='center').to_image((image_size, image_size))
        if star_ma is not None:
            seg_map[star_ma.astype(bool)] = k+2
    if draw:
        from .plotting import make_rand_cmap
        plt.figure(figsize=(5,5))
        plt.imshow(seg_map, vmin=1, cmap=make_rand_cmap(int(seg_map.max())))
        plt.show()
        
    # Save segmentation map built from catalog
    if save:
        check_save_path(dir_name, make_new=False, verbose=False)
        hdu_seg = fits.PrimaryHDU(seg_map.astype(int))
        
        b_name = band.lower()
        file_name = os.path.join(dir_name, "%s-segm_%smag_catalog_X%dY%d.fits" %(obj_name, b_name, Xmin, Ymin))
        hdu_seg.writeto(file_name, overwrite=True)
        print("Save segmentation map made from catalog as %s\n"%file_name)
        
    return seg_map


def save_thumbs(obj, filename):
    import pickle
    fname = filename+'.pkl'
    print("Save thumbs to: %s"%fname)
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_thumbs(filename):
    import pickle
    fname = filename+'.pkl'
    print("Read thumbs from: %s"%fname) 
    with open(fname, 'rb') as f:
        return pickle.load(f)

### Prior Helper ###    
def build_independent_priors(priors):
    """ Build priors for Bayesian fitting. Priors should has a (scipy-like) ppf class method."""
    def prior_transform(u):
        v = u.copy()
        for i in range(len(u)):
            v[i] = priors[i].ppf(u[i])
        return v
    return prior_transform    



### Recostruct PSF from fit ###    
    
def make_psf_from_fit(fit_res, psf, image_size=600, n_out=4, theta_out=1200, n_spline=2,
                      fit_sigma=True, fit_frac=False, leg2d=False, sigma=None):
    from .sampler import get_params_fit

    psf_fit = psf.copy()
    
    params, _, _ = get_params_fit(fit_res)
        
    K = 0
    if fit_frac: K += 1        
    if fit_sigma: K += 1
    
    if psf.aureole_model == "moffat":
        gamma1_fit, beta1_fit = params[:2]
        param_update = {'gamma1':gamma1_fit, 'beta1':beta1_fit}
        
    else:
        N_n = n_spline
        N_theta = n_spline - 1
    
        if psf.aureole_model == "power":
            n_fit = params[0]
            param_update = {'n':n_fit}

        elif psf.aureole_model == "multi-power":
            n_s_fit = np.concatenate([params[:N_n], [n_out]])
            theta_s_fit = np.concatenate([[psf.theta_0],
                          np.atleast_1d(10**params[N_n:N_n+N_theta]),[theta_out]])

            param_update = {'n_s':n_s_fit, 'theta_s':theta_s_fit}

    if fit_frac:
        frac = 10**params[-1]
        param_update['frac'] = frac

    psf_fit.update(param_update)

    mu_fit = params[-K-1]
    
    if fit_sigma:
        sigma_fit = 10**params[-K]
    else:
        sigma_fit = sigma
        
    if leg2d:
        psf_fit.A10, psf_fit.A01 = 10**params[-K-2], 10**params[-K-3]
        
    psf_fit.bkg, psf_fit.bkg_std  = mu_fit, sigma_fit
    
    _ = psf_fit.generate_core()
    _, _ = psf_fit.generate_aureole(psf_range=image_size)
    
    return psf_fit, params


def calculate_reduced_chi2(fit, data, uncertainty, dof=5):
#     uncertainty = 10**params[-1]
    chi2_reduced = np.sum((fit-data)**2/uncertainty**2)/(len(data)-dof)
    print("Reduced Chi^2: %.5f"%chi2_reduced)


class MyError(Exception): 
    def __init__(self, message):  self.message = message 
    def __str__(self): return(repr(self.message))
    def __repr__(self): return 'MyError(%r)'%(str(self))

class InconvergenceError(MyError): 
    def __init__(self, message):  self.message = message 
    def __repr__(self):
        return 'InconvergenceError: %r'%self.message
