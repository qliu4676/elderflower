import os
import re
import sys
import math
import time
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.spatial import distance
from scipy.special import gamma as Gamma
from skimage import morphology

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, Column, join
from astropy.modeling import models
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, median_absolute_deviation, gaussian_fwhm_to_sigma
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch
norm0 = ImageNormalize(stretch=AsinhStretch())
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())

from photutils.segmentation import SegmentationImage
from photutils import detect_sources, deblend_sources
from photutils import CircularAperture, CircularAnnulus, EllipticalAperture

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import multiprocess as mp


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
    I[I<BKG] = np.nan
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
    return min(2**math.ceil(math.log2(x)), 3 * 2**math.floor(math.log2(x)-1))
    
### Plotting Helpers ###

def vmin_3mad(img):
    """ lower limit of visual imshow defined by 3 mad above median """ 
    return np.median(img)-3*mad_std(img)

def vmax_2sig(img):
    """ upper limit of visual imshow defined by 2 sigma above median """ 
    return np.median(img)+2*np.std(img)

def colorbar(mappable, pad=0.2, size="5%", loc="right", color_nan='gray', **args):
    """ Customized colorbar """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    
    if loc=="bottom":
        orent = "horizontal"
        pad = 1.5*pad
        rot = 75
    else:
        orent = "vertical"
        rot = 0
    
    cax = divider.append_axes(loc, size=size, pad=pad)
    
    cb = fig.colorbar(mappable, cax=cax, orientation=orent, **args)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(),rotation=rot)
    
    cmap = cb.get_cmap()
    cmap.set_bad(color=color_nan, alpha=0.3)
    
    return cb


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
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    ax1.imshow(field, aspect="auto", cmap="gray", vmin=vmin_3mad(field), vmax=vmax_2sig(field),norm=norm1)
    im2 = ax2.imshow(back, aspect="auto", cmap='gray')
    colorbar(im2)
    ax3.imshow(field - back, aspect="auto", cmap='gray', vmin=0., vmax=vmax_2sig(field - back),norm=norm2)
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

def make_mask_map(image, sn_thre=3, b_size=25, npix=5, n_dilation=3):
    """ Make mask map with S/N > sn_thre """
    from photutils import detect_sources, deblend_sources
    
    # detect source
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=npix)
    
    # dilation
    segmap = segm0.data.copy()
    for i in range(n_dilation):
        segmap = morphology.dilation(segmap)
        
    segmap[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap!=0)
    
    return mask_deep, segmap

def make_mask_map_core(image, star_pos, r_core=12):
    """ Make stars out to r_core """

    # mask core
    yy, xx = np.indices(image.shape)
    mask_deep = np.zeros_like(image, dtype=bool)
    
    if np.ndim(r_core) == 0:
        r_core = np.ones(len(star_pos)) * r_core
    
    core_region= np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r for (pos,r) in zip(star_pos,r_core)])
    
    mask_deep[core_region] = 1
    segmap = mask_deep.astype(int).copy()
    
    return mask_deep, segmap

def make_mask_map_dual(image, stars, xx=None, yy=None,
                       pad=0, r_core=24, r_out=96,
                       mask_base=None, n_bright=25, sn_thre=3, 
                       nlevels=64, contrast=0.001, npix=4, 
                       b_size=64, n_dilation=3):
    """ Make mask map in dual mode: for faint stars, mask with S/N > sn_thre;
    for bright stars, mask core (r < r_core pix) """
    from photutils import detect_sources, deblend_sources
    
    if (xx is None) | (yy is None):
        yy, xx = np.mgrid[:image.shape[0]+2*pad, :image.shape[1]+2*pad]
        
    star_pos = stars.star_pos_bright + pad
    
    r_core_s = np.unique(r_core)[::-1]
    if len(r_core_s) == 1:
        r_core_A, r_core_B = r_core_s, r_core_s
        r_core_s = np.ones(len(star_pos)) * r_core_s
    else:
        r_core_A, r_core_B = r_core_s[:2]
        r_core_s = np.array([r_core_A if F >= stars.F_verybright else r_core_B
                             for F in stars.Flux_bright])

    if r_out is not None:
        r_out_s = np.unique(r_out)[::-1]
        if len(r_out_s) == 1:
            r_out_A, r_out_B = r_out_s, r_out_s
            r_out_s = np.ones(len(star_pos)) * r_out_s
        else:
            r_out_A, r_out_B = r_out_s[:2]
            r_out_s = np.array([r_out_A if F >= stars.F_verybright else r_out_B
                                 for F in stars.Flux_bright])
        print("Mask outer regions: r > %d (%d) pix "%(r_out_A, r_out_B))
            
    if sn_thre is not None:
        print("Detect and deblend source... Mask S/N > %.1f (%dth enlarged)"%(sn_thre, n_dilation))
        # detect all source first 
        back, back_rms = background_sub_SE(image, b_size=b_size)
        threshold = back + (sn_thre * back_rms)
        segm0 = detect_sources(image, threshold, npixels=npix)

        # deblend source
        segm_deb = deblend_sources(image, segm0, npixels=npix,
                                   nlevels=nlevels, contrast=contrast)

    #     for pos in star_pos:
    #         if (min(pos[0],pos[1]) > 0) & (pos[0] < image.shape[0]) & (pos[1] < image.shape[1]):
    #             star_lab = segmap[coord_Im2Array(pos[0], pos[1])]
    #             segm_deb.remove_label(star_lab)

        segmap = segm_deb.data.copy()
        max_lab = segm_deb.max_label

        # remove S/N mask map for input (bright) stars
        for pos in star_pos:
            rr2 = (xx-pos[0])**2+(yy-pos[1])**2
            lab = segmap[np.where(rr2==np.min(rr2))][0]
            segmap[segmap==lab] = 0

        # dilation
        for i in range(n_dilation):
            segmap = morphology.dilation(segmap)
            
    if mask_base is not None:
        print("Use mask map built from catalog: ", mask_base)
        segmap2 = fits.getdata(mask_base)
        
        if sn_thre is not None:
            # Combine Two mask
            segmap[segmap2>n_bright] = max_lab + segmap2[segmap2>n_bright]
            segm_deb = SegmentationImage(segmap)
        else:
            # Only use mask_base
            segm_deb = SegmentationImage(segmap2)
        
        max_lab = segm_deb.max_label
            
    # mask core for input (bright) stars
    print("Mask core regions: r < %d (%d) pix "%(r_core_A, r_core_B))
    core_region = np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r
                                        for (pos,r) in zip(star_pos,r_core_s)])
    mask_star = core_region.copy()
    
    if r_out is not None:
        # mask outer region for input (bright) stars
        outskirt = np.logical_and.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) > r
                                         for (pos,r) in zip(star_pos,r_out_s)])
        mask_star = (mask_star) | (outskirt)
    
    segmap[mask_star] = max_lab+2
    
    # set dilation border a different label (for visual)
    segmap[(segmap!=0)&(segm_deb.data==0)] = max_lab+1
    
    # set mask map
    mask_deep = (segmap!=0)
    
    return mask_deep, segmap

def make_mask_strip(stars, xx, yy, pad=0,
                    width=10, n_strip=16, dist_strip=500, dist_cross=100):    
    """ Make mask map in strips with width=width """
    
    print("Use sky strips crossing very bright stars")
    
    if stars.n_verybright>0:
        mask_strip_s = np.empty((stars.n_verybright, xx.shape[0], xx.shape[1]))
        mask_cross_s = np.empty_like(mask_strip_s)
    else:
        return None, None
    
    star_pos = stars.star_pos_verybright + pad
    
    phi_s = np.linspace(-90, 90, n_strip+1)
#     phi_s = np.setdiff1d(phi_s, [-90,0,90])
    a_s = np.tan(phi_s*np.pi/180)
    
    for k, (x_b, y_b) in enumerate(star_pos):
        m_s = (y_b-a_s*x_b)
        mask_strip = np.logical_or.reduce([abs((yy-a*xx-m)/math.sqrt(1+a**2)) < width 
                                           for (a, m) in zip(a_s, m_s)])
        mask_cross = np.logical_or.reduce([abs(yy-y_b)<10, abs(xx-x_b)<10])
        dist_map1 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_strip
        dist_map2 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_cross
        mask_strip_s[k] = mask_strip & dist_map1
        mask_cross_s[k] = mask_cross & dist_map2

    return mask_strip_s, mask_cross_s

def clean_lonely_stars(xx, yy, mask, star_pos, pad=0, dist_clean=60):
    
    star_pos = star_pos + pad
    
    clean = np.zeros(len(star_pos), dtype=bool)
    for k, pos in enumerate(star_pos):
        rr = np.sqrt((xx-pos[0])**2+(yy-pos[1])**2)
        if np.min(rr[~mask]) > dist_clean:
            clean[k] = True
            
    return clean
        
def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                   color="steelblue", xunit="pix", yunit="intensity",
                   seeing=2.5, pixel_scale=2.5, ZP=27.1, 
                   sky_mean=884, sky_std=3, dr=1.5, 
                   lw=2, alpha=0.7, markersize=5, I_shift=0,
                   core_undersample=False, label=None, plot_line=False, mock=False,
                   plot=True, scatter=False, fill=False, errorbar=False, verbose=False):
    """Calculate 1d radial profile of a given star postage"""
    if mask is None:
        mask =  np.zeros_like(img, dtype=bool)
    if back is None:     
        back = np.ones_like(img) * sky_mean
    if cen is None:
        cen = (img.shape[0]-1)/2., (img.shape[1]-1)/2.
        
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
        clip = lambda z: sigma_clip((z), sigma=3, maxiters=5)
    else:
        clip = lambda z: 10**sigma_clip(np.log10(z+1e-10), sigma=3, maxiters=5)
        
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:  
            # for undersampled core, bin in individual pixels 
            bins_inner = np.unique(r[r<r_core]) + 1e-3
        else: 
            bins_inner = np.linspace(0, r_core, int(min((r_core/d_r*2), 5))) + 1e-3

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
        
        r_rbin = np.append(r_rbin, np.mean(r[in_bin]))
        z_clip = clip(z[in_bin])

        zb = np.mean(z_clip)
        zstd_b = np.std(z_clip)
        
        z_rbin = np.append(z_rbin, zb)
        zstd_rbin = np.append(zstd_rbin, zstd_b)
        
    logzerr_rbin = 0.434 * abs( zstd_rbin / (z_rbin-sky_mean))
    
    if plot:
        if yunit == "Intensity":  
            # plot radius in Intensity
            plt.plot(r_rbin, np.log10(z_rbin), "-o", mec="k", lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label) 
            if scatter:
                plt.scatter(r[r<3*r_core], np.log10(z[r<3*r_core]), color=color, s=6, alpha=0.2, zorder=1)
                plt.scatter(r[r>3*r_core], np.log10(z[r>3*r_core]), color=color, s=3, alpha=0.1, zorder=1)
            if fill:
                plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                                 color=color, alpha=0.2, zorder=1)
            plt.ylabel("log Intensity")
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8, r_rbin[np.isfinite(r_rbin)][-1]*1.2)

        elif yunit == "SB":  
            # plot radius in Surface Brightness
            I_rbin = Intensity2SB(I=z_rbin, BKG=np.median(back),
                                  ZP=ZP, pixel_scale=pixel_scale) + I_shift
            I_sky = -2.5*np.log10(sky_std) + ZP + 2.5 * math.log10(pixel_scale**2)

            plt.plot(r_rbin, I_rbin, "-o", mec="k", lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label)   
            
            if scatter:
                I = Intensity2SB(I=z, BKG=np.median(back),
                                 ZP=ZP, pixel_scale=pixel_scale) + I_shift
                plt.scatter(r[r<3*r_core], I[r<3*r_core],
                            color=color, s=6, alpha=0.2, zorder=1)
                plt.scatter(r[r>3*r_core], I[r>3*r_core],
                            color=color, s=3, alpha=0.1, zorder=1)
                
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
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8,r_rbin[np.isfinite(r_rbin)][-1]*1.2)
            plt.ylim(30,17)
            
        plt.xlabel("r [acrsec]") if xunit == "arcsec" else plt.xlabel("r [pix]")

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
        dz_rbin = np.diff(np.log10(z_rbin)) 
        dz_cum = np.cumsum(dz_rbin)

        if plot_line:
            r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
            plt.axvline(r_satr,color="k",ls="--",alpha=0.9)
            plt.axvline(r_core,color="k",ls=":",alpha=0.9)
            plt.axhline(I_sky,color="gray",ls="-.",alpha=0.7)

#             use_range = (r_rbin>r_satr) & (r_rbin<r_core)
#         else:
#             use_range = True
            
    return r_rbin, z_rbin, logzerr_rbin

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
    win_size = int( n_win * min(max(fwhm,2), 8))
    
    # calculate boundary
    X_min, X_max = X_c - win_size, X_c + win_size
    Y_min, Y_max = Y_c - win_size, Y_c + win_size
    x_min, y_min = coord_Im2Array(X_min, Y_min, origin)
    x_max, y_max = coord_Im2Array(X_max, Y_max, origin)
    
    if verbose:
        num = star_cat[id]["NUMBER"]
        print("NUMBER: ", num)
        print("X_c, Y_c: ", (X_c, Y_c))
        print("RA, DEC: ", (star_cat[id]["X_WORLD"], star_cat[id]["Y_WORLD"]))
        print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
        print("X_min, X_max, Y_min, Y_max: ", X_min, X_max, Y_min, Y_max)
    
    # crop
    img_thumb = data[x_min:x_max, y_min:y_max].copy()
    if seg_map is None:
        seg_thumb = None
    else:
        seg_thumb = seg_map[x_min:x_max, y_min:y_max]
    mask_thumb = (seg_thumb!=0)    
    
    # the center position is converted from world with wcs
    X_cen, Y_cen = wcs.wcs_world2pix(star_cat[id]["X_WORLD"], star_cat[id]["Y_WORLD"], origin)
    cen_star = X_cen - X_min, Y_cen - Y_min
    
    return (img_thumb, seg_thumb, mask_thumb), cen_star
    
def extract_star(id, star_cat, wcs, data, seg_map=None, 
                 seeing=2.5, sn_thre=3, n_win=20, n_dilation=1,
                 display_bg=False, display=True, verbose=False):
    
    """ Return the image thubnail, mask map, backgroud estimates, and center of star.
        Do a finer detection&deblending to remove faint undetected source."""
    
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
    star_lab = segm_deblend.data[img_thumb.shape[0]//2, img_thumb.shape[1]//2]
    star_ma = ~((segm_deblend.data==star_lab) | (segm_deblend.data==0)) # mask other source
        
    # dilation
    for i in range(n_dilation):
        star_ma = morphology.dilation(star_ma)
    
    if display:
        med_back = np.median(back)
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
        ax1.imshow(img_thumb, vmin=med_back-1, vmax=10000, norm=norm1, cmap="viridis")
        ax1.set_title("star", fontsize=16)

        ax2.imshow(segm_deblend, cmap=segm_deblend.make_cmap(random_state=12345))
        ax2.set_title("segment", fontsize=16)

        img_thumb_ma = img_thumb.copy()
        img_thumb_ma[star_ma] = -1
        ax3.imshow(img_thumb_ma, cmap="viridis", norm=norm2,
                   vmin=med_back-1, vmax=med_back+10*np.median(back_rms))
        ax3.set_title("extracted star", fontsize=16)
        plt.tight_layout()
    
    return img_thumb, star_ma, back, cen_star


def compute_Rnorm(image, mask_field, cen, R=10, wid=0.5, mask_cross=True, display=False):
    """ Return 3 sigma-clipped mean, med and std of ring r=R (half-width=wid) for image.
        Note intensity is not background subtracted. """
    
    annulus_ma = CircularAnnulus(cen, R-wid, R+wid).to_mask()      
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
    
    z = 10**sigma_clip(np.log10(image[mask_clean]), sigma=3, maxiters=10)
    I_mean, I_med, I_std = np.mean(z), np.median(z.compressed()), np.std(z)
    
    if display:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        ax1.imshow(mask_clean * image, cmap="gray", norm=norm1, vmin=I_med-5*I_std, vmax=I_med+5*I_std)
        ax2 = plt.hist(sigma_clip(z))
        plt.axvline(I_mean, color='k')
    
    return I_mean, I_med, I_std, 0


def compute_Rnorm_batch(table_target, data, seg_map, wcs,
                        R=10, wid=0.5, return_full=False, verbose=True):
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
        I_mean, I_med, I_std, Iflag = compute_Rnorm(img, ma, cen, R=R, wid=wid)
        
        if (Iflag==1) & verbose: print ("Errorenous measurement: #", num)
        
        # Use the median value of background as the local background
        sky_mean = np.median(bkg)
        
        res_Rnorm[i] = np.array([I_mean, I_med, I_std, sky_mean, Iflag])
    
    return res_Rnorm, res_thumb

def measure_Rnorm_all(table, image_bound,
                      wcs_data, image, seg_map=None, 
                      r_scale=10, width=0.5, mag_thre=15,
                      read=False, save=True,
                      mag_name='rmag_PS', obj_name="",
                      dir_name='.', verbose=True):
    """ Measure normalization at r_scale for bright stars in table.
        If seg_map is not given, source detection will be run."""
    
    Xmin, Ymin = image_bound[:2]
    
    table_Rnorm_name = os.path.join(dir_name, '%s-norm_%dpix_%s%dmag_X%sY%s.txt'\
                                    %(obj_name, r_scale, mag_name[0], mag_thre, Xmin, Ymin))
    res_thumb_name = os.path.join(dir_name, '%s-thumbnail_%s%dmag_X%sY%s'\
                                  %(obj_name, mag_name[0], mag_thre, Xmin, Ymin))
    if read:
        table_res_Rnorm = Table.read(table_Rnorm_name, format="ascii")
        res_thumb = load_thumbs(res_thumb_name)
        
    else:
        tab = table[table[mag_name]<mag_thre]
        res_Rnorm, res_thumb = compute_Rnorm_batch(tab, image, seg_map, wcs_data,
                                                   R=r_scale, wid=width, return_full=True, verbose=verbose)

        table_res_Rnorm = tab['NUMBER', 'MAG_AUTO', 'MAG_AUTO_corr',
                              mag_name, 'X_IMAGE', 'Y_IMAGE'].copy()
    
        for j, name in enumerate(['Imean','Imed','Istd','Isky', 'Iflag']):
            table_res_Rnorm[name] = res_Rnorm[:,j]
        
        # Remove dubious measurement in the model list
        table_res_Rnorm = table_res_Rnorm[table_res_Rnorm["Iflag"]==0]
        
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
        sky_mean = np.median(sky)
        sky_std = max(mad_std(sky[sky>sky_mean]),5)

        fig, ax = plt.subplots(figsize=(12,8))       
        plt.imshow(data, norm=norm1, cmap="viridis",
                   vmin=sky_mean, vmax=sky_mean+10*sky_std, alpha=0.95)
        if weight_map is not None:
            plt.imshow(data*weight_map, norm=norm1, cmap="viridis",
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
    
    from astropy.table import Column, join
    
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

def cross_match(header, SE_catalog, bounds, radius=None, 
                pixel_scale=2.5, mag_thre=15, sep=5*u.arcsec,
                clean_catalog=True, mag_name='rmag',
                catalog={'Pan-STARRS': 'II/349/ps1'},
                columns={'Pan-STARRS': ['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
                                        'Qual', 'gmag', 'e_gmag', 'rmag', 'e_rmag']},
                column_filters={'Pan-STARRS': {'rmag':'{0} .. {1}'.format(5, 23)}},
                magnitude_name={'Pan-STARRS':['rmag','gmag']}):
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
    
    wcs_data = wcs.WCS(header)
    
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
        
        Cat_full = transform_coords2pixel(Cat_full, wcs_data, name=c_name)
        mag_full = Cat_full[m_name]
        mag_full[np.isnan(mag_full)] = 99
        Cat_bright = Cat_full[mag_full<mag_thre]
        mag_full.mask[mag_full==99] = True

        if clean_catalog:
            # Clean duplicate items in the catalog
            c_bright = SkyCoord(Cat_bright['RAJ2000'], Cat_bright['DEJ2000'], unit=u.deg)
            c_catalog = SkyCoord(Cat_full['RAJ2000'], Cat_full['DEJ2000'], unit=u.deg)
            idxc, idxcatalog, d2d, d3d = c_catalog.search_around_sky(c_bright, sep)
            inds_c, counts = np.unique(idxc, return_counts=True)
            
            row_duplicate = np.array([], dtype=int)
            
            # Use the measurement with min error in RA/DEC
            for i in inds_c[counts>1]:
                obj_duplicate = Cat_full[idxcatalog][idxc==i]
                
                has_mag = obj_duplicate[m_name]>0
                obj_duplicate = obj_duplicate[has_mag]
                
                e2_coord = obj_duplicate["e_RAJ2000"]**2 + obj_duplicate["e_DEJ2000"]**2
                min_e2_coord = np.nanmin(e2_coord)
                
                for ID in obj_duplicate[e2_coord>min_e2_coord]['ID'+'_'+c_name]:
                    k = np.where(Cat_full['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
            
            Cat_full.remove_rows(np.unique(row_duplicate))
            
        for m_name in magnitude_name[cat_name]:
            mag = Cat_full[m_name]
            print("%s %s:  %.3f ~ %.3f"%(cat_name, m_name, mag.min(), mag.max()))

        # Merge Catalog
        SE_columns = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
                      "MAG_AUTO", "FLUX_AUTO", "FWHM_IMAGE", "FLAGS"]
        keep_columns = SE_columns + ["ID"+'_'+c_name] + magnitude_name[cat_name] + \
                                    ["X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name]
        tab_match = merge_catalog(SE_catalog, Cat_full, sep=sep,
                                  keep_columns=keep_columns)
        tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                         keep_columns=keep_columns)
        
        for m_name in magnitude_name[cat_name]:
            tab_match[m_name].name = m_name+'_'+c_name
            tab_match_bright[m_name].name = m_name+'_'+c_name
        
        if j==0:
            tab_match_all = tab_match
            tab_match_bright_all = tab_match_bright
        else:
            tab_match_all = join(tab_match_all, tab_match, keys=SE_columns,
                                 join_type='left', metadata_conflicts='silent')
            tab_match_bright_all = join(tab_match_bright_all, tab_match_bright, keys=SE_columns,
                                        join_type='left', metadata_conflicts='silent')
            
    # Crop & Sort matched catalog by SE MAG_AUTO, and source catalog by the first used mag_name.
    tab_target_full = crop_catalog(tab_match_all, bounds,
                                   keys=("X_IMAGE", "Y_IMAGE"), sortby='MAG_AUTO')
    tab_target = crop_catalog(tab_match_bright_all, bounds,
                              keys=("X_IMAGE", "Y_IMAGE"), sortby='MAG_AUTO')
    
    Cat_crop = crop_catalog(Cat_full, bounds, sortby=np.atleast_1d(mag_name)[0],
                            keys=("X_IMAGE"+'_'+c_name, "Y_IMAGE"+'_'+c_name))

    mag = tab_target_full[mag_name+'_'+c_name]
    print("Matched stars with %s %s:  %.3f ~ %.3f"%(cat_name, mag_name, mag.min(), mag.max()))
    mag = tab_target[mag_name+'_'+c_name]
    print("Matched bright stars with %s %s:  %.3f ~ %.3f"\
          %(cat_name, mag_name, mag.min(), mag.max()))
    
    return tab_target, tab_target_full, Cat_crop

def calculate_color_term(tab_target, mag_range=[13,18], mag_name='gmag_PS', draw=True):
    """ Use non-saturated stars to calculate Color Correction between SE MAG_AUTO and matched mag. """
    mag = tab_target["MAG_AUTO"]
    d_mag = tab_target["MAG_AUTO"] - tab_target[mag_name]
    
    d_mag = d_mag[(mag>mag_range[0])&(mag<mag_range[1])]
    mag = mag[(mag>mag_range[0])&(mag<mag_range[1])]

    d_mag_clip = sigma_clip(d_mag, 3, maxiters=10)
    CT = np.mean(d_mag_clip)
    print('\nAverage Color Term [SE-catalog] = %.5f'%CT)
    
    if draw:
        plt.scatter(mag, d_mag, s=8, alpha=0.2, color='gray')
        plt.scatter(mag, d_mag_clip, s=5, alpha=0.3)
        plt.axhline(CT, color='k', alpha=0.7)
        plt.ylim(-3,3)
        plt.xlabel("MAG_AUTO (SE)")
        plt.ylabel("MAG_AUTO $-$ %s"%mag_name)
        plt.show()
        
    return np.around(CT,5)

def fit_empirical_aperture(tab_SE, seg_map, mag_name='rmag_PS',
                           mag_range=[12, 20], K=2, degree=3, draw=True):
    """ Fit an empirical curve for log radius of aperture on magnitude of stars in mag_range
        based on SE segmentation. Radius is enlarged K times."""
    
    print("\nFit %d-order empirical relation of aperture radii for catalog stars based on SE (X%.1f)."%(degree, K))

    # Read from SE segm map
    segm_deb = SegmentationImage(seg_map)
    R_aper = (segm_deb.get_areas(tab_SE["NUMBER"])/np.pi)**0.5
    tab_SE['logR'] = np.log10(K * R_aper)
    
    mag_match = tab_SE[mag_name]
    mag_match[np.isnan(mag_match)] = -1
    tab = tab_SE[(mag_match>mag_range[0])&(mag_match<mag_range[1])]

    mag_all = tab[mag_name]
    logR = tab['logR']
    p_poly = np.polyfit(mag_all, logR, degree)
    f_poly = np.poly1d(p_poly)

    if draw:
        plt.scatter(tab_SE[mag_name], tab_SE['logR'], s=8, alpha=0.2, color='gray')
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


def make_segm_from_catalog(catalog_star, image_bound, estimate_radius,
                           mag_name='rmag', cat_name='PS',
                           draw=True, save=False, dir_name='./Measure'):
    """ Make segmentation , aperture size from SE is fit based on SE"""
    
    catalog = catalog_star[~catalog_star[mag_name].mask]
    print("\nMake segmentation map based on catalog %s %s: %d stars"%(cat_name, mag_name, len(catalog)))
    
    R_est = np.array([estimate_radius(m) for m in catalog[mag_name]])
    Xmin, Ymin = image_bound[:2]

    apers = [CircularAperture((X_c-Xmin, Y_c-Ymin), r=r)
             for (X_c,Y_c, r) in zip(catalog['X_IMAGE'+'_'+cat_name],
                                     catalog['Y_IMAGE'+'_'+cat_name], R_est)] 
    
    image_size = image_bound[2] - image_bound[0]
    seg_map_catalog = np.zeros((image_size, image_size))
    
    # Segmentation k sorted by mag of source catalog
    for (k, aper) in enumerate(apers):
        star_ma = aper.to_mask(method='center').to_image((image_size, image_size))
        if star_ma is not None:
            seg_map_catalog[star_ma.astype(bool)] = k+2
    if draw:
        from plotting import make_rand_cmap
        plt.figure(figsize=(5,5))
        plt.imshow(seg_map_catalog, vmin=1, cmap=make_rand_cmap(int(seg_map_catalog.max())))
        plt.show()
        
    # Save segmentation map built from catalog
    if save:
        check_save_path(dir_name, make_new=False, verbose=False)
        hdu_seg = fits.PrimaryHDU(seg_map_catalog.astype(int))
        file_name = os.path.join(dir_name, "Seg_%s_X%dY%d.fits" %(cat_name, Xmin, Ymin))
        hdu_seg.writeto(file_name, overwrite=True)
        print("Save segmentation map made from catalog as %s\n"%file_name)
        
    return seg_map_catalog


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

### Nested Fitting Helper ###

class DynamicNestedSampler:
    def __init__(self,  loglike,  prior_transform, ndim,
                 sample='auto', bound='multi',
                 n_cpu=None, n_thread=None):
        
        self.ndim = ndim

        if n_cpu is None:
            n_cpu = mp.cpu_count()
            
        if n_thread is not None:
            n_thread = max(n_thread, n_cpu-1)
        
        if n_cpu > 1:
            self.open_pool(n_cpu)
            use_pool = {'update_bound': False}
        else:
            self.pool=None
            use_pool = None
            
        dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim,
                                                sample=sample, bound=bound,
                                                pool=self.pool, queue_size=n_thread,
                                                use_pool=use_pool)
        self.dsampler = dsampler
        
        
    def run_fitting(self, nlive_init=100,
                    maxiter=10000,
                    nlive_batch=50, maxbatch=2,
                    pfrac=0.8, close_pool=True,
                    print_progress=True):
    
        print("Run Nested Fitting for the image... Dim of params: %d"%self.ndim)
        start = time.time()
   
        dlogz = 1e-3 * (nlive_init - 1) + 0.01
        
        self.dsampler.run_nested(nlive_init=nlive_init, 
                                 nlive_batch=nlive_batch, 
                                 maxbatch=maxbatch,
                                 maxiter=maxiter,
                                 dlogz_init=dlogz, 
                                 wt_kwargs={'pfrac': pfrac},
                                 print_progress=print_progress) 
        
        end = time.time()
        self.run_time = (end-start)
        
        print("\nFinish Fitting! Total time elapsed: %.3g s"%self.run_time)
        
        if (self.pool is not None) & close_pool:
            self.close_pool()
        
    def open_pool(self, n_cpu):
        print("\nOpening new pool: # of CPU used: %d"%(n_cpu - 1))
        self.pool = mp.Pool(processes=n_cpu - 1)
        self.pool.size = n_cpu - 1
    
    def close_pool(self):
        print("\nPool Closed.")
        self.pool.close()
        self.pool.join()
    
    @property
    def results(self):
        res = getattr(self.dsampler, 'results', {})
        return res
    
    def save_result(self, filename, fit_info=None, dir_name='.'):
        res = {}
        if fit_info is not None:
            for key, val in fit_info.items():
                res[key] = val

        res['run_time'] = self.run_time
        res['fit_res'] = self.results
        
        fname = os.path.join(dir_name, filename)
        save_nested_fitting_result(res, fname)
        
        self.res = res
    
    def cornerplot(self, labels=None, truths=None, figsize=(16,15),
                   save=False, dir_name='.', suffix=''):
        from plotting import draw_cornerplot
        draw_cornerplot(self.results, self.ndim,
                        labels=labels, truths=truths, figsize=figsize,
                        save=save, dir_name=dir_name, suffix=suffix)
        
    def cornerbound(self, prior_transform, labels=None, figsize=(10,10),
                    save=False, dir_name='.', suffix=''):
        fig, axes = plt.subplots(self.ndim-1, self.ndim-1, figsize=figsize)
        fg, ax = dyplot.cornerbound(self.results, it=1000, labels=labels,
                                    prior_transform=prior_transform,
                                    show_live=True, fig=(fig, axes))
        if save:
            plt.savefig(os.path.join(dir_name, "Cornerbound%s.png"%suffix), dpi=120)
            plt.close()
    
    def plot_fit_PSF1D(self, psf, **kwargs):
        from plotting import plot_fit_PSF1D
        plot_fit_PSF1D(self.results, psf, **kwargs)
    
    def generate_fit(self, psf, stars, image_base, draw_real=True,
                     norm='brightness', leg2d=False, n_out=4, theta_out=1200):
        psf_fit, params = make_psf_from_fit(self.results, psf, leg2d=leg2d,
                                            n_out=n_out, theta_out=theta_out)
        from modeling import generate_image_fit
        image_star, noise_fit, bkg_fit = generate_image_fit(psf_fit, stars, norm=norm,
                                                            draw_real=draw_real, leg2d=leg2d)
        image_fit = image_star + image_base + bkg_fit
        
        self.image_fit = image_fit
        self.image_star = image_star
        self.bkg_fit = bkg_fit
        self.noise_fit = noise_fit
        
        return psf_fit, params
        
    def draw_comparison_2D(self, image, mask_fit, **kwargs):
        from plotting import draw_comparison_2D
        draw_comparison_2D(self.image_fit, image, mask_fit, self.image_star, self.noise_fit, **kwargs)
        
    def draw_background(self, save=False, dir_name='.', suffix=''):
        plt.figure()
        im = plt.imshow(self.bkg_fit); colorbar(im)
        if save:
            plt.savefig(os.path.join(dir_name,'Legendre2D%s.png'%(suffix)), dpi=80)
        else:
            plt.show()
            
def Run_Dynamic_Nested_Fitting(loglike, prior_transform, ndim,
                               nlive_init=100, sample='auto', 
                               nlive_batch=50, maxbatch=2,
                               pfrac=0.8, n_cpu=None, print_progress=True):
    
    print("Run Nested Fitting for the image... #a of params: %d"%ndim)
    
    start = time.time()
    
    if n_cpu is None:
        n_cpu = mp.cpu_count()-1
        
    with mp.Pool(processes=n_cpu) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu))
        pool.size = n_cpu

        dlogz = 1e-3 * (nlive_init - 1) + 0.01

        pdsampler = dynesty.DynamicNestedSampler(loglike,
                                                 prior_transform, ndim,
                                                 sample=sample,
                                                 pool=pool, use_pool={'update_bound': False})
        pdsampler.run_nested(nlive_init=nlive_init, 
                             nlive_batch=nlive_batch, 
                             maxbatch=maxbatch,
                             print_progress=print_progress, 
                             dlogz_init=dlogz, 
                             wt_kwargs={'pfrac': pfrac})
        
    end = time.time()
    print("Finish Fitting! Total time elapsed: %.3gs"%(end-start))
    
    return pdsampler

def merge_run(re_list):
    return dyfunc.merge_runs(res_list)


def get_params_fit(results, return_sample=False):
    samples = results.samples                                 # samples
    weights = np.exp(results.logwt - results.logz[-1])        # normalized weights 
    pmean, pcov = dyfunc.mean_and_cov(samples, weights)       # weighted mean and covariance
    samples_eq = dyfunc.resample_equal(samples, weights)      # resample weighted samples
    pmed = np.median(samples_eq,axis=0)
    
    if return_sample:
        return pmed, pmean, pcov, samples_eq
    else:
        return pmed, pmean, pcov
    
def make_psf_from_fit(fit_res, psf, n_out=4, theta_out=1200, leg2d=False):
    
    image_size = psf.image_size
    psf_fit = psf.copy()
    
    params, _, _ = get_params_fit(fit_res)
    
    if leg2d:
        N_n = (len(params)-4+1)//2
        N_theta = len(params)-4-N_n
        psf_fit.A10, psf_fit.A01 = 10**params[-3], 10**params[-4]
    else:
        N_n = (len(params)-2+1)//2
        N_theta = len(params)-2-N_n
    
    if psf.aureole_model == "power":
        n_fit, mu_fit, logsigma_fit = params
        psf_fit.update({'n':n_fit})
        
    elif psf.aureole_model == "multi-power":
        n_s_fit = np.concatenate([params[:N_n], [n_out]])
        theta_s_fit = np.concatenate([[psf.theta_0],
                                      np.atleast_1d(10**params[N_n:N_n+N_theta]),[theta_out]])
        psf_fit.update({'n_s':n_s_fit, 'theta_s':theta_s_fit})
        
    mu_fit, sigma_fit = params[-2], 10**params[-1]
    psf_fit.bkg, psf_fit.bkg_std  = mu_fit, sigma_fit
    
    _ = psf_fit.generate_core()
    _, _ = psf_fit.generate_aureole(psf_range=image_size)
    
    return psf_fit, params
    

def cal_reduced_chi2(fit, data, params):
    sigma = 10**params[-1]
    chi2_reduced = np.sum((fit-data)**2/sigma**2)/(len(data)-len(params))
    print("Reduced Chi^2: %.5f"%chi2_reduced)

def save_nested_fitting_result(res, filename='fit.res'):
    import dill
    with open(filename,'wb') as file:
        dill.dump(res, file)
        
def load_nested_fitting_result(filename='fit.res'):        
    import dill
    with open(filename, "rb") as file:
        res = dill.load(file)
    return res


class MyError(Exception): 
    def __init__(self, message):  self.message = message 
    def __str__(self): return(repr(self.message))
    def __repr__(self): return 'MyError(%r)'%(str(self))

class InconvergenceError(MyError): 
    def __init__(self, message):  self.message = message 
    def __repr__(self):
        return 'InconvergenceError: %r'%self.message
    
# # From TurbuStat
# def make_extended_ISM(imsize, powerlaw=2.0, theta=0., ellip=1.,
#                       return_fft=False, full_fft=True, randomseed=32768324):
#     '''
#     Generate a 2D power-law image with a specified index and random phases.
#     Adapted from https://github.com/keflavich/image_registration. Added ability
#     to make the power spectra elliptical. Also changed the random sampling so
#     the random phases are Hermitian (and the inverse FFT gives a real-valued
#     image).
#     Parameters
#     ----------
#     imsize : int
#         Array size.
#     powerlaw : float, optional
#         Powerlaw index.
#     theta : float, optional
#         Position angle of major axis in radians. Has no effect when ellip==1.
#     ellip : float, optional
#         Ratio of the minor to major axis. Must be > 0 and <= 1. Defaults to
#         the circular case (ellip=1).
#     return_fft : bool, optional
#         Return the FFT instead of the image. The full FFT is
#         returned, including the redundant negative phase phases for the RFFT.
#     full_fft : bool, optional
#         When `return_fft=True`, the full FFT, with negative frequencies, will
#         be returned. If `full_fft=False`, the RFFT is returned.
#     randomseed: int, optional
#         Seed for random number generator.
#     Returns
#     -------
#     newmap : np.ndarray
#         Two-dimensional array with the given power-law properties.
#     full_powermap : np.ndarray
#         The 2D array in Fourier space. The zero-frequency is shifted to
#         the centre.
#     '''
#     imsize = int(imsize)

#     if ellip > 1 or ellip <= 0:
#         raise ValueError("ellip must be > 0 and <= 1.")

#     yy, xx = np.meshgrid(np.fft.fftfreq(imsize),
#                          np.fft.rfftfreq(imsize), indexing="ij")

#     if ellip < 1:
#         # Apply a rotation and scale the x-axis (ellip).
#         costheta = np.cos(theta)
#         sintheta = np.sin(theta)

#         xprime = ellip * (xx * costheta - yy * sintheta)
#         yprime = xx * sintheta + yy * costheta

#         rr2 = xprime**2 + yprime**2

#         rr = rr2**0.5
#     else:
#         # Circular whenever ellip == 1
#         rr = (xx**2 + yy**2)**0.5

#     # flag out the bad point to avoid warnings
#     rr[rr == 0] = np.nan
    
#     from astropy.utils import NumpyRNGContext
#     with NumpyRNGContext(randomseed):

#         Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

#         angles = np.random.uniform(0, 2 * np.pi,
#                                    size=(imsize, Np1 + 1))

#     phases = np.cos(angles) + 1j * np.sin(angles)

#     # Rescale phases to an amplitude of unity
#     phases /= np.sqrt(np.sum(phases**2) / float(phases.size))

#     output = (rr**(-powerlaw / 2.)).astype('complex') * phases

#     output[np.isnan(output)] = 0.

#     # Impose symmetry
#     # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
#     if imsize % 2 == 0:
#         output[1:Np1, 0] = np.conj(output[imsize:Np1:-1, 0])
#         output[1:Np1, -1] = np.conj(output[imsize:Np1:-1, -1])
#         output[Np1, 0] = output[Np1, 0].real + 1j * 0.0
#         output[Np1, -1] = output[Np1, -1].real + 1j * 0.0

#     else:
#         output[1:Np1 + 1, 0] = np.conj(output[imsize:Np1:-1, 0])
#         output[1:Np1 + 1, -1] = np.conj(output[imsize:Np1:-1, -1])

#     # Zero freq components must have no imaginary part to be own conjugate
#     output[0, -1] = output[0, -1].real + 1j * 0.0
#     output[0, 0] = output[0, 0].real + 1j * 0.0

#     if return_fft:

#         if not full_fft:
#             return output

#         # Create the full power map, with the symmetric conjugate component
#         if imsize % 2 == 0:
#             power_map_symm = np.conj(output[:, -2:0:-1])
#         else:
#             power_map_symm = np.conj(output[:, -1:0:-1])

#         power_map_symm[1::, :] = power_map_symm[:0:-1, :]

#         full_powermap = np.concatenate((output, power_map_symm), axis=1)

#         if not full_powermap.shape[1] == imsize:
#             raise ValueError("The full output should have a square shape."
#                              " Instead has {}".format(full_powermap.shape))

#         return np.fft.fftshift(full_powermap)

#     newmap = np.fft.irfft2(output)

#     return newmap

