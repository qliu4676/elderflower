import sys
import os
import re
import math
import numpy as np
from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy import stats
from skimage import morphology

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, Column
from astropy.modeling import models
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, median_absolute_deviation, gaussian_fwhm_to_sigma
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch
norm0 = ImageNormalize(stretch=AsinhStretch())
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())

from photutils import detect_sources, deblend_sources
from photutils import CircularAperture, CircularAnnulus

### Baisc Funcs ###

def coord_Im2Array(X_IMAGE, Y_IMAGE, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE)-origin, 0)), int(max(round(X_IMAGE)-origin, 0))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    X_IMAGE, Y_IMAGE = y_arr+origin, x_arr+origin
    return X_IMAGE, Y_IMAGE


def Intensity2SB(y, BKG, ZP, pix_scale):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
#     y = np.atleast_1d(y)
#     y[y<BKG] = np.nan
    I_SB = -2.5*np.log10(y - BKG) + ZP + 2.5 * np.log10(pix_scale**2)
    return I_SB

def SB2Intensity(I_SB, BKG, ZP, pix_scale):
    """ Convert surface brightness (mag/arcsec^2)to intensity given the background value, zero point and pixel scale """ 
    y = 10** ((I_SB - ZP - 2.5 * np.log10(pix_scale**2))/ (-2.5)) + BKG
    return y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def process_counter(i, number):
    if np.mod((i+1), number//5) == 0:
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
    norm1 = ImageNormalize(stretch=LogStretch())
    norm2 = ImageNormalize(stretch=LogStretch())
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(14,4))
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
    mask_deep = np.zeros_like(image).astype(bool)
    
    if np.ndim(r_core) == 0:
        r_core = np.ones(len(star_pos)) * r_core
    
    core_region= np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r for (pos,r) in zip(star_pos,r_core)])
    
    mask_deep[core_region] = 1
    segmap = mask_deep.astype(int).copy()
    
    return mask_deep, segmap

def make_mask_map_dual(image, star_pos, r_core=24, sn_thre=2, 
                       nlevels=64, contrast=0.001, npix=4, 
                       b_size=25, n_dilation=1):
    """ Make mask map in dual mode: for faint stars, mask with S/N > sn_thre; for bright stars, mask core (r < r_core pix) """
    from photutils import detect_sources, deblend_sources
    yy, xx = np.indices(image.shape)

    # detect all source first 
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=npix)
    
    # deblend source
    segm_deb = deblend_sources(image, segm0, npixels=npix,
                               nlevels=nlevels, contrast=contrast)
    segmap = segm_deb.data.copy()
#     for pos in star_pos:
#         if (min(pos[0],pos[1]) > 0) & (pos[0] < image.shape[0]) & (pos[1] < image.shape[1]):
#             star_lab = segmap[coord_Im2Array(pos[0], pos[1])]
#             segm_deb.remove_label(star_lab)
    
    segmap2 = segm_deb.data.copy()
    
    # remove S/N mask map for input (bright) stars
    for pos in star_pos:
        rr = (xx-pos[0])**2+(yy-pos[1])**2
        lab = segmap2[np.where(rr==np.min(rr))][0]
        segmap2[segmap2==lab] = 0
    
    # dilation
    for i in range(n_dilation):
        segmap2 = morphology.dilation(segmap2)

    # mask core for input (bright) stars
    if np.ndim(r_core) == 0:
        r_core = np.ones(len(star_pos)) * r_core
    core_region= np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r for (pos,r) in zip(star_pos,r_core)])
    
    segmap2[core_region] = segmap.max()+1
    
    # set dilation border a different label (for visual)
    segmap2[(segmap2!=0)&(segm_deb.data==0)] = segmap.max()+2
    
    # set mask map
    mask_deep = (segmap2!=0)
    
    return mask_deep, segmap2, core_region

def make_mask_strip(image_size, star_pos, fluxs, width=5, n_strip=12, dist_strip=300):    
    """ Make mask map in strips with width=width """
    yy, xx = np.mgrid[:image_size, :image_size]
    phi_s = np.linspace(-90, 90, n_strip+1)
    phi_s = np.setdiff1d(phi_s, [-90,0,90])
    a_s = np.tan(phi_s*np.pi/180)
    
    mask_strip_s = np.empty((len(star_pos), image_size, image_size))
    mask_cross_s = np.empty_like(mask_strip_s)
    
    for k, (x_b, y_b) in enumerate(star_pos[fluxs.argsort()]):
        m_s = (y_b-a_s*x_b)
        mask_strip = np.logical_or.reduce([abs((yy-a*xx-m)/math.sqrt(1+a**2)) < width 
                                           for (a, m) in zip(a_s, m_s)])
        mask_cross = np.logical_or.reduce([abs(yy-y_b)< width, abs(xx-x_b)< width])
        dist_map1 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_strip
        dist_map2 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_strip//2
        mask_strip_s[k] = mask_strip & dist_map1
        mask_cross_s[k] = mask_cross & dist_map2

    return mask_strip_s, mask_cross_s

    
        
def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                   color="steelblue", xunit="pix", yunit="intensity",
                   seeing=2.5, pix_scale=2.5, ZP=27.1, 
                   sky_mean=884, sky_std=3, dr=1, 
                   lw=2, alpha=0.7, markersize=5, I_shift=0,
                   core_undersample=False, label=None, plot_line=False,
                   draw=True, scatter=False, verbose=False):
    """Calculate 1d radial profile of a given star postage"""
    if mask is None:
        mask =  np.zeros_like(img).astype("bool")
    if back is None:     
        back = np.ones_like(img) * sky_mean
    if cen is None:
        cen = (img.shape[0]-1)/2., (img.shape[1]-1)/2.
        
    yy, xx = np.indices((img.shape))
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int(3 * seeing/pix_scale) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * np.median(back)
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.0001 * z_diff[-1]))
    r_max = np.sqrt(n_pix_max/np.pi)
    r_max = np.min([img.shape[0]//2, r_max])
    if verbose:
        print("Maximum R: %d (pix)"%np.int(r_max))    
    
    if xunit == "arcsec":
        r = r * pix_scale   # radius in arcsec
        r_core = r_core * pix_scale
        r_max = r_max * pix_scale
        
    d_r = dr * pix_scale if xunit == "arcsec" else dr
    
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
    logzerr_rbin = np.array([])
    for k,b in enumerate(bins[:-1]):
        in_bin = (r>bins[k])&(r<bins[k+1])
        r_rbin = np.append(r_rbin, np.mean(r[in_bin]))
      
        z_clip = sigma_clip(z[in_bin], sigma=3, maxiters=10)
        zb = np.mean(z_clip)
        sigma_zb = np.std(z_clip)
        z_rbin = np.append(z_rbin, zb)
        logzerr_rbin = np.append(logzerr_rbin, 0.434 * ( sigma_zb / zb))

    if draw:
        if yunit == "intensity":  
            # plot radius in Intensity
            plt.plot(r_rbin, np.log10(z_rbin), "-o", mec="k", lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label) 
            if scatter:
                plt.scatter(r[r<3*r_core], np.log10(z[r<3*r_core]), color=color, s=6, alpha=0.2, zorder=1)
                plt.scatter(r[r>3*r_core], np.log10(z[r>3*r_core]), color=color, s=3, alpha=0.1, zorder=1)
            plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                             color=color, alpha=0.2, zorder=1)
            plt.ylabel("log Intensity")
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8, r_rbin[np.isfinite(r_rbin)][-1]*1.2)

        elif yunit == "SB":  
            # plot radius in Surface Brightness
            I_rbin = Intensity2SB(y=z_rbin, BKG=np.median(back), ZP=ZP, pix_scale=pix_scale) + I_shift
            I_sky = Intensity2SB(y=sky_std, BKG=0, ZP=ZP, pix_scale=pix_scale)

            plt.plot(r_rbin, I_rbin, "-o", mec="k", lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label)   
            if scatter:
                I = Intensity2SB(y=z, BKG=np.median(back), ZP=ZP, pix_scale=pix_scale) + I_shift
                plt.scatter(r[r<3*r_core], I[r<3*r_core], color=color, s=6, alpha=0.2, zorder=1)
                plt.scatter(r[r>3*r_core], I[r>3*r_core], color=color, s=3, alpha=0.1, zorder=1)
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

            use_range = (r_rbin>r_satr) & (r_rbin<r_core)
        else:
            use_range = True
            
    return r_rbin, z_rbin, logzerr_rbin


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
        print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
    
    # crop
    img_thumb = data[x_min:x_max, y_min:y_max].copy()
    seg_thumb = seg_map[x_min:x_max, y_min:y_max]
    mask_thumb = (seg_thumb!=0)    
    
    # the center position is converted from world with wcs
    X_cen, Y_cen = wcs.wcs_world2pix(star_cat[id]["X_WORLD"], star_cat[id]["Y_WORLD"], origin)
    cen_star = X_cen - X_min, Y_cen - Y_min
    
    return (img_thumb, seg_thumb, mask_thumb), cen_star
    
def extract_star(id, star_cat, wcs, data, seg_map, 
                 seeing=2.5, sn_thre=2.5, n_win=20, 
                 display_bg=False, display=True, verbose=False):
    
    """ Return the image thubnail, mask map, backgroud estimates, and center of star.
        Do a finer detection&deblending to remove faint undetected source."""
    
    thumb_list, cen_star = get_star_thumb(id, star_cat, wcs, data, seg_map,
                                          n_win=n_win, seeing=seeing, verbose=verbose)
    img_thumb, seg_thumb, mask_thumb = thumb_list
    
    # the same thumbnail size
    fwhm = max([star_cat[id]["FWHM_IMAGE"], seeing])
    
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
            
    # do segmentation (a second time) to remove faint undetected stars using photutils
    sigma = seeing * gaussian_fwhm_to_sigma
    threshold = back + (sn_thre * back_rms)
    segm = detect_sources(img_thumb, threshold, npixels=5)
    
    # do deblending using photutils
    segm_deblend = deblend_sources(img_thumb, segm, npixels=5,
                                   nlevels=64, contrast=0.005)
    
    # the target star is at the center of the thumbnail
    star_lab = segm_deblend.data[img_thumb.shape[0]//2, img_thumb.shape[1]//2]
    star_ma = ~((segm_deblend.data==star_lab) | (segm_deblend.data==0)) # mask other source
    
    if display:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1,ncols=4,figsize=(21,5))
        ax1.imshow(img_thumb, vmin=np.median(back)-1, vmax=10000, norm=norm1, cmap="viridis")
        ax1.set_title("star", fontsize=15)
        ax2.imshow(segm, cmap=segm.make_cmap(random_state=12345))
        ax2.set_title("segment", fontsize=15)
        ax3.imshow(segm_deblend, cmap=segm_deblend.make_cmap(random_state=12345))
        ax3.set_title("deblend", fontsize=15)

        img_thumb_ma = img_thumb.copy()
        img_thumb_ma[star_ma] = -1
        ax4.imshow(img_thumb_ma, cmap="viridis", norm=norm2,
                   vmin=np.median(back)-1, vmax=np.median(back)+10*np.median(back_rms))
        ax4.set_title("extracted star", fontsize=15)
    
    return img_thumb, star_ma, back, cen_star


def compute_Rnorm(image, mask_field, cen, R=10, wid=0.5, mask_cross=False, display=False):
    """ Return 3 sigma-clipped mean, med and std of ring r=R (half-width=wid) for image.
        Note intensity is not background subtracted. """
    
    annulus_ma = CircularAnnulus(cen, R-wid, R+wid).to_mask()      
    mask_ring = annulus_ma.to_image(image.shape) > 0.5    # sky ring (R-wid, R+wid)
    mask_clean = mask_ring & (~mask_field)                # sky ring with other sources masked
    
    # Whether to mask the cross regions, important if R is small
    if mask_cross:
        yy, xx = np.indices(image.shape)
        rr = np.sqrt((xx-cen[0])**2+(yy-cen[1])**2)
        cross = ((abs(xx-cen[0])<1.)|(abs(yy-cen[1])<1.))
        mask_clean = mask_clean * (~cross)
        
    I_mean, I_med, I_std = sigma_clipped_stats(image[mask_clean], sigma=3)
    
    if display:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        ax1.imshow(mask_clean * image, cmap="gray", norm=norm1, vmin=I_med-5*I_std, vmax=I_med+5*I_std)
        ax2 = plt.hist(sigma_clip(image[mask_clean]))
    
    return I_mean, I_med, I_std

def compute_Rnorm_batch(df_target, SE_catalog, wcs, data, seg_map, 
                        R=10, wid=0.5, return_full=False):
    """ Combining the above functions. Compute for all object in df_target.
        Return an arry with measurement on the intensity and a dictionary containing maps and centers."""
    
    # Initialize
    res_thumb = {}    
    res_Rnorm = np.empty((len(df_target),4))
    
    for i, (num, rmag_auto) in enumerate(zip(df_target['NUMBER'], df_target['RMAG_AUTO'])):
        process_counter(i, len(df_target))
        ind = num - 1
        
        # For very bright sources, use a broader window
        n_win = 30 if rmag_auto < 11 else 20
            
        img, ma, bkg, cen = extract_star(ind, SE_catalog, wcs, data, seg_map,
                                         n_win=n_win, display_bg=False, display=False)
        
        res_thumb[num] = {"image":img, "mask":ma, "bkg":bkg, "center":cen}
        
        # Measure the mean, med and std of intensity at R
        I_mean, I_med, I_std = compute_Rnorm(img, ma, cen, R=R, wid=wid)
        
        # Use the median value of background as the local background
        sky_mean = np.median(bkg)
        
        res_Rnorm[i] = np.array([I_mean, I_med, I_std, sky_mean])
    
    return res_Rnorm, res_thumb


### Catalog / Data Manipulation Helper ###

def check_save_path(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if len(os.listdir(dir_name)) != 0:
            while os.path.exists(dir_name):
                dir_name = input("'%s' already existed. Enter a directory name for saving:"%dir_name)
            os.makedirs(dir_name)
    print("Results will be saved in %s"%dir_name)
    

def crop_catalog(cat, bounds, keys=("X_IMAGE", "Y_IMAGE")):
    Xmin, Ymin, Xmax, Ymax = bounds
    A, B = keys
    crop = (cat[A]>=Xmin) & (cat[A]<=Xmax) & (cat[B]>=Ymin) & (cat[B]<=Ymax)
    return cat[crop]

def crop_image(data, bounds, SE_seg_map=None, weight_map=None,
               sky_mean=0, sky_std=1, origin=1, color="r", draw=False):
    from matplotlib import patches  
    patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = bounds
    patch_xmin, patch_ymin = coord_Im2Array(patch_Xmin, patch_Ymin, origin)
    patch_xmax, patch_ymax = coord_Im2Array(patch_Xmax, patch_Ymax, origin)

    patch = np.copy(data[patch_xmin:patch_xmax, patch_ymin:patch_ymax])
    if SE_seg_map is None:
        seg_patch = None
    else:
        seg_patch = np.copy(SE_seg_map[patch_xmin:patch_xmax, patch_ymin:patch_ymax])
    
    if draw:
        fig, ax = plt.subplots(figsize=(12,8))       
        plt.imshow(data, norm=norm1, cmap="viridis",
                   vmin=sky_mean, vmax=sky_mean+10*sky_std, alpha=0.95)
        if weight_map is not None:
            plt.imshow(data*weight_map, norm=norm1, cmap="viridis",
                       vmin=sky_mean, vmax=sky_mean+10*sky_std, alpha=0.3)
        plt.plot([600,840],[650,650],"w",lw=4)
        plt.text(560,400, r"$\bf 10'$",color='w', fontsize=20)
        rect = patches.Rectangle((patch_Xmin, patch_Ymin), bounds[2]-bounds[0], bounds[3]-bounds[1],
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        plt.show()
        
    return patch, seg_patch

def query_vizier(catalog_name, radius, columns, column_filters, header):
    """ Query catalog in Vizier database with the given catalog name, search radius and column names """
    from astroquery.vizier import Vizier
    from astropy import units as u
    
    # Prepare for quearyinig Vizier with filters up to infinitely many rows. By default, this is 50.
    viz_filt = Vizier(columns=columns, column_filters=column_filters)
    viz_filt.ROW_LIMIT = -1

    RA, DEC = re.split(",", header['RADEC'])
    coords = SkyCoord(RA+" "+DEC , unit=(u.hourangle, u.deg))

    # Query!
    result = viz_filt.query_region(coords, 
                                   radius=radius, 
                                   catalog=[catalog_name])
    return result

def transform_coords2pixel(table, wcs, cat_name='', RA_key="RAJ2000", DE_key="DEJ2000", origin=1):
    """ Transform the RA/DEC columns in the table into pixel coordinates given wcs"""
    coords = np.vstack([np.array(table[RA_key]), 
                        np.array(table[DE_key])]).T
    pos = wcs.wcs_world2pix(coords, origin)
    table.add_column(Column(np.around(pos[:,0], 4)*u.pix), name='X_IMAGE'+'_'+cat_name)
    table.add_column(Column(np.around(pos[:,1], 4)*u.pix), name='Y_IMAGE'+'_'+cat_name)
    table.add_column(Column(np.arange(len(table))+1), index=0, name="ID"+'_'+cat_name)
    return table

def merge_catalog(SE_catalog, table_merge, sep=2.5 * u.arcsec,
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
    
    df_match = cat_match.to_pandas()
    return df_match

def save_thumbs(obj, filename):
    import pickle
    print("Save thumbs to: %s"%filename)
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_thumbs(filename):
    import pickle
    print("Read thumbs from: %s"%filename) 
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


### Nested Fitting Helper ###

def Run_Dynamic_Nested_Fitting(loglike, prior_transform, ndim,
                               nlive_init=100, sample='auto', 
                               nlive_batch=50, maxbatch=4,
                               pfrac=0.8, n_cpu=None, print_progress=True):
    
    """ Run dynamic nested fitting. """
    
    import dynesty
    import multiprocess as mp
    
    if n_cpu is None:
        n_cpu = mp.cpu_count()-1
        
    print("Run Nested Fitting for the image... Dim of paramas: %d"%ndim)    
    
    with mp.Pool(processes=n_cpu) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu))
        pool.size = 3

        dlogz = 1e-3 * (nlive_init - 1) + 0.01

        pdsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, sample=sample,
                                                 pool=pool, use_pool={'update_bound': False})
        pdsampler.run_nested(nlive_init=nlive_init, 
                             nlive_batch=nlive_batch, 
                             maxbatch=maxbatch,
                             print_progress=print_progress, 
                             dlogz_init=dlogz, 
                             wt_kwargs={'pfrac': pfrac})
    return pdsampler


def Run_2lin_Nested_Fitting(X, Y, priors, display=True):
    
    """ Run piece-wise linear fitting for [X, Y]. 
        Priors need to be given in [loc, loc+scale] by a dictionary {"loc"/"scale":[x0,y0,k1,k2,sigma]} """
    
    loc, scale = priors["loc"], priors["scale"]
    
    def prior_tf_2lin(u):
        v = u.copy()
        v[0] = u[0] * scale[0] + loc[0]  #x0
        v[1] = u[1] * scale[1] + loc[1]  #y0
        v[2] = u[2] * scale[2] + loc[2]  #k1
        v[3] = u[3] * scale[3] + loc[3]  #k2
        v[4] = u[4] * scale[4] + loc[4]  #sigma
        return v

    def loglike_2lin(v):
        x0, y0, k2, sigma = v
        ypred = np.piecewise(X, [X < x0], [lambda x: k1*x + y0-k1*x0, lambda x: k2*x + y0-k2*x0])
        residsq = (ypred - Y)**2 / sigma**2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
        if not np.isfinite(loglike):
            loglike = -1e100
        return loglike
    
    pdsampler = Run_Dynamic_Nested_Fitting(loglike=loglike_2lin, prior_transform=prior_tf_2lin, ndim=5)
    pdres = pdsampler.results
    
    if display:
        labels = ["x0", "y0", "k1", "k2", "sigma"]
        fig, axes = dyplot.cornerplot(pdres, show_titles=True, 
                                      color="royalblue", labels=labels,
                                      title_kwargs={'fontsize':15, 'y': 1.02}, 
                                      label_kwargs={'fontsize':12},
                                      fig=plt.subplots(5, 5, figsize=(11, 10)))
    return pdres


def get_params_fit(res):
    from dynesty import utils as dyfunc
    samples = res.samples                                 # samples
    weights = np.exp(res.logwt - res.logz[-1])            # normalized weights 
    pmean, pcov = dyfunc.mean_and_cov(samples, weights)     # weighted mean and covariance
    samples_eq = dyfunc.resample_equal(samples, weights)  # resample weighted samples
    pmed = np.median(samples_eq,axis=0)
    return pmed, pmean, pcov




def save_nested_fitting_result(res, filename='fit.res'):
    import dill
    with open(filename,'wb') as file:
        dill.dump(res, file)
        
def open_nested_fitting_result(filename='fit.res'):        
    import dill
    with open(filename, "rb") as file:
        res = dill.load(file)
    return res

from matplotlib import rcParams
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'xtick.major.pad': '5.0'})
rcParams.update({'xtick.major.size': '4'})
rcParams.update({'xtick.major.width': '1.'})
rcParams.update({'xtick.minor.pad': '5.0'})
rcParams.update({'xtick.minor.size': '4'})
rcParams.update({'xtick.minor.width': '0.8'})
rcParams.update({'ytick.major.pad': '5.0'})
rcParams.update({'ytick.major.size': '4'})
rcParams.update({'ytick.major.width': '1.'})
rcParams.update({'ytick.minor.pad': '5.0'})
rcParams.update({'ytick.minor.size': '4'})
rcParams.update({'ytick.minor.width': '0.8'})
rcParams.update({'font.size': 14})


# From TurbuStat
def make_extended(imsize, powerlaw=2.0, theta=0., ellip=1.,
                  return_fft=False, full_fft=True, randomseed=32768324):
    '''
    Generate a 2D power-law image with a specified index and random phases.
    Adapted from https://github.com/keflavich/image_registration. Added ability
    to make the power spectra elliptical. Also changed the random sampling so
    the random phases are Hermitian (and the inverse FFT gives a real-valued
    image).
    Parameters
    ----------
    imsize : int
        Array size.
    powerlaw : float, optional
        Powerlaw index.
    theta : float, optional
        Position angle of major axis in radians. Has no effect when ellip==1.
    ellip : float, optional
        Ratio of the minor to major axis. Must be > 0 and <= 1. Defaults to
        the circular case (ellip=1).
    return_fft : bool, optional
        Return the FFT instead of the image. The full FFT is
        returned, including the redundant negative phase phases for the RFFT.
    full_fft : bool, optional
        When `return_fft=True`, the full FFT, with negative frequencies, will
        be returned. If `full_fft=False`, the RFFT is returned.
    randomseed: int, optional
        Seed for random number generator.
    Returns
    -------
    newmap : np.ndarray
        Two-dimensional array with the given power-law properties.
    full_powermap : np.ndarray
        The 2D array in Fourier space. The zero-frequency is shifted to
        the centre.
    '''
    imsize = int(imsize)

    if ellip > 1 or ellip <= 0:
        raise ValueError("ellip must be > 0 and <= 1.")

    yy, xx = np.meshgrid(np.fft.fftfreq(imsize),
                         np.fft.rfftfreq(imsize), indexing="ij")

    if ellip < 1:
        # Apply a rotation and scale the x-axis (ellip).
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        xprime = ellip * (xx * costheta - yy * sintheta)
        yprime = xx * sintheta + yy * costheta

        rr2 = xprime**2 + yprime**2

        rr = rr2**0.5
    else:
        # Circular whenever ellip == 1
        rr = (xx**2 + yy**2)**0.5

    # flag out the bad point to avoid warnings
    rr[rr == 0] = np.nan
    
    from astropy.utils import NumpyRNGContext
    with NumpyRNGContext(randomseed):

        Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=(imsize, Np1 + 1))

    phases = np.cos(angles) + 1j * np.sin(angles)

    # Rescale phases to an amplitude of unity
    phases /= np.sqrt(np.sum(phases**2) / float(phases.size))

    output = (rr**(-powerlaw / 2.)).astype('complex') * phases

    output[np.isnan(output)] = 0.

    # Impose symmetry
    # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
    if imsize % 2 == 0:
        output[1:Np1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1, -1] = np.conj(output[imsize:Np1:-1, -1])
        output[Np1, 0] = output[Np1, 0].real + 1j * 0.0
        output[Np1, -1] = output[Np1, -1].real + 1j * 0.0

    else:
        output[1:Np1 + 1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1 + 1, -1] = np.conj(output[imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    output[0, -1] = output[0, -1].real + 1j * 0.0
    output[0, 0] = output[0, 0].real + 1j * 0.0

    if return_fft:

        if not full_fft:
            return output

        # Create the full power map, with the symmetric conjugate component
        if imsize % 2 == 0:
            power_map_symm = np.conj(output[:, -2:0:-1])
        else:
            power_map_symm = np.conj(output[:, -1:0:-1])

        power_map_symm[1::, :] = power_map_symm[:0:-1, :]

        full_powermap = np.concatenate((output, power_map_symm), axis=1)

        if not full_powermap.shape[1] == imsize:
            raise ValueError("The full output should have a square shape."
                             " Instead has {}".format(full_powermap.shape))

        return np.fft.fftshift(full_powermap)

    newmap = np.fft.irfft2(output)

    return newmap

