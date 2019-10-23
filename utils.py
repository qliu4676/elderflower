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
from astropy.visualization import LogStretch, SqrtStretch
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

### funcs on single element ###

def trunc_pow(x, n, theta0, I_theta0=1):
    """ Truncated power law for single element, I = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * x**(-n) if x > theta0 else I_theta0
    return y

def compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0):
    """ Compute normalization factor of each power law component """
    a_s = np.zeros(len(n_s))
    a0 = I_theta0 * theta0**(n0)

    I_theta_i = a0 * float(theta_s[0])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        try:
            a_s[i] = a_i
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass    
    return a0, a_s

def compute_multi_pow_norm(n_s, theta_s, I_theta0):
    """ Compute normalization factor of each power law component """
    n0, theta0 = n_s[0], theta_s[0]
    a0 = I_theta0 * theta0**(n0)
    a_s = np.zeros(len(n_s))   
    a_s[0] = a0
    
    I_theta_i = a0 * float(theta_s[1])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s[1:], theta_s[1:])):
        a_i = I_theta_i/(theta_s[i+1])**(-n_i)
        try:
            a_s[i+1] = a_i
            I_theta_i = a_i * float(theta_s[i+2])**(-n_i)
        except IndexError:
            pass    
    return a_s

def multi_pow0(x, n0, n_s, theta0, theta_s, I_theta0, a0=None, a_s=None):
    """ Continuous multi-power law for single element """
    if a0 is None:
        a0, a_s = compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0)
        
    if x <= theta0:
        return I_theta0
    elif x<= theta_s[0]:
        y = a0 * x**(-n0)
        return y
    else:
        for k in range(len(a_s-1)):
            try:
                if x <= theta_s[k+1]:
                    y = a_s[k] * x**(-n_s[k])
                    return y
            except IndexError:
                pass
        else:
            y = a_s[k] * x**(-n_s[k])
            return y
        
def multi_pow(x, n_s, theta_s, I_theta0, a_s=None):
    """ Continuous multi-power law for single element """
    
    if a_s is None:
        a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
    
    if x <= theta0:
        return I_theta0
    elif x<= theta_s[1]:
        y = a0 * x**(-n0)
        return y
    else:
        for k in range(len(a_s)):
            try:
                if x <= theta_s[k+2]:
                    y = a_s[k+1] * x**(-n_s[k+1])
                    return y
            except IndexError:
                pass
        else:
            y = a_s[-1] * x**(-n_s[-1])
            return y

### 1D functions ###

def power1d(x, n, theta0, I_theta0):
    a = I_theta0/(theta0)**(-n)
    y = a * np.power(x, -n)
    return y

def trunc_power1d(x, n, theta0, I_theta0=1): 
    """ Truncated power law for 1d array, I = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x + 1e-6, -n) 
    y[x<=theta0] = I_theta0
    return y

def multi_power1d0(x, n0, theta0, I_theta0, n_s, theta_s):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a0, a_s = compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0)
    
    y = a0 * np.power(x + 1e-6, -n0)
    y[x<=theta0] = I_theta0
    for i, (n_i, a_i, theta_i) in enumerate(zip(n_s, a_s, theta_s)):
        y_i = a_i * np.power(x, -n_i)
        y[x>theta_i] = y_i[x>theta_i]
    return y

def multi_power1d(x, n_s, theta_s, I_theta0):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    theta0 = theta_s[0]
    
    y = np.zeros_like(x)
    y[x<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (x>theta_s[k]) & (x<=theta_s[k+1]) if k<len(a_s)-1 else (x>theta_s[k])  
        y[reg] = a_s[k] * np.power(x[reg], -n_s[k])
    return y


def moffat_power1d(x, gamma, alpha, n, theta0, A=1):
    """ Moffat + Power for 1d array, flux normalized = 1 """
    Mof_mod_1d = models.Moffat1D(amplitude=A, x_0=0, gamma=gamma, alpha=alpha)
    y = Mof_mod_1d(x)
    y[x>theta0] = power1d(x[x>theta0], n, theta0, Mof_mod_1d(theta0))
    return y

def trunc_power1d_normed(x, n, theta0):
    """ Truncated power law for 1d array, flux normalized = 1 """
    norm_pow = quad(trunc_pow, 0, np.inf, args=(n, theta0, 1))[0]
    y = trunc_power1d(x, n, theta0, 1) / norm_pow  
    return y

def moffat1d_normed(x, gamma, alpha):
    """ Truncated Moffat for 1d array, flux normalized = 1 """
    Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    norm_mof = quad(Mof_mod_1d, 0, np.inf)[0] 
    y = Mof_mod_1d(x) / norm_mof
    return y

def multi_power1d_normed(x, n_s, theta_s):
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    norm_mpow = quad(multi_pow, 0, np.inf, args=(n_s, theta_s, 1, a_s))[0]
    y = multi_power1d(x, n_s, theta_s, 1) / norm_mpow
    return y

### 2D functions ###

def map2d(f, xx=None, yy=None):
    return f(xx,yy)

def power2d(xx, yy, n, theta0, I_theta0, cen): 
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    return z 

def trunc_power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Truncated power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    z[rr<=theta0] = I_theta0
    return z

def multi_power2d_cover(xx, yy, n0, theta0, I_theta0, n_s, theta_s, cen):
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(rr, -n0) 
    z[rr<=theta0] = I_theta0
    
    I_theta_i = a0 * float(theta_s[0])**(-n0)
    
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(rr, -n_i)
        z[rr>theta_i] = z_i[rr>theta_i]
        try:
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass
    return z

def multi_power2d(xx, yy, n_s, theta_s, I_theta0, cen):
    """ Multi-power law for 2d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    
    r = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2).ravel()
    z = np.zeros(xx.size) 
    theta0 = theta_s[0]
    z[r<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (r>theta_s[k]) & (r<=theta_s[k+1]) if k<len(a_s)-1 else (r>theta_s[k])     
        z[reg] = a_s[k] * np.power(r[reg], -n_s[k])
        
    return z.reshape(xx.shape)


### Flux/Amplitude Convertion ###

def moffat1d_Flux2Amp(r_core, beta, Flux=1):
    """ Calculate the (astropy) amplitude of 1d Moffat profile given the core width, power index, and total flux F.
    Note in astropy unit (x,y) the amplitude should be scaled with 1/sqrt(pi)."""
    Amp = Flux * Gamma(beta) / ( r_core * np.sqrt(np.pi) * Gamma(beta-1./2) ) # Derived scaling factor
    return  Amp

def moffat1d_Amp2Flux(r_core, beta, Amp=1):
    """ Calculate the (astropy) amplitude of 1d Moffat profile given the core width, power index, and total flux F.
    Note in astropy unit (x,y) the amplitude should be scaled with 1/sqrt(pi)."""
    F = Amp *  r_core * np.sqrt(np.pi) * Gamma(beta-1./2) / Gamma(beta) # Derived scaling factor
    return  F

def power1d_Flux2Amp(n, theta0, Flux=1, trunc=True):
    if trunc:
        I_theta0 = Flux * (n-1)/n / theta0
    else:
        I_theta0 = Flux * (n-1) / theta0
    return I_theta0

def power1d_Amp2Flux(n, theta0, Amp=1, trunc=True):
    if trunc:
        Flux = Amp * n/(n-1) * theta0
    else:
        Flux = Amp * 1./(n-1) * theta0
    return Flux

def moffat2d_Flux2Amp(r_core, beta, Flux=1):
    return Flux * (beta-1) / r_core**2 / np.pi

def moffat2d_Amp2Flux(r_core, beta, Amp=1):
    return Amp / moffat2d_Flux2Amp(r_core, beta, Flux=1)

def power2d_Flux2Amp(n, theta0, Flux=1, trunc=True):
    if trunc:
        I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    else:
        I_theta0 = (1./np.pi) * Flux * (n-2)/2 / theta0**2
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1, trunc=True):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1, trunc=trunc)

def multi_power2d_Amp2Flux(n_s, theta_s, Amp=1):
    a_s = compute_multi_pow_norm(n_s, theta_s, Amp)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
    
    I_2D = Amp * np.pi * theta0**2
    for k in range(len(n_s)-1):
        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2) 
    I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2)   
    return I_2D

def multi_power2d_Flux2Amp(n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n_s, theta_s, Amp=1)


def Ir2Flux_tot(frac, n, theta0, r, Ir):
    """ Convert Intensity Ir at r to total flux with frac = fraction of power law """
    Flux_pow = power2d_Amp2Flux(n, theta0, Amp=Ir * (r/theta0)**n)
    Flux_tot = Flux_pow / frac
    return Flux_tot


### 1D/2D conversion factor ###

def C_mof2Dto1D(r_core, beta):
    """ gamma in pixel """
    return 1./(beta-1) * 2*math.sqrt(np.pi) * r_core * Gamma(beta) / Gamma(beta-1./2) 

def C_mof1Dto2D(r_core, beta):
    """ gamma in pixel """
    return 1. / C_mof2Dto1D(r_core, beta)

def C_pow2Dto1D(n, theta0):
    """ theta0 in pixel """
    return np.pi * theta0 * (n-1) / (n-2)

def C_pow1Dto2D(n, theta0):
    """ theta0 in pixel """
    return 1. / C_pow2Dto1D(n, theta0)

def C_mpow2Dto1D(n_s, theta_s):
    """ theta in pixel """
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
 
    I_2D = 1. * np.pi * theta0**2
    for k in range(len(n_s)-1):
        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2) 
    I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2)   
    
    I_1D = 1. * theta0
    for k in range(len(n_s)-1):
        if n_s[k] == 1:
            I_1D += a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_1D += a_s[k] * (theta_s[k]**(1-n_s[k]) - theta_s[k+1]**(1-n_s[k])) / (n_s[k]-1) 
    I_1D += a_s[-1] * theta_s[-1]**(1-n_s[-1]) / (n_s[-1]-1)
    
    return I_2D / I_1D 

def C_mpow1Dto2D(n_s, theta_s):
    """ theta in pixel """
    return 1. / C_mpow2Dto1D(n_s, theta_s)

    
### Plotting Helpers ###

def vmin_3mad(img):
    """ lower limit of visual imshow defined by 3 mad above median """ 
    return np.median(img)-3*mad_std(img)

def vmax_2sig(img):
    """ upper limit of visual imshow defined by 2 sigma above median """ 
    return np.median(img)+2*np.std(img)

def colorbar(mappable, pad=0.2, size="5%", loc="right", **args):
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
    ax1.imshow(field, origin="lower", aspect="auto", cmap="gray", vmin=vmin_3mad(field), vmax=vmax_2sig(field),norm=norm1)
    im2 = ax2.imshow(back, origin='lower', aspect="auto", cmap='gray')
    colorbar(im2)
    ax3.imshow(field - back, origin='lower', aspect="auto", cmap='gray', vmin=0., vmax=vmax_2sig(field - back),norm=norm2)
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

def make_mask_map_dual(image, star_pos, r_in=24, sn_thre=2, 
                       nlevels=64, contrast=0.001, npix=4, 
                       b_size=25, n_dilation=1):
    """ Make mask map in dual mode: for faint stars, mask with S/N > sn_thre; for bright stars, mask core (r < r_in pix) """
    from photutils import detect_sources, deblend_sources
    
    # detect source
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=npix)
    
    # deblend
    segm_deb = deblend_sources(image, segm0, npixels=npix,
                               nlevels=nlevels, contrast=contrast)
    segmap = segm_deb.data.copy()
    for pos in star_pos:
        if (min(pos[0],pos[1]) > 0) & (pos[0] < image.shape[0]) & (pos[1] < image.shape[1]):
            star_lab = segmap[coord_Im2Array(pos[0], pos[1])]
            segm_deb.remove_label(star_lab)
    
    # dilation
    segmap2 = segm_deb.data.copy()
    for i in range(n_dilation):
        segmap2 = morphology.dilation(segmap2)
    
    # set dilation border a different label (for visual)
    segmap2[(segmap2!=0)&(segm_deb.data==0)] = segmap.max()+1
    
    # mask core
    yy, xx = np.indices(image.shape)
    if np.ndim(r_in) == 0:
        r_in = np.ones(len(star_pos)) * r_in
    core_region= np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r for (pos,r) in zip(star_pos,r_in)])
    segmap2[core_region] = segmap.max()+1
    
    mask_deep = (segmap2!=0)
    
    return mask_deep, segmap2, core_region

def make_mask_strip(image_size, star_pos, fluxs, width=5, n_strip=12, dist_strip=480):    
    """ Make mask map in strips with width=width """
    yy, xx = np.mgrid[:image_size, :image_size]
    phi_s = np.linspace(-90, 90, n_strip+1)
    a_s = np.tan(phi_s*np.pi/180)
    mask_strip_s = np.empty((len(star_pos), image_size, image_size))
    
    for k, (x_b, y_b) in enumerate(star_pos[fluxs.argsort()]):
        m_s = (y_b-a_s*x_b)
        mask_strip = np.logical_or.reduce([abs((yy-a*xx-m)/math.sqrt(1+a**2)) < width 
                                           for (a, m) in zip(a_s, m_s)])
        dist_map = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_strip
        mask_strip_s[k] = mask_strip & dist_map

    return mask_strip_s


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
        ax1.imshow(img_thumb, vmin=np.median(back)-1, vmax=10000, norm=norm1, origin="lower", cmap="viridis")
        ax1.set_title("star", fontsize=15)
        ax2.imshow(segm, origin="lower", cmap=segm.make_cmap(random_state=12345))
        ax2.set_title("segment", fontsize=15)
        ax3.imshow(segm_deblend, origin="lower", cmap=segm_deblend.make_cmap(random_state=12345))
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

def crop_image(data, SE_seg_map, bounds, sky_mean=0, sky_std=1, origin=1, weight_map=None, color="r", draw=False):
    from matplotlib import patches  
    patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = bounds
    patch_xmin, patch_ymin = coord_Im2Array(patch_Xmin, patch_Ymin, origin)
    patch_xmax, patch_ymax = coord_Im2Array(patch_Xmax, patch_Ymax, origin)

    patch = np.copy(data[patch_xmin:patch_xmax, patch_ymin:patch_ymax])
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


