import sys
import os
import math
import numpy as np
from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
from astropy.stats import mad_std, median_absolute_deviation, gaussian_fwhm_to_sigma
from astropy.stats import sigma_clip, SigmaClip
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())

from photutils import detect_sources

from skimage import morphology


### Baisc Funcs ###

def coord_Im2Array(X_IMAGE, Y_IMAGE):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE)-1, 0)), int(max(round(X_IMAGE)-1, 0))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr):
    """ Convert image coordniate to numpy array coordinate """
    X_IMAGE, Y_IMAGE = y_arr+1, x_arr+1
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

def multi_pow(x, n0, n_s, theta0, theta_s, I_theta0, a0=None, a_s=None):
    """ Continuous multi-power law for single element """
    if a0 is None:
        a0, a_s = compute_multi_pow_norm(n0, n_s, theta0, theta_s, I_theta0)
        
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

def multi_power1d(x, n0, theta0, I_theta0, n_s, theta_s):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a0, a_s = compute_multi_pow_norm(n0, n_s, theta0, theta_s, I_theta0)
    
    y = a0 * np.power(x + 1e-6, -n0)
    y[x<=theta0] = I_theta0
    for i, (n_i, a_i, theta_i) in enumerate(zip(n_s, a_s, theta_s)):
        y_i = a_i * np.power(x, -n_i)
        y[x>theta_i] = y_i[x>theta_i]
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

def compute_multi_pow_norm(n0, n_s, theta0, theta_s, I_theta0):
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

def multi_power1d_normed(x, n0, theta0, n_s, theta_s):
    a0, a_s = compute_multi_pow_norm(n0, n_s, theta0, theta_s, 1)
    norm_mpow = quad(multi_pow, 0, np.inf, args=(n0, n_s, theta0, theta_s, 1, a0, a_s))[0]
    y = multi_power1d(x, n0, theta0, 1, n_s, theta_s) / norm_mpow
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

def multi_power2d(xx, yy, n0, theta0, I_theta0, n_s, theta_s, cen):
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(rr, -n0) 
    z[rr<=theta0] = I_theta0
    
    I_theta_i = a0 * np.power(1.*theta_s[0], -n0)
    
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(rr, -n_i)
        z[rr>theta_i] = z_i[rr>theta_i]
        try:
            I_theta_i = a_i * np.power(1.*theta_s[i+1], -n_i)
        except IndexError:
            pass
    return z


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

def multi_power2d_Amp2Flux(n0, theta0, n_s, theta_s, Amp=1):
    a0, a_s = compute_multi_pow_norm(n0, n_s, theta0, theta_s, Amp)
    n_all = np.concatenate([[n0],n_s])
    a_all = np.concatenate([[a0],a_s])
    theta_all = np.concatenate([[theta0],theta_s])
    
    I_2D = Amp * np.pi * theta0**2
    for k in range(len(n_s)):
        if n_all[k] == 2:
            I_2D += 2*np.pi * a_all[k] * np.log(theta_all[k+1]/theta_all[k])
        else:
            I_2D += 2*np.pi * a_all[k] * (theta_all[k]**(2-n_all[k]) - theta_all[k+1]**(2-n_all[k])) / (n_all[k]-2) 
    I_2D += 2*np.pi * a_all[-1] * theta_all[-1]**(2-n_all[-1]) / (n_all[-1]-2)   
    return I_2D

def multi_power2d_Flux2Amp(n0, theta0, n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n0, theta0, n_s, theta_s, Amp=1)


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
    """ gamma in pixel """
    return 1. / C_pow2Dto1D(n, theta0)

def C_mpow2Dto1D(n0, theta0, n_s, theta_s):
    """ theta0 in pixel """
    a0, a_s = compute_multi_pow_norm(n0, n_s, theta0, theta_s, 1)
    n_all = np.concatenate([[n0],n_s])
    a_all = np.concatenate([[a0],a_s])
    theta_all = np.concatenate([[theta0],theta_s])
    
    I_2D = 1. * np.pi * theta0**2
    for k in range(len(n_s)):
        if n_all[k] == 2:
            I_2D += 2*np.pi * a_all[k] * np.log(theta_all[k+1]/theta_all[k])
        else:
            I_2D += 2*np.pi * a_all[k] * (theta_all[k]**(2-n_all[k]) - theta_all[k+1]**(2-n_all[k])) / (n_all[k]-2) 
    I_2D += 2*np.pi * a_all[-1] * theta_all[-1]**(2-n_all[-1]) / (n_all[-1]-2)   
    
    I_1D = 1. * theta0
    for k in range(len(n_s)):
        if n_all[k] == 1:
            I_1D += a_all[k] * np.log(theta_all[k+1]/theta_all[k])
        else:
            I_1D += a_all[k] * (theta_all[k]**(1-n_all[k]) - theta_all[k+1]**(1-n_all[k])) / (n_all[k]-1) 
    I_1D += a_all[-1] * theta_all[-1]**(1-n_all[-1]) / (n_all[-1]-1)
    
    return I_2D / I_1D 

def C_mpow1Dto2D(n0, theta0, n_s, theta_s):
    return 1. / C_mpow2Dto1D(n0, theta0, n_s, theta_s)

    
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

def source_detection(data, sn=2, b_size=120, k_size=3, fwhm=3, 
                     morph_oper=morphology.dilation,
                     sub_background=True, mask=None):
    from astropy.convolution import Gaussian2DKernel
    from photutils import detect_sources
    
    if sub_background:
        back, back_rms = background_sub_SE(data, b_size=b_size)
        threshold = back + (sn * back_rms)
    else:
        back = np.zeros_like(data)
        threshold = np.nanstd(data)
    sigma = fwhm * gaussian_fwhm_to_sigma    # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=k_size, y_size=k_size)
    kernel.normalize()
    segm_sm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel, mask=mask)

    segm_sm = morph_oper(segm_sm.data)
    data_ma = data.copy() - back
    data_ma[segm_sm!=0] = np.nan

    return data_ma, segm_sm

def make_mask_map(image, sn_thre=2.5, b_size=25, n_dilation=5):
    
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=5)
    
    segmap = segm0.data.copy()
    for i in range(n_dilation):
        segmap = morphology.dilation(segmap)
        
    segmap2 = segm0.data.copy()
    segmap2[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap2!=0)
    
    return mask_deep, segmap2

def make_mask_strip(image_size, star_pos, fluxs, width=5, n_strip=12, dist_strip=480):    
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

    
def cal_profile_1d(img, cen=None, mask=None, back=None, 
                   color="steelblue", xunit="pix", yunit="intensity",
                   seeing=2.5, pix_scale=2.5, ZP=27.1, sky_mean=884, sky_std=3,
                   core_undersample=False, dr=1, lw=2, alpha=0.7, label=None, 
                   draw=True, scatter=False, plot_line=False, verbose=False):
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
        plt.xlabel("r [acrsec]")
    elif xunit == "pix":
        plt.xlabel("r [pix]")
        
    d_r = dr * pix_scale if xunit == "arcsec" else dr
    
    # Radial bins: discrete/linear within r_core + log beyond it
    if core_undersample:  
        # for undersampled core, bin in individual pixels 
        bins_inner = np.unique(r[r<r_core]) + 1e-3
    else: 
        bins_inner = np.linspace(0, r_core, np.int(r_core/d_r)) + 1e-3
        
    n_bin_outer = np.max([7, np.min([np.int(r_max/d_r/10), 50])])
    if r_max > (r_core+d_r):
        bins_outer = np.logspace(np.log10(r_core+d_r), np.log10(r_max-d_r), n_bin_outer)
    else:
        bins_outer = []
    bins = np.concatenate([bins_inner, bins_outer])
    n_pix_rbin, bins = np.histogram(r, bins=bins)
    
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
            plt.plot(r_rbin, np.log10(z_rbin), "-o", mec="k", lw=lw, color=color, alpha=alpha, zorder=3, label=label) 
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
            B_rbin = Intensity2SB(y=z_rbin, BKG=np.median(back), ZP=ZP, pix_scale=pix_scale)
            B_sky = Intensity2SB(y=sky_std, BKG=0, ZP=ZP, pix_scale=pix_scale)

            plt.plot(r_rbin, B_rbin, "-o", mec="k", lw=lw, color=color, alpha=alpha, zorder=3, label=label)   
            if scatter:
                B = Intensity2SB(y=z, BKG=np.median(back), ZP=ZP, pix_scale=pix_scale)
                plt.scatter(r[r<3*r_core], B[r<3*r_core], color=color, s=6, alpha=0.2, zorder=1)
                plt.scatter(r[r>3*r_core], B[r>3*r_core], color=color, s=3, alpha=0.1, zorder=1)
            plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
            plt.gca().invert_yaxis()
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8,r_rbin[np.isfinite(r_rbin)][-1]*1.2)
            plt.ylim(30,17)

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
        dz_rbin = np.diff(np.log10(z_rbin)) 
        dz_cum = np.cumsum(dz_rbin)

        if plot_line:
            r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
            plt.axvline(r_satr,color="k",ls="--",alpha=0.9)
            plt.axvline(r_core,color="k",ls=":",alpha=0.9)
            plt.axhline(B_sky,color="gray",ls="-.",alpha=0.7)

            use_range = (r_rbin>r_satr) & (r_rbin<r_core)
        else:
            use_range = True
    return r_rbin, z_rbin, logzerr_rbin

### Catalog Manipulation Helper ###

def crop_catalog(cat, bounds, keys=("X_IMAGE", "Y_IMAGE")):
    Xmin, Ymin, Xmax, Ymax = bounds
    A, B = keys
    crop = (cat[A]>=Xmin) & (cat[A]<=Xmax) & (cat[B]>=Ymin) & (cat[B]<=Ymax)
    return cat[crop]



### Nested Fitting Helper ###

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

