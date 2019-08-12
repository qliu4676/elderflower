import sys
import os
import numpy as np
import math
from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy import wcs
from astropy.table import Table
from astropy.modeling import models
from astropy.stats import sigma_clip, SigmaClip, mad_std, median_absolute_deviation, gaussian_fwhm_to_sigma
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())
from skimage import morphology


### Baisc Funcs ###
def coord_Im2Array(X_IMAGE, Y_IMAGE):
    """ Convert image coordniate to numpy array coordinate """ 
    x_arr, y_arr = int(round(Y_IMAGE-1)), int(round(X_IMAGE-1))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr):
    """ Convert image coordniate to numpy array coordinate """ 
    X_IMAGE, Y_IMAGE = y_arr+1, x_arr+1
    return X_IMAGE, Y_IMAGE

def Intensity2SB(y, BKG, ZP, pix_scale):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """ 
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

def round_good_fft(x):
    return min(2**math.ceil(math.log2(x)), 3 * 2**math.floor(math.log2(x)-1))

def power1d(x, n, theta0, I_theta0): 
#     x[x<=0] = x[x>0].min()
    a = I_theta0/(theta0)**(-n)
    y = a * np.power(x, -n)
    return y

def trunc_pow(x, n, theta0, I_theta0=1):
    """ Truncated power law for single element, normalized = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * x**(-n) if x > theta0 else I_theta0
    return y

def trunc_power1d(x, n, theta0, I_theta0=1): 
    """ Truncated power law for 1d array, normalized = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x, -n) 
    y[x<theta0] = I_theta0
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

def multi_power1d(x, n0, theta0, I_theta0, n_s, theta_s):
#     n_s, theta_s = np.atleast1d(n_s), np.atleast1d(theta_s)
    ind_slice = [np.argmin(x<theta_i) for theta_i in theta_s]
    a0 = I_theta0/(theta0)**(-n0)
    y = a0 * np.power(x, -n0)
    I_theta_i = a0 * np.power(1.*theta_s[0], -n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        y_i = a_i * np.power(x, -n_i)
        y[x>theta_i] = y_i[x>theta_i]
        try:
            I_theta_i = a_i * np.power(1.*theta_s[i+1], -n_i)
        except IndexError:
            pass
    return y

def power2d(xx, yy, n, theta0, I_theta0, cen): 
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    return z 

def trunc_power2d(x, y, n, theta0, I_theta0, cen): 
    """ Truncated power law for 2d array, normalized = I_theta0 at theta0 """
    r = np.sqrt((x-cen[0])**2 + (y-cen[1])**2) + 1e-6
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(r, -n) 
    z[r<theta0] = I_theta0
    return z

def multi_power2d(x, y, n0, theta0, I_theta0, n_s, theta_s, cen):
    r = np.sqrt((x-cen[0])**2 + (y-cen[1])**2) + 1e-6
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(r, -n0) 
    I_theta_i = a0 * np.power(1.*theta_s[0], -n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(r, -n_i)
        z[r>theta_i] = z_i[r>theta_i]
        try:
            I_theta_i = a_i * np.power(1.*theta_s[i+1], -n_i)
        except IndexError:
            pass
    return z

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

def moffat2d_Flux2Amp(r_core, beta, Flux=1):
    return Flux * (beta-1) / r_core**2 / np.pi

def moffat2d_Amp2Flux(r_core, beta, Amp=1):
    return Amp * r_core**2 * np.pi / (beta-1)

def power2d_Flux2Amp(n, theta0, Flux=1):
    I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    return I_theta0


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
    from photutils import detect_sources 
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=5)

    from skimage import morphology
    segmap = segm0.data.copy()
    for i in range(n_dilation):
        segmap = morphology.dilation(segmap)
    segmap2 = segm0.data.copy()
    segmap2[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap2!=0)
    
    return mask_deep, segmap2

def make_mask_strip(image_size, star_pos, fluxs, width=5, n_strip=12):    
    yy, xx = np.mgrid[:image_size, :image_size]
    phi_s = np.linspace(-90, 90, n_strip+1)
    a_s = np.tan(phi_s*np.pi/180)
    mask_strip_s = np.empty((len(star_pos), image_size, image_size))
    
    for k, (x_b, y_b) in enumerate(star_pos[fluxs.argsort()]):
        m_s = (y_b-a_s*x_b)
        mask_strip = np.logical_or.reduce([abs((yy-a*xx-m)/math.sqrt(1+a**2)) < width 
                                           for (a, m) in zip(a_s, m_s)])
        mask_strip_s[k] = mask_strip
        
    return mask_strip_s

    
def cal_profile_1d(img, cen=None, mask=None, back=None, 
                   color="steelblue", xunit="pix", yunit="intensity",
                   seeing=2.5, pix_scale=2.5, ZP=0, sky_mean=0, sky_std=1,
                   core_undersample=True, plot_line=False, label=None):
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
    r_core = np.int(3 * seeing/pix_scale) # core radisu in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * np.median(back)
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.0001 * z_diff[-1]))
    r_max = np.sqrt(n_pix_max/np.pi)
    r_max = np.min([img.shape[0]//2, r_max])
    print("Maximum R: %d (pix)"%np.int(r_max))    
    
    if xunit == "arcsec":
        r = r * pix_scale   # radius in arcsec
        r_core = r_core * pix_scale
        r_max = r_max * pix_scale
        plt.xlabel("r [acrsec]")
    elif xunit == "pix":
        plt.xlabel("r [pix]")
        
    d_r = 1 * pix_scale if xunit == "arcsec" else 1
    
    # Radial bins: discrete/linear within r_core + log beyond it
    if core_undersample:  
        # for undersampled core, bin in individual pixels 
        bins_inner = np.unique(r[r<r_core]) + 1e-3
    else: 
        bins_inner = np.linspace(0, r_core, np.int(r_core/d_r)) + 1e-3
        
    n_bin_outer = np.max([10, np.min([np.int(r_max/d_r/10), 50])])
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
      
        z_clip = sigma_clip(z[in_bin], sigma=5, maxiters=10)
        zb = np.mean(z_clip)
        sigma_zb = np.std(z_clip)
        z_rbin = np.append(z_rbin, zb)
        logzerr_rbin = np.append(logzerr_rbin, 0.434 * ( sigma_zb / zb))

    if yunit == "intensity":  
        # plot radius in Intensity
        plt.plot(r_rbin, np.log10(z_rbin), "-o", mec="k", color=color, alpha=0.9,zorder=3, label=label)   
        plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                         color=color, alpha=0.2, zorder=1)
        plt.ylabel("log Intensity")
        plt.xlim(r_rbin[np.isfinite(r_rbin)][0]-0.1, r_rbin[np.isfinite(r_rbin)][-1] + 2)
        
    elif yunit == "SB":  
        # plot radius in Surface Brightness
        B_rbin = Intensity2SB(y=z_rbin, BKG=np.median(back), ZP=ZP, pix_scale=pix_scale)
        B_sky = Intensity2SB(y=sky_std, BKG=0, ZP=ZP, pix_scale=pix_scale)
        
        plt.plot(r_rbin, B_rbin, "-o", mec="k", color=color, alpha=0.9, zorder=3, label=label)   
        plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
        plt.gca().invert_yaxis()
        plt.xscale("log")
        plt.axhline(B_sky,color="gray",ls="-.",alpha=0.7)
        plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8,r_rbin[np.isfinite(r_rbin)][-1]*1.2)
    
    # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
    dz_rbin = np.diff(np.log10(z_rbin)) 
    dz_cum = np.cumsum(dz_rbin)
    
    if plot_line:
        r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
        plt.axvline(r_satr,color="k",ls="--")
        plt.axvline(r_core,color="k",ls=":")
        
        use_range = (r_rbin>r_satr) & (r_rbin<r_core)
    else:
        use_range = True
    return r_rbin, z_rbin, logzerr_rbin, use_range


### Nested Fitting Helper ###
def save_nested_fitting_result(res, filename='fit.res'):
    import dill
    with open(filename,'wb') as file:
        dill.dump(res, file)
        
# def plot_fitting_vs_truth_PSF(res, true_pars, n_bootsrap=100, save=False, version=""):
#     from dynesty import utils as dyfunc
    
#     samples = res.samples  # samples
#     weights = np.exp(res.logwt - res.logz[-1])  # normalized weights
#     # Compute weighted mean and covariance.
#     mean, cov = dyfunc.mean_and_cov(samples, weights)
#     # Resample weighted samples.
#     samples_eq = dyfunc.resample_equal(samples, weights)
    
#     from astropy.stats import bootstrap
#     samples_eq_bs = bootstrap(samples_eq, bootnum=1, samples=n_bootsrap)[0]
    
#     gamma, alpha, n, theta, mu, sigma = true_pars.values()
    
#     Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    
    
#     plt.figure(figsize=(8,6))
#     r = np.logspace(0.,3,100)
#     plt.semilogy(r, Mof_mod_1d(r) + power1d(r, n, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)),
#                  label="Truth", color="steelblue", lw=3, zorder=2)
#     for n_k, theta_k in zip(samples_eq_bs[:,0].T, samples_eq_bs[:,1].T):
#         plt.semilogy(r, Mof_mod_1d(r) + power1d(r, n_k, theta0=theta_k*gamma, I_theta0=Mof_mod_1d(theta_k*gamma)),
#                      color="lightblue",alpha=0.1,zorder=1)
#     else:
#         plt.semilogy(r, Mof_mod_1d(r) + power1d(r, mean[0], theta0=mean[1]*gamma, I_theta0=Mof_mod_1d(mean[1]*gamma)),
#                      label="Fit", color="mediumblue",ls="--",lw=2,alpha=0.75,zorder=2)
#     plt.semilogy(r, Mof_mod_1d(r), label="Moffat", ls=":", color="orange", lw=2, alpha=0.7,zorder=1)
#     plt.semilogy(r, power1d(r, n, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)),
#                  label="Power", color="g", lw=2, ls=":",alpha=0.7,zorder=1)
# #     plt.axhline(mu/1e6,color="k",ls=":")
#     plt.ylim(3e-9,3)
#     plt.xlabel(r"$\rm r\,[pix]$",fontsize=18)
#     plt.ylabel(r"$\rm Intensity$",fontsize=18)
#     plt.xscale("log")
#     plt.tight_layout()
#     plt.legend(fontsize=12)
#     if save:
#         plt.savefig("./tmp/PSF%s.png"%version,dpi=150)
#     plt.show()