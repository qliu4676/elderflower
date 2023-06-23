import os
import re
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation

from astropy.io import fits
from astropy.table import Table
from photutils.aperture import CircularAperture

from .io import logger
from .plotting import LogNorm
from .utils import background_annulus
from . import DF_pixel_scale, DF_raw_pixel_scale
                     
### Stacking PSF functions ###

def resample_thumb(image, mask, center, shape_new=None):
    """
    Shift and resample the thumb image & mask to have odd dimensions
    and center at the center pixel. The new dimension can be specified.
    
    Parameters
    ----------
    
    image : input image, 2d array
    mask : input mask, 2d bool array (masked = 1)
    center : center of the target, array or turple
    shape_new : new shape after resampling
    
    Returns
    -------
    image_ : output image, 2d array
    mask_ : output mask, 2d bool array (masked =1)
    center_ : center of the target after the shift
        
    """

    from scipy.interpolate import RectBivariateSpline

    X_c, Y_c = center
    NX, NY = image.shape

    # original grid points
    Xp, Yp = np.linspace(0, NX-1, NX), np.linspace(0, NY-1, NY)

    rbspl = RectBivariateSpline(Xp, Yp, image, kx=3, ky=3)
    rbspl_ma = RectBivariateSpline(Xp, Yp, mask, kx=1, ky=1)

    # new NAXIS
    if shape_new is None:
        NX_ = NY_ = int(np.floor(image.shape[0]/2) * 2) - 3
    else:
        NX_, NY_ = shape_new

    # shift grid points
    Xp_ = np.linspace(X_c - NX_//2, X_c + NX_//2, NX_)
    Yp_ = np.linspace(Y_c - NY_//2, Y_c + NY_//2, NY_)

    # resample image
    image_ = rbspl(Xp_, Yp_)
    mask_ = rbspl_ma(Xp_, Yp_) > 0.5

    center_ = np.array([X_c - Xp_[0], Y_c - Yp_[0]])
    return image_, mask_, center_

def stack_star_image(table_stack, res_thumb, size=61, verbose=True):
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
    
    size = int(size/2) * 2 + 1
    shape = (size, size)
    canvas = np.zeros(shape)
    footprint = np.zeros_like(canvas)
    
    i = 0
    if verbose:
        logger.info("Stacking {0} non-staurated stars to obtain the PSF core...".format(len(table_stack)))
    for num in table_stack['NUMBER']:

        # Read image, mask and center
        img_star = res_thumb[num]['image']
        mask_star = res_thumb[num]['mask']
        cen_star = res_thumb[num]['center']
        
        # enlarge mask
        for j in range(1):
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
        img_star_ = img_star_/img_star_.sum()
        
        # add cutout to canvas
        dx = abs(shape_star_[0]-canvas.shape[0])//2
        dy = abs(shape_star_[1]-canvas.shape[1])//2
        if shape_star_[0] > size:
            cutout = img_star_[dx:-dx,dy:-dy]
            canvas += cutout
            footprint += (cutout!=0)
        elif shape_star_[0] < size:
            cutout = img_star_
            canvas[dx:-dx,dy:-dy] += cutout
            footprint[dx:-dx,dy:-dy] += (cutout!=0)
        else:
            canvas += img_star_
            footprint += 1
            
        i += 1

    image_stack = canvas/footprint
    image_stack = image_stack/image_stack.sum()

    return image_stack

def make_global_stack_PSF(dir_name,
                          bounds_list,
                          obj_name, band,
                          overwrite=True,
                          verbose=True):
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
                
        image_stack = image_stack/np.nansum(image_stack)
        
        if i>0:
            if verbose:
                logger.info("Read & stack {:} PSF.".format(i+1))

        fits.writeto(fn_stack, data=image_stack, overwrite=True)
        if verbose:
            logger.info("Saved stacked PSF as {:}".format(fn_stack))
    else:
        logger.warning("{:} existed. Skip Stack.".format(fn_stack))

        
def montage_psf_image(image_psf, image_wide_psf, r=12, dr=0.5, wid_cross=None):
    """
    Montage the core of the stacked psf and the wing of the wide psf model.
    
    Parameters
    ----------
    image_psf : 2d array
        The image of the inner PSF.
    image_wide_psf : 2d array
        The image of the wide-angle PSF.
    r : int, optional, default 12
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
    
    if wid_cross is not None:
        mask_cross = np.logical_or.reduce([abs(yy_psf-cen_psf[0])<wid_cross, abs(xx_psf-cen_psf[1])<wid_cross])
    else:
        mask_cross = np.zeros_like(image_psf, dtype=bool)
    I_psf = np.median(image_psf[(rr_psf<r+dr)&(rr_psf>r-dr)&(~mask_cross)])

    # Montage
    image_PSF[rr<r] = image_psf[rr_psf<r]/ I_psf * I_wide
    image_PSF = image_PSF/image_PSF.sum()

    return image_PSF


def get_aperture_flux_fraction(image, frac):
    """ Get radius within which contains certain fraction of total flux. """

    shape = image.shape
    size = min(shape)

    cen = ((shape[1]-1)/2., (shape[0]-1)/2.)

    r_aper_list = np.array(list(np.around(np.logspace(0.3, np.log10(size//2), 50), 1)))
    flux_list = np.empty_like(r_aper_list)

    for k, r_aper in enumerate(r_aper_list):
        aper = CircularAperture(cen, r=r_aper)
        aper_ma = aper.to_mask().to_image(shape)
        flux_list[k] = (image*aper_ma).sum()
        
    total_flux = np.ma.sum(image) * frac
    r_aper_target = r_aper_list[np.argmin(abs(flux_list-total_flux))]

    return round(r_aper_target)

def fine_stack_PSF_image(table_stack, res_thumb, size=61, fwhm_psf=5, n_iter=2, verbose=True):

    from scipy.ndimage import shift
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.convolution import convolve_fft, Gaussian2DKernel
    
    from photutils.background import MADStdBackgroundRMS, MMMBackground
    from photutils.detection import IRAFStarFinder
    from photutils.psf import (DAOGroup, IterativelySubtractedPSFPhotometry)
    from photutils.psf.epsf import EPSFModel
    from photutils.segmentation import detect_sources

    from .utils import background_extraction
    
    # image grid
    image_shape = (size, size)
    image_cen = ((size-1)/2., (size-1)/2.)
    x_ = y_ = np.linspace(0,size-1,size)
    
    if verbose:
        logger.info("Stacking {0} non-staurated stars to obtain the PSF core...".format(len(table_stack)))
        
    stars_scaled = np.ma.empty((len(table_stack), size, size))
    
    # Shift image center and normalize by aperture flux
    for ind in range(len(table_stack)):
        num = table_stack['NUMBER'][ind]
        bkg = res_thumb[num]['bkg']
        image = res_thumb[num]['image']
        mask = res_thumb[num]['mask']
        center = res_thumb[num]['center']
        
        for j in range(3):
            mask = binary_dilation(mask)
                
        image = np.ma.array(image - bkg, mask=mask)
        
        # resample
        image_new, mask_new, center_new = resample_thumb(image, mask, center, shape_new=image_shape)
        
        # rough estimate of first total flux
        r_aper = get_aperture_flux_fraction(np.ma.array(image_new, mask=mask_new), frac=0.99)
        r_aper = min(r_aper, size)
        aper = CircularAperture(center, r=r_aper)
        aper_new = CircularAperture(center_new, r=r_aper)
        aper_ma = aper_new.to_mask().to_image(image_new.shape)
        
        # normalization
        flux = np.ma.sum(image_new[(aper_ma == 1) & (~mask_new)])
        stars_scaled[ind] = image_new / flux
    
    # first median stack
    star_med_0 = np.ma.median(stars_scaled, axis=0)
    star_med = star_med_0.copy()
    
    # some static PSF photometry setup
    bkgrms = MADStdBackgroundRMS()
    daogroup = DAOGroup(2.0 * fwhm_psf)
    mmmbkg = MMMBackground()
    fitter = LevMarLSQFitter()
    iraf_finder_kws = dict(fwhm=fwhm_psf, brightest=5,
                           minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                           sharplo=0.0, sharphi=2.0)
    # mask growth kernel
    kernel_mask = Gaussian2DKernel(fwhm_psf*gaussian_fwhm_to_sigma)
    
    for i in range(n_iter):
        # first R99 estimate
        r_aper_star = get_aperture_flux_fraction(star_med, frac=0.99)
        
        # some dynamic PSF photometry setup
        psf_model = EPSFModel(star_med, x0=image_cen[0], y0=image_cen[1])
        psf_photometry_kws = dict(group_maker=DAOGroup(2.0 * fwhm_psf),
                                  bkg_estimator=MMMBackground(),
                                  psf_model=psf_model,
                                  fitter=LevMarLSQFitter(),
                                  aperture_radius=r_aper_star,
                                  fitshape=image_shape, niters=1)

        # Do PSF photometry
        stars_out = np.ma.empty_like(stars_scaled)

        for k, image in enumerate(stars_scaled):
            
            # Aperture mask
            aper = CircularAperture(image_cen, r=r_aper_star)
            aper_ma = aper.to_mask().to_image(image_shape) == 1
            
            # def PSF photometry model
            iraffind = IRAFStarFinder(threshold=3*bkgrms(image[~aper_ma]), **iraf_finder_kws)
            photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, **psf_photometry_kws)
            
            # do photometry
            result_tab = photometry(image=image)
            irow = ((result_tab['x_fit'] - image_cen[0])**2+(result_tab['y_fit'] - image_cen[1])**2).argmin()
            
            # get residual
            residual_image = photometry.get_residual_image()
            residual_image_ma = residual_image.copy()

            # mask target star
            residual_image_ma[aper_ma] = np.nan

            # detect nearby souces
            std_res_ma = bkgrms(residual_image_ma)
            segm = detect_sources(residual_image_ma, threshold=3*std_res_ma, npixels=5)
            
            if segm is None:
                mask = np.zeros_like(image, dtype=bool)
            else:
                mask = segm.data > 0
                
            mask = convolve_fft(mask, kernel_mask) > 0.1
            
            # shift
            dy, dx = (image_cen[1]-result_tab[irow]['y_fit'], image_cen[0]-result_tab[irow]['x_fit'])
            image_shifted = shift(image, [dy, dx], order=3, mode='nearest')
            mask_shifted = shift(mask, [dy, dx], order=0, mode='constant', cval=0)
            
            # norm
            image_star = np.ma.array(image_shifted, mask=mask_shifted)
            
            bkg_val = np.ma.median(np.ma.array(residual_image, mask=mask | aper_ma | np.isnan(image)))
            image_star = image_star - bkg_val
            image_star = image_star/image_star.sum()
            
            image_star = np.ma.array(image_star, mask=mask0)
            
            stars_out[k] = image_star
            
        star_med_out = np.nanmedian(stars_out, axis=0)
        star_med = star_med_out / star_med_out.sum()
    
    image_stack = star_med.copy()
    
    return image_stack
