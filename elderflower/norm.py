import os
import re
import sys
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip
from photutils import CircularAnnulus

from .io import logger, save_pickle, load_pickle, check_save_path
from .stack import stack_star_image

def counter(i, number):
    if np.mod((i+1), number//3) == 0:
        print("    - completed: %d/%d"%(i+1, number))
    
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
                        mag_saturate=13.5,
                        mag_limit=15,
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
    mag_saturate : float, optional, default 13.5
        Estimate of magnitude at which the image is saturated.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured.
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
        if verbose:
            counter(i, len(table_target))
        num, mag_auto = row['NUMBER'], row['MAG_AUTO']
        
        wid_cross_ = wid_cross  # spikes mask
        
        # For brighter sources, use a broader window
        if mag_auto <= mag_saturate-3:
            n_win = int(40 * k_win)
        elif mag_saturate-3 < mag_auto < mag_saturate:
            n_win = int(30 * k_win)
        elif mag_saturate < mag_auto < mag_limit:
            n_win = int(20 * k_win)
            wid_cross_ = max(wid_cross//2, 1)
        else:
            n_win = int(10 * k_win)
            wid_cross_ = 0
        
        # Make thumbnail of the star and mask sources
        thumb = Thumb_Image(row, wcs)
        thumb.extract_star(image, seg_map, n_win=n_win)
        
        # Measure the mean, med and std of intensity at r_scale
        thumb.compute_Rnorm(R=r_scale,
                            wid_ring=wid_ring,
                            wid_cross=wid_cross_,
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

def measure_Rnorm_all(hdu_path,
                      table,
                      bounds,
                      seg_map=None,
                      r_scale=12,
                      mag_limit=15,
                      mag_saturate=13.5,
                      mag_stack_limit=None,
                      mag_name='rmag_PS',
                      k_enlarge=1,
                      width_ring=0.5,
                      width_cross=4,
                      obj_name="",
                      display=False,
                      save=True, dir_name='.',
                      read=False, verbose=True):
    """
    Measure intensity at r_scale for bright stars in table.

    Parameters
    ----------
    hdu_path : str
        path of hdu data
    table : astropy.table.Table
        SExtractor table containing measurements of sources.
    bounds : 1d array or list
        Boundaries of the region in the image [Xmin, Ymin, Xmax, Ymax].
    seg_map : 2d array, optional, default None
        Full segmentation map used to mask nearby sources during the measurement.
        If not given, it will be done locally by photutils.
    r_scale : int, optional, default 12
        Radius in pixel at which the flux scaling is measured.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured.
    mag_saturate : float, optional, default 13.5
        Estimate of magnitude at which the image is saturated.
    mag_stack_limit : float, optional, default None
        Max limit for stacking core PSF. Use mag_limit if None.
    mag_name : str, optional, default 'rmag_PS'
        Column name of magnitude used in the table.
    k_enlarge : int, optional, default 1
        Enlargement factor for extracting thumbnails.
    width_ring : float, optional, default 0.5
        Half-width in pixel of ring used to measure the scaling.
    width_cross : float, optional, default 4
        Half-width in pixel of the spike mask when measuring the scaling.
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
    
    from .utils import convert_decimal_string
    
    with fits.open(hdu_path) as hdul:
        image = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)
    
    if verbose:
        msg = "Measure intensity at R = {0} ".format(r_scale)
        msg += "for catalog stars {0:s} < {1:.1f} in ".format(mag_name, mag_limit)
        msg += "{0}.".format(bounds)
        logger.info(msg)
    
    band = mag_name[0]
    range_str = 'X[{0:d}-{2:d}]Y[{1:d}-{3:d}]'.format(*bounds)
    
    mag_str = convert_decimal_string(mag_limit)
    
    fn_table_norm = os.path.join(dir_name, '%s-norm_%dpix_%smag%s_%s.txt'\
                                %(obj_name, r_scale, band, mag_str, range_str))
    fn_res_thumb = os.path.join(dir_name, '%s-thumbnail_%smag%s_%s.pkl'\
                                %(obj_name, band, mag_str, range_str))
    
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
                                                      mag_saturate=mag_saturate,
                                                      mag_limit=mag_limit,
                                                      k_win=k_enlarge,
                                                      display=display,
                                                      verbose=verbose)
        
        keep_columns = ['NUMBER', 'MAG_AUTO', 'MAG_AUTO_corr', 'MU_MAX', 'FLAGS', mag_name] \
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
    psf_size = 5 * r_scale + 1
    psf_size = int(psf_size/2) * 2 + 1 # round to odd

    not_edge = (table_norm['X_IMAGE'] > bounds[0] + psf_size) & \
               (table_norm['X_IMAGE'] < bounds[2] - psf_size) & \
               (table_norm['Y_IMAGE'] > bounds[1] + psf_size) & \
               (table_norm['Y_IMAGE'] < bounds[3] - psf_size)

    if mag_stack_limit is None:
        mag_stack_limit = mag_limit
        
    to_stack = (table_norm['MAG_AUTO']>mag_saturate+0.5) & (table_norm['MAG_AUTO']<mag_stack_limit) & (table_norm['FLAGS']<3) & not_edge
    table_stack = table_norm[to_stack]

    psf_stack = stack_star_image(table_stack, res_thumb,
                                 size=psf_size, verbose=verbose)

    if save:
        fits.writeto(fn_psf_satck, data=psf_stack, overwrite=True)
        if verbose:
            logger.info(f"Saved stacked PSF to {fn_psf_satck}")
            
    return table_norm, res_thumb

