#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from functools import partial

from astropy.io import fits
from astropy.table import Table

from .io import (find_keyword_header, check_save_path, get_SExtractor_path,
                 default_SE_config, default_SE_conv, default_SE_nnw)
from .image import DF_pixel_scale


SE_params = ['NUMBER','X_WORLD','Y_WORLD','FLUXERR_AUTO','MAG_AUTO',
             'MU_MAX','CLASS_STAR','ELLIPTICITY']
SE_config_path = default_SE_config
SE_executable = get_SExtractor_path()


def Run_Detection(hdu_path, obj_name, band,
                  threshold=5, work_dir='./',
                  config_path=None,
                  executable=SE_executable,
                  ZP_keyname='REFZP', ZP=None,
                  ref_cat='APASSref.cat',
                  apass_dir='~/Data/apass/', **kwargs):
                  
    """
    
    Run a first-step source detection with SExtractor. This step generates a SExtractor catalog
     and segementation map for the cross-match and measurement in Match_Mask_Measure.
    
    Magnitudes are converted using the zero-point stored in the header ('ZP_keyname'). If not
    stored in the header, it will try to compute the zero-point by cross-match with the APASS
    catalog. In this case, the directory to the APASS catalogs is needed ('apass_dir'). If a
    reference catalog already exists, it can be provided ('ref_cat') to save time.
    
    
    Parameters
    ----------
    
    hdu_path : str
        Full path of hdu data
    obj_name : str
        Object name
    band : str
        Filter name ('G', 'g' or 'R', 'r')
    threshold : int, optional, default 5
        Detection and analysis threshold of SExtractor
    work_dir : str, optional, default current directory
        Full path of directory for saving
    config_path : str, optional, None
        Full path of configuration file of running SExtractor.
        By default it uses the one stored in configs/
    executable : str, optional, SE_executable
        Full path of the SExtractor executable. If SExtractor is installed this can be retrieved
        by typing '$which sex'  or  '$which source-extractor' in the shell.
    ZP_keyname : str, optional, default REFZP
        Keyword names of zero point in the header.
        If not found, a value can be passed by ZP.
    ZP : float or None, optional, default None
        Zero point value. If None, it finds ZP_keyname in the header. If not provided either,
        it will compute a zero point by cross-match with the APASS catalog.
    ref_cat : str, optional, default 'APASSref.cat'
        Full path file name of the APASS reference catalog.
        If not found, it will generate a reference catalog.
    apass_dir : str, optional, default '~/Data/apass/'
        Full path of the diectory of the APASS catalogs.
        
    Returns
    -------
    ZP: float
        Zero point value from the header, or a crossmatch with APASS, or a user-input.
        
    Notes
    -----
    
    SExtractor must be installed and the local executable path needs to be correct.
    A configuration file can be passed by config_path than default, but parameters can be
    overwritten by passing them as kwargs, e.g. (note SExtractor keywords are in capital):
    
        Run_Detection(..., DETECT_THRESH=10)
        
    will override threshold.
    
    """
                
    from dfreduce.detection import sextractor
    
    print(f"Run SExtractor on {hdu_path}...")
    
    b_name = band.lower()
    
    segname = os.path.join(work_dir, f'{obj_name}-{b_name}_seg.fits')
    catname = os.path.join(work_dir, f'{obj_name}-{b_name}.cat')
    
    header = fits.getheader(hdu_path)
    
    if config_path is None: config_path = SE_config_path
    
    # Find zero-point in the fits header
    if ZP_keyname not in header.keys():
    
        # If not in the header, check kwargs
        if type(ZP) is not float:
            
            # If not available in kwargs, compute by crossmatch with refcat
            from dfreduce.utils.catalogues import (match_catalogues, load_apass_in_region)
            print("Compute zero-point from crossmatch with APASS catalog...")
            
            # Run sextractor with free zero-point
            SE_catalog = sextractor.run(hdu_path,
                                        extra_params=SE_params,
                                        config_path=config_path,
                                        catalog_path=catname+'_zp',
                                        executable=executable,
                                        DETECT_THRESH=5, ANALYSIS_THRESH=5,
                                        FILTER_NAME=default_SE_conv,
                                        STARNNW_NAME=default_SE_nnw)
                                        
            # Load (APASS) reference catalog
            if os.path.exists(ref_cat):
                refcat = Table.read(ref_cat, format='ascii')
            else:
                print("Generate APASS reference catalog... It takes time.")
                ra_range = header['CRPIX1'] * header['CD1_1']
                dec_range = header['CRPIX2'] * header['CD2_2']

                minra, maxra = sorted([header['CRVAL1'] - ra_range, header['CRVAL1'] + ra_range])
                mindec, maxdec = sorted([header['CRVAL2'] - dec_range, header['CRVAL2'] + dec_range])
                if os.path.exists(apass_dir):
                    refcat = load_apass_in_region(apass_dir,
                                                  bounds=[mindec, maxdec, minra, maxra])
                    refcat.write(ref_cat, format='ascii')
                else:
                    raise FileNotFoundError('APASS directory not available.')

            # Crossmatch SE catalog with reference catalog
            imagecat_match, refcat_match = match_catalogues(SE_catalog, refcat, band, sep_max=3.)
            
            # Get the mean ZP from the crossmatched catalog
            ZP = np.mean(refcat_match[band] - imagecat_match[band])
            print("Matched zero-point = {:.3f}".format(ZP))
        
    else:
        ZP = np.float(header[ZP_keyname])
        print("Read zero-point from header : ZP = {:.3f}".format(ZP))
    
    SE_key = kwargs.keys()
    for THRE in ['DETECT_THRESH', 'ANALYSIS_THRESH']:
        if THRE not in SE_key: kwargs[THRE] = threshold
    if 'FILTER_NAME' not in SE_key : kwargs['FILTER_NAME'] = default_SE_conv
    if 'STARNNW_NAME' not in SE_key : kwargs['STARNNW_NAME'] = default_SE_nnw
    for key in ['CHECKIMAGE_TYPE', 'CHECKIMAGE_TYPE', 'MAG_ZEROPOINT']:
        if key in SE_key: SE_key.pop(key, None); print(f'WARNING: {NAME} are reserved.')
   
    SE_catalog = sextractor.run(hdu_path,
                                extra_params=SE_params,
                                config_path=config_path,
                                catalog_path=catname,
                                executable=executable,
                                MAG_ZEROPOINT=ZP,
                                CHECKIMAGE_TYPE='SEGMENTATION',
                                CHECKIMAGE_NAME=segname, **kwargs)
    
    if not (os.path.isfile(catname)) & (os.path.isfile(segname)):
        raise FileNotFoundError('SE catalog/segmentation not saved properly.')
        
    print(f"CATALOG saved as {catname}")
    print(f"SEGMENTATION saved as {segname}")
    
    return ZP


def Match_Mask_Measure(hdu_path,
                       bounds_list,
                       obj_name,
                       band,
                       pixel_scale=DF_pixel_scale,
                       ZP=None,
                       bkg=None,
                       field_pad=50,
                       r_scale=12,
                       mag_limit=15,
                       mag_saturate=13,
                       draw=True,
                       save=True,
                       use_PS1_DR2=False,
                       work_dir='./'):
                       
    """
    
    Generate a series of files as preparations for the fitting.
    
    The function completes by the following steps:
    1) Identify bright extended sources empirically and mask them.
    2) Crossmatch the SExtractor table with the PANSTARRS catalog.
    3) Correct the catalogued magnitudes to the used filter.
    4) Add saturated stars missing in the crossmatch by a correction.
    5) Make mask maps for dim stars with empirical apertures enlarged from SExtractor.
    6) Measure brightness in annuli around bright stars
    
    The output files are saved in:
    work_dir/obj_name/Measure-PS1 or work_dir/obj_name/Measure-PS2
    
    
    Parameters
    ----------
    
    hdu_path : str
        Full path of hdu data
    bounds_list : 2D list / turple
        List of boundaries of regions to be fit (Nx4)
        [[X min, Y min, X max, Y max],[...],...]
    obj_name : str
        Object name
    band : str
        Filter name ('G', 'g' or 'R', 'r')
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pixel
    ZP : float or None, optional, default None
        Zero point value (if None, read ZP from header)
    bkg : float or None, optional, default None
        Background estimated value (if None, read BACKVAL from header)
    field_pad : int, optional, default 100
        Padding size (in pix) of the field for crossmatch.
        Only used if use_PS1_DR2=False
    r_scale : int, optional, default 12
        Radius (in pix) at which the brightness is measured
        Default is 30" for Dragonfly.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured
    mag_saturate : float, optional, default 13
        Estimate of magnitude at which the image is saturated.
        The exact value will be fit if ZP provided.
    draw : bool, optional, default True
        Whether to draw diagnostic plots
    save : bool, optional, default True
        Whether to save results.
    use_PS1_DR2 : bool, optional, default False
        Whether to use PANSTARRS DR2. Crossmatch with DR2 is done by MAST query, which
        could easily fail if a field is too large (> 1 deg^2)
    work_dir : str, optional, default current directory
        Full path of directory for saving
    
    
    Returns
    -------
    None
        None
        
    """
    
    print("""Measure the intensity at R = %d for stars < %.1f
            as normalization of fitting\n"""%(r_scale, mag_limit))
    
    b_name = band.lower()
    bounds_list = np.atleast_2d(bounds_list)
    
    ##################################################
    # Read and Display
    ##################################################
    from .utils import crop_image, crop_catalog
    from astropy.stats import mad_std
    from astropy import wcs
    
    # Read hdu
    if not os.path.isfile(hdu_path):
        raise FileNotFoundError("Image does not exist. Check path.")
        
    with fits.open(hdu_path) as hdul:
        print("Read Image :", hdu_path)
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)

    # Read output from SExtractor detection
    SE_cat_full = Table.read(os.path.join(work_dir, f'{obj_name}-{b_name}.cat'), format="ascii.sextractor")
    
    seg_map = fits.getdata(os.path.join(work_dir, f'{obj_name}-{b_name}_seg.fits'))

    # Read global background model ZP from header
    if bkg is None: bkg = find_keyword_header(header, "BACKVAL")
    if ZP is None: ZP = find_keyword_header(header, "ZP")
    
    std = mad_std(data)
   
    # Short summary
    print("BACKVAL: %.2f +/- %.2f , ZP: %.2f\n"%(bkg, std, ZP))

    # Convert SE measured flux into mag
    flux = SE_cat_full["FLUX_AUTO"]
    mag = -2.5 * np.ma.log10(flux).filled(flux[flux>0].min()) + ZP
    SE_cat_full["MAG_AUTO"] = np.around(mag, 5)
    
    field_bounds = [field_pad, field_pad,
                    data.shape[1]-field_pad,
                    data.shape[0]-field_pad]
    
    if not use_PS1_DR2: print("Match field %r with catalog\n"%field_bounds)
    
    print("Measure Sky Patch (X min, Y min, X max, Y max) :")
    [print("%r"%b) for b in bounds_list.tolist()]
    
    # Display field_bounds and sub-regions to be matched
    patch, _ = crop_image(data, field_bounds,
                          sub_bounds=bounds_list,
                          seg_map=seg_map,
                          origin=0, draw=draw)
    
    # Crop parent SE catalog
    SE_cat = crop_catalog(SE_cat_full, field_bounds)

    ##################################################
    # Crossmatch with Star Catalog (across the field)
    ##################################################
    
    import astropy.units as u
    from .utils import (identify_extended_source,
                        cross_match_PS1,
                        calculate_color_term,
                        add_supplementary_SE_star)
    from urllib.error import HTTPError
    
    # Identify bright extended sources and enlarge their mask
    SE_cat_target, ext_cat = identify_extended_source(SE_cat, draw=draw)

    # Use PANSTARRS DR1 or DR2?
    if use_PS1_DR2:
        mag_name = mag_name_cat = b_name+'MeanPSFMag'
        bounds_crossmatch = bounds_list
        dir_name = os.path.join(work_dir, 'Measure-PS2/')
    else:
        mag_name = b_name+'mag'
        mag_name_cat = mag_name+'_PS'
        bounds_crossmatch = field_bounds
        dir_name = os.path.join(work_dir, 'Measure-PS1/')
    
    # Crossmatch with PANSTRRS mag < mag_limit
    tab_target, tab_target_full, catalog_star = \
                                cross_match_PS1(band, wcs_data,
                                                SE_cat_target,
                                                bounds_crossmatch,
                                                mag_limit=mag_limit,
                                                use_PS1_DR2=use_PS1_DR2)
   
   # Calculate color correction between PANSTARRS and DF filter
    CT = calculate_color_term(tab_target_full, mag_range=[mag_saturate,18],
                              mag_name=mag_name_cat, draw=draw)
    
    catalog_star["MAG_AUTO_corr"] = catalog_star[mag_name] + CT #corrected mag
    tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
   
    # Mannually add stars missed in the crossmatch or w/ weird mag to table
    tab_target = add_supplementary_SE_star(tab_target, SE_cat_target,
                                           mag_saturate, draw=draw)
                                            
    
    ##################################################
    # Save matched table and catalog
    ##################################################
    if save:
        check_save_path(dir_name, make_new=False, verbose=False)
        
        tab_target_name = os.path.join(dir_name,
       '%s-catalog_match_%smag%d.txt'%(obj_name, b_name, mag_limit))
        tab_target.write(tab_target_name,
                         overwrite=True, format='ascii')

        catalog_star_name = os.path.join(dir_name,
                 f'{obj_name}-catalog_PS_{b_name}_all.txt')
        catalog_star.write(catalog_star_name, 
                           overwrite=True, format='ascii')
        
        print(f'Save PANSTARRS catalog & matched sources in {dir_name}')
    
    
    ##################################################
    # Build Mask & Measure Scaling (in selected patch)
    ##################################################
    from .utils import (fit_empirical_aperture,
                        make_segm_from_catalog,
                        measure_Rnorm_all)
    from .plotting import plot_bright_star_profile
    
    # Empirical enlarged aperture size from magnitude based on matched SE detection
    estimate_radius = fit_empirical_aperture(tab_target_full, seg_map,
                                             mag_name=mag_name_cat,
                                             mag_range=[10,22], K=3,
                                             degree=2, draw=draw)
    
    for bounds in bounds_list:
        
        # Crop the star catalog and matched SE catalog
        patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = bounds
                                         
        # Catalog bound slightly wider than the region
        cat_bounds = (patch_Xmin-50, patch_Ymin-50,
                     patch_Xmax+50, patch_Ymax+50)

        catalog_star_patch = crop_catalog(catalog_star, cat_bounds,
                                          sortby=mag_name,
                                          keys=("X_IMAGE"+'_PS',
                                                "Y_IMAGE"+'_PS'))
        
        tab_target_patch = crop_catalog(tab_target, cat_bounds,
                                        sortby=mag_name_cat,
                                        keys=("X_IMAGE", "Y_IMAGE"))

        # Make segmentation map from catalog based on SE seg map of one band
        seg_map_c = make_segm_from_catalog(catalog_star_patch,
                                         bounds,
                                         estimate_radius,
                                         mag_name=mag_name,
                                         cat_name='PS',
                                         obj_name=obj_name,
                                         band=band,
                                         ext_cat=ext_cat,
                                         draw=draw,
                                         save=save,
                                         dir_name=dir_name)

        # Measure average intensity (source+background) at e_scale
        print("""Measure intensity at R = %d
                for catalog stars %s < %.1f in %r:"""\
              %(r_scale, mag_name, mag_limit, bounds))
        
        tab_norm, res_thumb = \
                    measure_Rnorm_all(tab_target_patch, bounds,
                                      wcs_data, data, seg_map,
                                      mag_limit=mag_limit,
                                      r_scale=r_scale, width=1,
                                      obj_name=obj_name,
                                      mag_name=mag_name_cat,
                                      save=save, dir_name=dir_name)
        if draw:
            plot_bright_star_profile(tab_target_patch,
                                     tab_norm, res_thumb,
                                     bkg_sky=bkg, std_sky=std, ZP=ZP,
                                     pixel_scale=pixel_scale)
        
        
def Run_PSF_Fitting(hdu_path,
                    bounds_list,
                    obj_name,
                    band,
                    pixel_scale=DF_pixel_scale,
                    ZP=None,
                    bkg=None,
                    G_eff=None,
                    pad=50,
                    r_scale=12,
                    mag_limit=15,
                    mag_threshold=[13.,11],
                    mask_type='aper',
                    mask_obj=None,
                    wid_strip=24,
                    n_strip=48,
                    SB_threshold=24.5,
                    n_spline=3,
                    r_core=24,
                    r_out=None,
                    cutoff=True,
                    n_cutoff=4,
                    theta_cutoff=1200,
                    n0=None,
                    fit_n0=True,
                    fit_sigma=True,
                    fit_frac=False,
                    leg2d=False,
                    draw_real=True,
                    brightest_only=False,
                    parallel=True,
                    n_cpu=None,
                    nlive_init=None,
                    sample_method='auto',
                    print_progress=True,
                    draw=True,
                    save=True,
                    stop=False,
                    use_PS1_DR2=False,
                    work_dir='./'):
    
    """
    
    Run the wide-angle PSF fitting.

    
    Parameters
    ----------
    
    hdu_path : str
        Full path of hdu data
    bounds_list : 2D int list / turple
        List of boundaries of regions to be fit (Nx4)
        [[X min, Y min, X max, Y max],[...],...]
    obj_name : str
        Object name
    band : str
        Filter name ('G', 'g' or 'R', 'r')
    pixel_scale : float, optional, default 2.5
        Pixel scale in arcsec/pixel
    ZP : float or None, optional, default None
        Zero point value (if None, read ZP from header)
    bkg : float or None, optional, default None
        Background estimated value (if None, read BACKVAL from header)
    G_eff : float or None (default)
        Effective gain (e-/ADU)
    pad : int, optional, default 50
        Padding size of the field for fitting
    r_scale : int, optional, default 12
        Radius (in pix) at which the brightness is measured.
        Default is 30" for Dragonfly.
    mag_limit : float, optional, default 15
        Magnitude upper limit below which are measured
    mag_threshold : [float, float], default: [14, 11]
        Magnitude theresholds to classify faint stars, medium bright stars and very bright stars.
        The conversion from brightness is using a static PSF. (* will change to stacked profiles)
    mask_type : 'aper' or 'brightness', optional, default 'aper'
        "aper": aperture masking
        "brightness": brightness-limit masking
    mask_obj : str, optional, default None
        full path to the target object mask (w/ the same shape with image)
    wid_strip : int, optional, default 24
        Width of strip for masks of very bright stars.
    n_strip : int, optional, default 48
        Number of strip for masks of very bright stars.
    SB_threshold : float, optional, default 24.5
        Surface brightness upper limit for masking.
        Only used if mask_type = 'brightness'.
    n_spline : int, optional, default 3
        Number of power-law component for the aureole models.
        The speed goes down as n_spline goes up. Default is 3.
    r_core : int or [int, int], optional, default 24
        Radius (in pix) for the inner mask of [very, medium]
        bright stars. Default is 1' for Dragonfly.
    r_out : int or [int, int] or None, optional, default None
        Radius (in pix) for the outer mask of [very, medium]
        bright stars. If None, turn off outer mask.
    cutoff : bool, optional, default True
        If True, the aureole will be cutoff at theta_cutoff.
    n_cutoff : float, optional, default 4
        Cutoff slope for the aureole model.
        Default is 4 for Dragonfly.
    theta_cutoff : float, optional, default 1200
        Cutoff range (in arcsec) for the aureole model.
        Default is 20' for Dragonfly.
    n0 : float, optional, default None
        Power index of the first component, only used if fit_n0=False.
        If not None, n0 will be fixed at that value in the prior.
    fit_n0 : bool, optional, default True
        Whether to fit n0 with profiles of bright stars in advance.
    fit_sigma : bool, optional, default False
        Whether to fit the background stddev.
        If False, will use the estimated value
    fit_frac : bool, optional, default False
        Whether to fit the fraction of the aureole.
        If False, use a static value.
        (* will change to values from stacked profiles)
    leg2d : bool, optional, default False
        Whether to fit a varied background with 2D Legendre polynomial.
        Currently only support 1st order.
    draw_real : bool, optional, default True
        Whether to draw very bright stars in real space.
        Recommended to be turned on.
    brightest_only : bool, optional, default False
        Whether to draw very bright stars only.
        If turned on the fitting will ignore medium bright stars.
    parallel : bool, optional, default True
        Whether to run drawing for medium bright stars in parallel.
    n_cpu : int, optional, default None
        Number of cpu used for fitting and/or drawing.
    nlive_init : int, optional, default None
        Number of initial live points in dynesty. If None will
        use nlive_init = ndim*10.
    sample_method : {'auto', 'unif', 'rwalk', 'rstagger', 'slice', 'rslice', 'hslice', callable},
                    optional, default is 'auto'
        Samplimg method in dynesty. If 'auto', the method is 'unif' for ndim < 10, 'rwalk' for
        10 <= ndim <= 20, 'slice' for ndim > 20.
    print_progress : bool, optional, default True
        Whether to turn on the progress bar of dynesty
    draw : bool, optional, default True
        Whether to draw diagnostic plots
    save : bool, optional, default True
        Whether to save results
    use_PS1_DR2 : bool, optional, default False
        Whether to use PANSTARRS DR2.
        Crossmatch with DR2 is done by MAST query, which might fail
        if a field is too large (> 1 deg^2)
    work_dir : str, optional, default current directory
        Full Path of directory for saving
        
        
    Returns
    -------
    samplers : list
        A list of Sampler class which contains fitting results.
        
    """
    
    plot_dir = os.path.join(work_dir, 'plot')
    check_save_path(plot_dir, make_new=False, verbose=False)
    
    if use_PS1_DR2:
        dir_measure = os.path.join(work_dir, 'Measure-PS2/')
    else:
        dir_measure = os.path.join(work_dir, 'Measure-PS1/')
    
    ############################################
    # Read Image and Table
    ############################################
    from .image import ImageList, DF_Gain
    
    # Read global background model ZP from header
    header = fits.getheader(hdu_path)
    if bkg is None: bkg = find_keyword_header(header, "BACKVAL")
    if ZP is None: ZP = find_keyword_header(header, "ZP")
    if G_eff is None:
        N_frames = find_keyword_header(header, "NFRAMES", default=1e5)
        G_eff = DF_Gain * N_frames
        if N_frames==1e5:
            print('Ineffective Gain. Will ignore short noise.')
        else:
            print('Effective Gain = %.3f'%G_eff)
    
    bounds_list = np.atleast_2d(bounds_list)
    
    # Construct Image List
    DF_Images = ImageList(hdu_path, bounds_list,
                          obj_name, band,
                          pixel_scale=pixel_scale,
                          pad=pad, ZP=ZP, bkg=bkg, G_eff=G_eff)
    
    # Read faint stars info and brightness measurement
    DF_Images.read_measurement_tables(dir_measure,
                                      r_scale=r_scale,
                                      mag_limit=mag_limit,
                                      use_PS1_DR2=use_PS1_DR2)
                                     
    ############################################
    # Setup Stars
    ############################################
    from .utils import assign_star_props
    stars_b, stars_all = DF_Images.assign_star_props(r_scale=r_scale,
                                                     mag_threshold=mag_threshold,
                                                     verbose=True, draw=False,
                                                     save=save, save_dir=plot_dir)
    # bright stars and all stars
    
    ############################################
    # Setup PSF
    ############################################
    from .modeling import PSF_Model
    
    # PSF Parameters (some from fitting stacked PSF)
    frac = 0.3              # fraction of aureole
    beta = 6.72             # moffat beta, in arcsec
    fwhm = 6.07             # moffat fwhm, in arcsec

    theta_0 = 5.                
    # radius in which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([3.4, 2.2, n_cutoff])         # initial guess
    theta_s = np.array([theta_0, 10**2., theta_cutoff])
        # initial transition radius in arcsec

    # Multi-power-law PSF
    params_mpow = {"fwhm":fwhm, "beta":beta, "frac":frac,
                   "n_s":n_s, "theta_s":theta_s, "cutoff":cutoff,
                   "n_c":n_cutoff, "theta_c":theta_cutoff}
    psf = PSF_Model(params=params_mpow,
                    aureole_model='multi-power')

    # Pixelize PSF
    psf.pixelize(pixel_scale)

    # Generate core and aureole PSF
    psf_c = psf.generate_core()
    psf_e, psf_size = psf.generate_aureole(contrast=1e6,
                                           psf_range=1000)
                   
    ############################################
    # Setup Basement Image
    ############################################
    # Make fixed background of dim stars
    DF_Images.make_base_image(psf.psf_star, stars_all, draw=False)
    
    ############################################
    # Masking
    ############################################
    from .mask import Mask
    
    if mask_type=='brightness':
        from .utils import SB2Intensity
        count = SB2Intensity(SB_threshold, DF_Images.bkg,
                             DF_Images.ZP, DF_Image.pixel_scale)[0]
    elif mask_type=='aper':
        count = None
        
    # Mask faint and centers of bright stars
#    r_max = int(theta_cutoff/pixel_scale)
    DF_Images.make_mask(stars_b, dir_measure,
                        by=mask_type, r_core=r_core, r_out=None,
                        wid_strip=wid_strip, n_strip=n_strip, dist_strip=None,
                        sn_thre=2.5, draw=draw, mask_obj=mask_obj,
                        save=save, save_dir=plot_dir)

    # Collect stars for fit. Choose if only use brightest stars
    if brightest_only:
        stars = [s.use_verybright() for s in DF_Images.stars]
    else:
        stars = DF_Images.stars # for fit

    ## (a stop for developer)
    if stop:
        print('Pause... Is the Mask Reasonable? [c/exit]')
        breakpoint()
    
    ############################################
    # Estimate Background & Fit n0
    ############################################
    DF_Images.estimate_bkg()
    if fit_n0:
        DF_Images.fit_n0(dir_measure, pixel_scale=pixel_scale,
                         r_scale=r_scale, mag_limit=mag_limit,
                         mag_max=12, draw=draw)
    else:
        DF_Images._n0 = n0  # fixed n0 if not None

    ############################################
    # Setup Priors and Likelihood Models for Fitting
    ############################################
    DF_Images.set_container(psf, stars,
                            n_spline=n_spline,
                            theta_in=50, theta_out=300,
                            n_min=1, leg2d=leg2d,
                            parallel=parallel,
                            draw_real=draw_real,
                            fit_sigma=fit_sigma,
                            fit_frac=fit_frac,
                            brightest_only=brightest_only)
    
    ############################################
    # Run Sampling
    ############################################
    from .sampler import Sampler
    from .io import DateToday, AsciiUpper, save_pickle
    
    samplers = []
    
    for i, reg in enumerate(AsciiUpper(DF_Images.N_Image)):

        ct = DF_Images.containers[i]
        ndim = ct.ndim

        s = Sampler(ct, n_cpu=n_cpu, sample=sample_method)
                                  
        if nlive_init is None: nlive_init = ndim*10
        # Run dynesty
        s.run_fitting(nlive_init=nlive_init,
                      nlive_batch=5*ndim+5, maxbatch=2,
                      print_progress=print_progress)
    
        if save:
            # Save outputs
            s.fit_info = {'obj_name':obj_name,
                          'band':band.lower(),
                          'n_spline':n_spline,
                          'bounds':bounds_list[i],
                          'r_scale':r_scale,
                          'n_cutoff': n_cutoff,
                          'theta_cutoff': theta_cutoff,
                          'cutoff': cutoff,
                          'fit_n0':fit_n0,
                          'date':DateToday()}
                        
            suffix = str(n_spline)+'p'
            if leg2d: suffix+='l'
            if fit_frac: suffix+='f'
            if brightest_only: suffix += 'b'
            if use_PS1_DR2: suffix += '_ps2'
            
            Xmin, Ymin, Xmax, Ymax = bounds_list[i]
            
            fname = f'{obj_name}-{reg}-X[{Xmin}-{Xmax}]Y[{Ymin}-{Ymax}]-{band}-fit{suffix}.res'
    
            s.save_results(fname, save_dir=work_dir)
            stars[i].save(f'stars-{reg}-X[{Xmin}-{Xmax}]Y[{Ymin}-{Ymax}]-{band.upper()}', save_dir=work_dir)
        
        ############################################
        # Plot Results
        ############################################
        from .plotting import AsinhNorm
        
        suffix = str(n_spline)+'p'+'_'+obj_name
        
        # Recovered 1D PSF
        s.generate_fit(psf, stars[i])
        
        if draw:
            s.cornerplot(figsize=(18, 16),
                         save=save, save_dir=plot_dir, suffix=suffix)

            # Plot recovered PSF
            s.plot_fit_PSF1D(psf, n_bootstrap=500, r_core=r_core,
                             save=save, save_dir=plot_dir, suffix=suffix)

            # Calculate Chi^2
            s.calculate_reduced_chi2(Gain=G_eff, dof=ndim)

            # Draw 2D compaison
            s.draw_comparison_2D(r_core=r_core, Gain=G_eff, 
                                 vmin=DF_Images.bkg-s.bkg_std_fit,
                                 vmax=DF_Images.bkg+20*s.bkg_std_fit,
                                 save=save, save_dir=plot_dir, suffix=suffix)

            if leg2d:
                # Draw background
                s.draw_background(save=save, save_dir=plot_dir,
                                  suffix=suffix)
        
        # Append the sampler
        samplers += [s]
        
    # Delete Stars to avoid pickling error in rerun
    for variable in dir():
        if 'stars' in variable:
            del locals()[variable]
        
    return samplers
    
    

class berry:
    
    """
    
    Fruit of elderflower.
    A class wrapper for running the package.
    
    Parameters
    ----------
    
    hdu_path : str
        path of hdu data
    bounds_list : list [[X min, Y min, X max, Y max],[...],...]
        list of boundaries of regions to be fit (Nx4)
    obj_name : str
        object name
    band : str
        filter name
    work_dir : str, optional, default current directory
        Full Path of directory for saving
    config_file : yaml, optional, default None
        configuration file which contains keyword arguments.
        If None, use the default configuration file.
        
        
    Example
    -------
        
    # Initialize the task
        from elderflower.task import berry
        
        elder = berry(hdu_path, bounds, obj_name, filt='g', work_dir='...', config_file='...')
                  
    # Check keyword parameters listed in the configuration:
        elder.parameters
    
    # Run the task
        elder.detection()

        elder.run()
        
    """

    def __init__(self,
                 hdu_path,
                 bounds_list,
                 obj_name,
                 band,
                 work_dir='./',
                 config_file=None):
        
        self.hdu_path = hdu_path
        self.bounds_list = bounds_list
        self.obj_name = obj_name
        self.band = band
        
        self.work_dir = work_dir
        
        from .io import config_kwargs, default_config
        if config_file is None: config_file = default_config
        self.config_func = partial(config_kwargs, config_file=config_file)
    

    @property
    def parameters(self):
        """ Keyword parameter list in the configuration file """
        @self.config_func
        def _kwargs(**kwargs):
            return kwargs
            
        return _kwargs()
    
    def detection(self, **kwargs):
        """ Run the source detection. """
        
        self.ZP = Run_Detection(self.hdu_path,
                                self.obj_name, self.band,
                                work_dir=self.work_dir,
                                FILTER_NAME=default_SE_conv,
                                STARNNW_NAME=default_SE_nnw, **kwargs)
        
    def run(self, **kwargs):
        """ Run the task (Match_Mask_Measure + Run_PSF_Fitting). """
            
        @self.config_func
        def _run(func, **kwargs):
            keys = set(kwargs.keys()).intersection(func.__code__.co_varnames)
            pars = {key: kwargs[key] for key in keys}
            
            return func(self.hdu_path, self.bounds_list,
                        self.obj_name, self.band,
                        work_dir=self.work_dir, **pars)
                        
        _run(Match_Mask_Measure, **kwargs)
        self.samplers = _run(Run_PSF_Fitting, **kwargs)
