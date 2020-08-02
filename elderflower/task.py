#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from .utils import find_keyword_header, check_save_path

from astropy.table import Table
from astropy.io import fits
import astropy.units as u

SE_params = ['NUMBER','X_WORLD','Y_WORLD','FLUXERR_AUTO','MAG_AUTO','MU_MAX','CLASS_STAR','ELLIPTICITY']
SE_executable = '/opt/local/bin/source-extractor'

apass_dir = '/Users/qliu/Data/apass/'

def Run_Detection(hdu_path, obj_name, filt='g',
                  threshold=3, work_dir='./',
                  ZP_keyname='REF_ZP', ZP=None,
                  ref_cat='APASSref.cat',
                  apass_dir=apass_dir,
                  config_path='default.sex',
                  executable=SE_executable, **kwargs):
                
    from dfreduce.detection import sextractor
    
    print(f"Run SExtractor on {hdu_path}...")
    
    segname = os.path.join(work_dir, f'{obj_name}_seg.fits')
    catname = os.path.join(work_dir, f'{obj_name}.cat')
    
    header = fits.getheader(hdu_path)
    
    # Find zero-point in the fits header
    if ZP_keyname not in header.keys():
    
        # If not in the header, check kwargs
        if type(ZP) is not float:
            
            # If not available in kwargs, compute by crossmatch with refcat
            from dfreduce.utils.catalogues import match_catalogues
            print("Compute zero-point from crossmatch with APASS catalog...")
            
            # Run sextractor with free zero-point
            SE_catalog = sextractor.run(hdu_path,
                                        extra_params=SE_params,
                                        config_path=config_path,
                                        catalog_path=catname,
                                        executable=executable,
                                        DETECT_THRESH=5,
                                        ANALYSIS_THRESH=5)
                                        
            # Load (APASS) reference catalog
            if os.path.exists(ref_cat):
                refcat = Table.read(ref_cat, format='ascii')
            else:
                print("Generate APASS reference catalog... It takes time.")
                ra_range = header['CRPIX1'] * header['CD1_1']
                dec_range = header['CRPIX2'] * header['CD2_2']

                minra, maxra = sorted([header['CRVAL1'] - ra_range, header['CRVAL1'] + ra_range])
                mindec, maxdec = sorted([header['CRVAL2'] - dec_range, header['CRVAL2'] + dec_range])

                refcat = catalogues.load_apass_in_region(apass_dir,
                                                         bounds=[mindec, maxdec, minra, maxra])
                refcat.write(ref_cat, format='ascii')

            # Crossmatch SE catalog with reference catalog
            imagecat_match, refcat_match = match_catalogues(SE_catalog, refcat, filt, sep_max=3.)
            
            # Get the mean ZP from the crossmatched catalog
            ZP = np.mean(refcat_match[filt] - imagecat_match[filt])
            print("Matched zero-point = {:.3f}".format(ZP))
        
    else:
        ZP = np.float(header['REFZP'])
        print("Read zero-point from header : ZP = {:.3f}".format(ZP))
    
    SE_catalog = sextractor.run(hdu_path,
                                extra_params=SE_params,
                                config_path=config_path,
                                catalog_path=catname,
                                executable=executable,
                                DETECT_THRESH=threshold,
                                ANALYSIS_THRESH=threshold,
                                CHECKIMAGE_TYPE='SEGMENTATION',
                                CHECKIMAGE_NAME=segname,
                                MAG_ZEROPOINT=ZP, **kwargs)
                                
    print(f"CATALOG saved as {catname}")
    print(f"SEGMENTATION saved as {segname}")


def Match_Mask_Measure(hdu_path, bounds_list,
                       obj_name='', band="G",
                       pixel_scale=2.5,
                       ZP=None,bkg=None,field_pad=500,
                       r_scale=12, mag_limit=15, mag_saturate=13,
                       draw=True, save=True,
                       use_PS1_DR2=False,
                       work_dir='./'):
    
    print("""Measure the intensity at R = %d for stars < %.1f
            as normalization of fitting\n"""%(r_scale, mag_limit))
    
    b_name = band.lower()
    bounds_list = np.atleast_2d(bounds_list)
    
    ##################################################
    # Read and Display
    ##################################################
    from .utils import crop_image, crop_catalog
    from astropy.stats import mad_std
    from astropy.table import setdiff, join
    from astropy import wcs
    
    # Read hdu
    if os.path.isfile(hdu_path) is False:
        sys.exit("Image does not exist. Check path.")
        
    with fits.open(hdu_path) as hdul:
        print("Read Image :", hdu_path)
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)

    # Read output from SExtractor detection
    SE_cat_full = Table.read(os.path.join(work_dir, f'{obj_name}.cat'), format="ascii.sextractor")
    
    seg_map = fits.getdata(os.path.join(work_dir, f'{obj_name}_seg.fits'))

     
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
    patch, _ = crop_image(data, field_bounds, seg_map,
                          sub_bounds=bounds_list,
                          origin=0, draw=draw)
    
    # Crop parent SE catalog
    SE_cat_field = crop_catalog(SE_cat_full, field_bounds)

    ##################################################
    # Crossmatch with Star Catalog (across the field)
    ##################################################
    
    from .utils import identify_extended_source
    from .utils import cross_match_PS1_DR2, cross_match
    from .utils import calculate_color_term
    from .utils import add_supplementary_SE_star
    from urllib.error import HTTPError
    
    # Identify bright extended sources and enlarge their mask
    ext_cat = identify_extended_source(SE_cat_field)
    SE_cat_target = setdiff(SE_cat_field, ext_cat)

    # Crossmatch with PANSTRRS mag < mag_limit
    if use_PS1_DR2:
        # Give 3 attempts in matching PS1 DR2 via MAST.
        # This could fail if the FoV is too large.
        for attempt in range(4):
            try:
                tab_target, tab_target_full, catalog_star = \
                            cross_match_PS1_DR2(wcs_data,
                                                SE_cat_target,
                                                bounds_list,
                                                sep=3*u.arcsec,
                                                mag_limit=mag_limit,
                                                band=b_name) 
            except HTTPError:
                print('Gateway Time-out. Try Again.')
            else:
                break
        else:
            sys.exit('504 Server Error: 4 Failed Attempts. Exit.')
            
    else:
        mag_name = b_name+'mag'
        tab_target, tab_target_full, catalog_star = \
                            cross_match(wcs_data,
                                        SE_cat_target,
                                        field_bounds,
                                        sep=3*u.arcsec,
                                        mag_limit=mag_limit,
                                        mag_name=mag_name)
        

    # Use PANSTARRS DR1 or DR2?
    if use_PS1_DR2:
        mag_name = mag_name_cat = b_name+'MeanPSFMag'
        dir_name = os.path.join(work_dir, 'Measure-PS2/')
    else:
        mag_name_cat = mag_name+'_PS'
        dir_name = os.path.join(work_dir, 'Measure-PS1/')
   
   # Calculate color correction between PANSTARRS and DF filter
    CT = calculate_color_term(tab_target_full, mag_range=[mag_saturate,18],
                              mag_name=mag_name_cat, draw=draw)
    
    catalog_star["MAG_AUTO_corr"] = catalog_star[mag_name] + CT #corrected mag
    tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
   
    # Mannually add stars missed in the crossmatch or w/ weird mag to table
    tab_target = add_supplementary_SE_star(tab_target, SE_cat_target,
                                            mag_saturate, draw=draw)
    
    # Save matched table and catalog
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
    from .utils import fit_empirical_aperture, make_segm_from_catalog
    from .utils import measure_Rnorm_all
    from .plotting import plot_bright_star_profile
    
    # Empirical enlarged aperture size from magnitude based on matched SE detection
    estimate_radius = fit_empirical_aperture(tab_target_full, seg_map,
                                             mag_name=mag_name_cat,
                                             mag_range=[mag_saturate,22], K=2.5,
                                             degree=3, draw=False)
    
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
        seg_map_cat = make_segm_from_catalog(catalog_star_patch,
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
        
        tab_res_Rnorm, res_thumb = \
                        measure_Rnorm_all(tab_target_patch, bounds,
                                          wcs_data, data, seg_map,
                                          mag_limit=mag_limit,
                                          r_scale=r_scale, width=1,
                                          obj_name=obj_name,
                                          mag_name=mag_name_cat,
                                          save=save, dir_name=dir_name)
        
        plot_bright_star_profile(tab_target_patch,
                                 tab_res_Rnorm, res_thumb,
                                 bkg_sky=bkg, std_sky=std, ZP=ZP,
                                 pixel_scale=pixel_scale)
        
        
        
def Run_PSF_Fitting(hdu_path, bounds0,
                    obj_name='DFfield', band="G",
                    n_spline=2, work_dir='./', use_PS1_DR2=True,
                    pixel_scale=2.5, ZP=None, bkg=None, pad=100,
                    r_scale=12, mag_limit=15, mag_threshold=[14,11],
                    mask_type='radius', SB_fit_thre=24.5,
                    r_core=24, r_out=None, theta_cutoff=1200,
                    fit_sigma=True, fit_frac=False, leg2d=False,
                    wid_strip=24, n_strip=48, 
                    n_cpu=None, parallel=False, 
                    brightest_only=False, draw_real=True,
                    draw=True, save=False,
                    print_progress=True):
    
    ############################################
    # Read Image and Table
    ############################################
    from .image import ImageList
    from .utils import read_measurement_tables
    
    # Read global background model ZP from header
    header = fits.getheader(hdu_path)
    if bkg is None: bkg = find_keyword_header(header, "BACKVAL")
    if ZP is None: ZP = find_keyword_header(header, "ZP")
    
    # Construct Image List
    DF_Images = ImageList(hdu_path, bounds0,
                          obj_name, band,
                          pixel_scale, ZP, bkg, pad)
    
    # Read faint stars info and brightness measurement
    if use_PS1_DR2:
        dir_measure = os.path.join(work_dir, 'Measure-PS2/')
    else:
        dir_measure = os.path.join(work_dir, 'Measure-PS1/')
        
    tables_faint, tables_res_Rnorm = \
                read_measurement_tables(dir_measure,
                                        bounds0,
                                        obj_name=obj_name,
                                        band=band, pad=pad,
                                        r_scale=r_scale,
                                        mag_limit=mag_limit,
                                        use_PS1_DR2=use_PS1_DR2)
    
    ############################################
    # Setup PSF
    ############################################
    from .modeling import PSF_Model
    
    # PSF Parameters (some from fitting stacked PSF)
    frac = 0.3                  # fraction of aureole
    beta = 10                   # moffat beta, in arcsec
    fwhm = 2.3 * pixel_scale    # moffat fwhm, in arcsec

    n0 = 3.2                    # estimated true power index
    theta_0 = 5.                
    # radius in which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([n0, 2., 4])         # power index
    theta_s = np.array([theta_0, 10**1.9, theta_cutoff])
        # transition radius in arcsec
    
    if n_spline == 1:
        # Single-power PSF
        params_pow = {"fwhm":fwhm, "beta":beta,
                      "frac":frac, "n":n0, 'theta_0':theta_0}
        psf = PSF_Model(params=params_pow,
                        aureole_model='power')
        
    else:
        # Multi-power PSF
        params_mpow = {"fwhm":fwhm, "beta":beta,
                       "frac":frac, "n_s":n_s, 'theta_s':theta_s}
        psf = PSF_Model(params=params_mpow,
                        aureole_model='multi-power')

    # Pixelize PSF
    psf.pixelize(pixel_scale)

    # Generate core and aureole PSF
    psf_c = psf.generate_core()
    psf_e, psf_size = psf.generate_aureole(contrast=1e6,
                                           psf_range=1000)
                                           
    # Deep copy
    psf_tri = psf.copy()
    
    ############################################
    # Setup Stars
    ############################################    
    from .utils import assign_star_props
    
    stars_0, stars_all = \
         DF_Images.assign_star_props(tables_faint,
                                     tables_res_Rnorm,
                                     r_scale=r_scale,
                                     mag_threshold=mag_threshold,
                                     verbose=True, draw=False,
                                     save=save, save_dir=work_dir)
    
    #breakpoint()
    
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
        count = SB2Intensity(SB_fit_thre, DF_Images.bkg_val,
                             DF_Images.zp_val, DF_Image.pixel_scale)[0]
    else:
        count = None
        
    # Mask faint and centers of bright stars
    DF_Images.make_mask(stars_0, dir_measure,
                        by=mask_type, r_core=r_core, r_out=None,
                        wid_strip=wid_strip, n_strip=n_strip,
                        sn_thre=2.5, n_dilation=5, draw=True,
                        save=save, save_dir=work_dir)

    # Collect stars for fit. Choose if only use brightest stars
    if brightest_only:
        stars = [s.use_verybright() for s in DF_Images.stars]
    else:
        stars = DF_Images.stars # for fit

    # Copy stars
    stars_tri = stars.copy()

    proceed = input('Is the Mask Reasonable?[y/n]')
    if proceed == 'n': sys.exit("Reset the Mask.")
    
    ############################################
    # Estimate Background
    ############################################
    DF_Images.estimate_bkg()

    ############################################
    # Setup Priors and Likelihood Models for Fitting
    ############################################
    DF_Images.set_container(psf_tri, stars_tri, 
                            n_spline=n_spline,
                            n_min=1, n_est=n0,
                            theta_in=50, theta_out=240,
                            theta_cutoff=theta_cutoff,
                            leg2d=leg2d, parallel=parallel,
                            draw_real=draw_real,
                            fit_sigma=fit_sigma,
                            fit_frac=fit_frac,
                            brightest_only=brightest_only)

    ############################################
    # Run Sampling
    ############################################
    from .sampler import DynamicNestedSampler
    
    dsamplers = []
    
    for i in range(DF_Images.N_Image):
        
        container = DF_Images.containers[i]
        
        ndim = container.ndim

        ds = DynamicNestedSampler(container,
                                  sample='auto', n_cpu=n_cpu)
        
        ds.run_fitting(nlive_init=ndim*10,
                       nlive_batch=2*ndim+2, maxbatch=2,
                       print_progress=print_progress)
    
#         if save:
#             fit_info = {'n_spline':n_spline, 'image_size':image_size,
#                         'bounds0':bounds0, 'leg2d':leg2d,
#                         'r_core':r_core, 'r_scale':r_scale}

#             method = str(n_spline)+'p'
#             fname='NGC5907-%s-fit_best_X%dY%d_%s'\
#                         %(band, bounds0[0], bounds0[1], method)
#             if leg2d: fname+='l'
#             if brightest_only: fname += 'b'

#             ds.save_results(fname+'.res', fit_info, save_dir=work_dir)
        
        ############################################
        # Plot Results
        ############################################
        from .plotting import AsinhNorm
        method = str(n_spline)+'p'
        
        ds.cornerplot(figsize=(18, 16),
                      save=save, save_dir=work_dir,
                      suffix='_'+method)

        # Plot recovered PSF
        ds.plot_fit_PSF1D(psf, n_bootstrap=500, r_core=r_core,
                          save=save, save_dir=work_dir,
                          theta_cutoff=theta_cutoff, suffix='_'+method)

        # Recovered 1D PSF
        psf_fit, params = ds.generate_fit(psf, stars_tri[i],
                                          n_cutoff=4, theta_cutoff=theta_cutoff)

        # Calculate Chi^2
        ds.calculate_reduced_chi2()

        # Draw 2D compaison
        ds.draw_comparison_2D(r_core=r_core,
                              norm=AsinhNorm(a=0.01),
                              vmin=DF_Images.bkg-2,
                              vmax=DF_Images.bkg+50, 
                              save=save, save_dir=work_dir,
                              suffix='_'+method)

        if leg2d:
            ds.draw_background(save=save, save_dir=work_dir,
                               suffix='_'+method)

        dsamplers += [ds]
        
    
    # Delete Stars to avoid pickling error in rerun
    for variable in dir():
        if 'stars' in variable:
            del locals()[variable]
        
    return dsamplers
