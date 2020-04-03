#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

def Match_Mask_Measure(hdu_path, image_bounds,
                       SE_segmap, SE_catalog,
                       weight_map=None,
                       obj_name='', band="G",
                       pixel_scale=2.5,
                       ZP=None, field_pad=500,
                       r_scale=12, mag_thre=15,
                       draw=True, save=True,
                       use_PS1_DR2=False,
                       dir_name='../output/Measure'):
    
    print("""Measure the intensity at R = %d for stars < %.1f
            as normalization of fitting\n"""%(r_scale, mag_thre))
    
    b_name = band.lower()
    image_bounds = np.atleast_2d(image_bounds)
    
    ##################################################
    # Read and Display
    ##################################################
    from .utils import crop_image, crop_catalog
    from .utils import find_keyword_header
    from astropy.stats import mad_std
    from astropy.table import Table
    from astropy.io import fits
    from astropy import wcs
    
    # Read hdu
    if os.path.isfile(hdu_path) is False:
        sys.exit("Image does not exist. Check path.")
        
    with fits.open(hdu_path) as hdul:
        print("Read Image :", hdu_path)
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)

    # Read output from create_photometric_light_APASS 
    if os.path.isfile(SE_segmap):
        seg_map = fits.getdata(SE_segmap)
    else:
        seg_map = None
        
    SE_cat_full = Table.read(SE_catalog, format="ascii.sextractor")
    
    if weight_map is not None:
        weight_edge = fits.getdata(weight_map)
    else:
        weight_edge = np.ones_like(data)
     
    # Read global background model ZP and pixel scale from header
    
    bkg = find_keyword_header(header, "BACKVAL")
    if ZP is None:
        ZP = find_keyword_header(header, "ZP")
    
    # Estimate of background fluctuation (just for plot)
    if seg_map is not None:
        std = mad_std(data[(seg_map==0) & (weight_edge>0.5)]) 
    else:
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
    
    if not use_PS1_DR2:
        print("Match field %r with catalog\n"%field_bounds)
    
    print("Measure Sky Patch (X min, Y min, X max, Y max) :")
    [print("%r"%b) for b in image_bounds.tolist()]
    
    # Display field_bounds and sub-regions to be matched
    patch, seg_patch = crop_image(data, field_bounds, seg_map,
                                  weight_map=weight_edge,
                                  sub_bounds=image_bounds,
                                  origin=0, draw=draw)
    

    ##################################################
    # Crossmatch with Star Catalog (across the field)
    ##################################################
    from .utils import cross_match_PS1_DR2, cross_match
    from .utils import calculate_color_term

    # Crossmatch with PANSTRRS at threshold of mag_thre mag
    if use_PS1_DR2:
        # Give 3 attempts in matching PS1 DR2 via MAST.
        # This could fail if the FoV is too large.
        n_attempts = 0
        while n_attempts < 3:
            try:
                tab_target, tab_target_full, catalog_star = \
                            cross_match_PS1_DR2(wcs_data,
                                                SE_cat_full,
                                                image_bounds,
                                                mag_thre=mag_thre,
                                                band=b_name) 
                break
                
            except HTTPError:
                attempts += 1
                print('Gateway Time-out. Try Again.')
                
        if n_attempts>=3:
            sys.exit('504 Server Error: 3 Failed Attempts. Exit.')
            
    else:
        mag_name = b_name+'mag'
        tab_target, tab_target_full, catalog_star = \
                            cross_match(wcs_data,
                                        SE_cat_full,
                                        field_bounds,
                                        mag_thre=mag_thre, 
                                        mag_name=mag_name)
        

    # Calculate color correction between PANSTARRS and DF filter
    if use_PS1_DR2:
        mag_name = mag_name_cat = b_name+'MeanPSFMag'
    else:
        mag_name_cat = mag_name+'_PS'
        
    CT = calculate_color_term(tab_target_full,
                              mag_name=mag_name_cat, draw=draw)
    
    catalog_star["MAG_AUTO"] = catalog_star[mag_name] + CT
    
    # Save matched table and catalog
    if save:
        tab_target_name = os.path.join(dir_name,
       '%s-catalog_match_%smag%d.txt'%(obj_name, b_name, mag_thre))
        
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
        
        tab_target.write(tab_target_name,
                         overwrite=True, format='ascii')

        catalog_star_name = os.path.join(dir_name,
                 '%s-catalog_PS_%s_all.txt'%(obj_name, b_name))
        
        catalog_star["FLUX_AUTO"] = 10**((catalog_star["MAG_AUTO"]-ZP)/(-2.5))
        
        catalog_star.write(catalog_star_name, 
                           overwrite=True, format='ascii')
        
        print('Save PANSTARRS catalog & matched sources in %s'%dir_name)
        
    ##################################################
    # Build Mask & Measure Scaling (in selected patch)
    ##################################################
    from .utils import fit_empirical_aperture, make_segm_from_catalog
    from .utils import measure_Rnorm_all
    from .plotting import plot_bright_star_profile
    
    # Empirical enlarged aperture size from magnitude based on matched SE detection
    estimate_radius = fit_empirical_aperture(tab_target_full, seg_map,
                                             mag_name=mag_name_cat,
                                             mag_range=[13,22], K=2.5,
                                             degree=3, draw=draw)
    
    for image_bound in image_bounds:
        
        # Crop the star catalog and matched SE catalog
        patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = image_bound
                                         
        # Catalog bound slightly wider than the region
        cat_bound = (patch_Xmin-50, patch_Ymin-50,
                     patch_Xmax+50, patch_Ymax+50)

        catalog_star_patch = crop_catalog(catalog_star, cat_bound,
                                          sortby=mag_name,
                                          keys=("X_IMAGE"+'_PS',
                                                "Y_IMAGE"+'_PS'))
        
        tab_target_patch = crop_catalog(tab_target, cat_bound,
                                        sortby=mag_name_cat,
                                        keys=("X_IMAGE", "Y_IMAGE"))

        # Make segmentation map from catalog based on SE seg map of one band
        seg_map_cat = make_segm_from_catalog(catalog_star_patch,
                                             image_bound,
                                             estimate_radius,
                                             mag_name=mag_name,
                                             cat_name='PS',
                                             obj_name=obj_name,
                                             band=band, draw=draw,
                                             save=save, dir_name=dir_name)

        # Measure average intensity (source+background) at e_scale
        print("""Measure intensity at R = %d
                for catalog stars %s < %.1f in %r:"""\
              %(r_scale, mag_name, mag_thre, image_bound))
        
        tab_res_Rnorm, res_thumb = \
                measure_Rnorm_all(tab_target_patch, image_bound,
                                  wcs_data, data, seg_map,
                                  mag_thre=mag_thre,
                                  r_scale=r_scale, width=1, 
                                  obj_name=obj_name,
                                  mag_name=mag_name_cat, 
                                  save=save, dir_name=dir_name)
        
        plot_bright_star_profile(tab_target_patch,
                                 tab_res_Rnorm, res_thumb,
                                 bkg_sky=bkg, std_sky=std, ZP=ZP,
                                 pixel_scale=pixel_scale)
        
        
        
def Run_PSF_Fitting(hdu_path, image_bounds0,
                    n_spline=2, obj_name='', band="G", 
                    pixel_scale=2.5, ZP=None, pad=100,  
                    r_scale=12, mag_threshold=[14,11], 
                    mask_type='radius', SB_fit_thre=24.5,
                    r_core=24, r_out=None,
                    fit_sigma=True, fit_frac=False, leg2d=False,
                    wid_strip=24, n_strip=48, 
                    n_cpu=None, parallel=False, 
                    brightest_only=False, draw_real=True,
                    draw=True, print_progress=True,
                    save=False, dir_name='./',
                    dir_measure='../output/Measure-PS'):
    
    ############################################
    # Read Image and Table
    ############################################
    from .image import ImageList
    DF_Images = ImageList(hdu_path, image_bounds0,
                          obj_name, band,
                          pixel_scale, ZP, pad)
    
    from .utils import read_measurement_tables
    tables_faint, tables_res_Rnorm = \
                read_measurement_tables(dir_measure,
                                        image_bounds0,
                                        obj_name=obj_name,
                                        band=band,
                                        pad=pad,
                                        r_scale=r_scale)
    
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
    theta_s = np.array([theta_0, 10**1.9, 1200])  
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
    psf.pixelize(pixel_scale=pixel_scale)

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
                                 save=save, save_dir=dir_name)
    
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
        count = SB2Intensity(SB_fit_thre, DF_Images.bkg,
                             DF_Images.ZP, DF_Image.pixel_scale)[0]
    else:
        count = None
        
    # Mask faint and centers of bright stars
    DF_Images.make_mask(stars_0, dir_measure,
                        by=mask_type, r_core=r_core, r_out=None,
                        wid_strip=wid_strip, n_strip=n_strip,
                        sn_thre=2.5, n_dilation=5, draw=True,
                        save=save, save_dir=dir_name)

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
#                         'image_bounds0':image_bounds0, 'leg2d':leg2d,
#                         'r_core':r_core, 'r_scale':r_scale}

#             method = str(n_spline)+'p'
#             fname='NGC5907-%s-fit_best_X%dY%d_%s'\
#                         %(band, image_bounds0[0], image_bounds0[1], method)
#             if leg2d: fname+='l'
#             if brightest_only: fname += 'b'

#             ds.save_results(fname+'.res', fit_info, save_dir=dir_name)
        
        ############################################
        # Plot Results
        ############################################
        from .plotting import AsinhNorm
        method = str(n_spline)+'p'
        
        ds.cornerplot(figsize=(18, 16),
                      save=save, save_dir=dir_name,
                      suffix='_'+method)

        # Plot recovered PSF
        ds.plot_fit_PSF1D(psf, n_bootstrap=500, r_core=r_core,
                          save=save, save_dir=dir_name,
                          suffix='_'+method)

        # Recovered 1D PSF
        psf_fit, params = ds.generate_fit(psf, stars_tri[i],
                                          n_out=4, theta_out=1200)

        # Calculate Chi^2
        ds.calculate_reduced_chi2()

        # Draw 2D compaison
        ds.draw_comparison_2D(r_core=r_core,
                              norm=AsinhNorm(a=0.01),
                              vmin=DF_Images.bkg-2,
                              vmax=DF_Images.bkg+50, 
                              save=save, save_dir=dir_name,
                              suffix='_'+method)

        if leg2d:
            ds.draw_background(save=save, save_dir=dir_name,
                               suffix='_'+method)

        dsamplers += [ds]
        
    return dsamplers