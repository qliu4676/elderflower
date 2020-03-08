#!/usr/bin/env python
# -*- coding: utf-8 -*-

def Run_PSF_Fitting(hdu_path, image_bounds0,
                    n_spline=2, band="G",
                    r_scale=12, mag_threshold=[14,11], 
                    mask_type='radius', SB_fit_thre=24.5,
                    r_core=24, r_out=None,
                    fit_frac=False, leg2d=False,
                    pad=100, pixel_scale=2.5, 
                    wid_strip=24, n_strip=48, 
                    n_cpu=None, parallel=False, 
                    brightest_only=True, draw_real=True,
                    draw=True, print_progress=True,
                    save=False, dir_name='./',
                    dir_measure='./Measure'):
    
    ############################################
    # Read Image and Table
    ############################################
    from src.image import ImageList
    DF_Images = ImageList(hdu_path, image_bounds0, pixel_scale, pad)
    
    from src.utils import read_measurement_tables
    tables_faint, tables_res_Rnorm = read_measurement_tables(dir_measure,
                                                             image_bounds0)
    
    ############################################
    # Setup PSF
    ############################################
    from src.modeling import PSF_Model
    
    # PSF Parameters (some from fitting stacked PSF)
    frac = 0.3                  # fraction of aureole
    beta = 10                   # moffat beta, in arcsec
    fwhm = 2.3 * pixel_scale    # moffat fwhm, in arcsec

    n0 = 3.2                    # estimated true power index
    theta_0 = 5.                # radius at which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([n0, 2.26, 1.31, 4])                         # power index
    theta_s = np.array([theta_0, 10**1.88, 10**2.2, 1200])      # transition radius in arcsec
    
    if n_spline == 1:
        # Single-power PSF
        params_pow = {"fwhm":fwhm, "beta":beta,
                      "frac":frac, "n":n0, 'theta_0':theta_0}
        psf = PSF_Model(params=params_pow, aureole_model='power')
        
    else:
        # Multi-power PSF
        params_mpow = {"fwhm":fwhm, "beta":beta,
                       "frac":frac, "n_s":n_s, 'theta_s':theta_s}
        psf = PSF_Model(params=params_mpow, aureole_model='multi-power')

    # Pixelize PSF
    psf.pixelize(pixel_scale=pixel_scale)

    # Generate core and aureole PSF
    psf_c = psf.generate_core()
    psf_e, psf_size = psf.generate_aureole(contrast=1e6, psf_range=1000)

    # Galsim 2D model averaged in 1D
    psf.plot_PSF_model_galsim()

    # Deep copy
    psf_tri = psf.copy()
    
    ############################################
    # Setup Stars
    ############################################    
    from src.utils import assign_star_props
    
    stars_0, stars_all = DF_Images.assign_star_props(tables_faint,
                                                     tables_res_Rnorm, 
                                                     r_scale=r_scale,
                                                     mag_threshold=mag_threshold,
                                                     verbose=True, draw=True,
                                                     save=save, save_dir=dir_name)
    
    ############################################
    # Setup Basement Image
    ############################################
    # Make fixed background of dim stars
    DF_Images.make_base_image(psf.psf_star, stars_all)
    
    ############################################
    # Masking
    ############################################
    from src.mask import Mask
    
    if mask_type=='brightness':
        from src.utils import SB2Intensity
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
        stars = DF_Images.stars.use_verybright()
    else:
        stars = DF_Images.stars # for fit

    # Copy stars
    stars_tri = stars.copy()

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
    dsamplers = []
    
    for i in range(DF_Images.N_Image):
        
        container = DF_Images.containers[i]
        
        ndim = container.ndim

        ds = DynamicNestedSampler(container, sample='auto', n_cpu=n_cpu)
        
        ds.run_fitting(nlive_init=ndim*10, nlive_batch=2*ndim+2,
                       maxbatch=2, print_progress=print_progress)
    
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

        ds.cornerplot(figsize=(18, 16),
                      save=save, save_dir=dir_name, suffix='_'+method)

        # Plot recovered PSF
        ds.plot_fit_PSF1D(psf, n_bootstrap=500, r_core=r_core, leg2d=leg2d,
                          save=save, save_dir=dir_name, suffix='_'+method)

        # Recovered 1D PSF
        psf_fit, params = ds.generate_fit(psf, stars_tri[i],
                                          n_out=4, theta_out=1200)

        # Calculate Chi^2
        ds.calculate_reduced_chi2()

        # Draw 2D compaison
        ds.draw_comparison_2D(r_core=r_core, norm=AsinhNorm(),
                              vmin=DF_Images.bkg-2, vmax=DF_Images.bkg+50, 
                              save=save, save_dir=dir_name, suffix='_'+method)

        if leg2d:
            ds.draw_background(save=save, save_dir=dir_name, suffix='_'+method)

        dsamplers += [ds]
        
    return dsamplers