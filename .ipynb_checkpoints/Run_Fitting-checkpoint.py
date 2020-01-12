#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run 2D Bayesian PSF fitting on a sub-region with
dynamic nested sampling. The Model PSF is composed
of an inner (fixed) Moffat core and an outer (user-
specified) multi-power law aureole. The fitting 
result containing the joint PDF, samples and 
weights, etc. and diagnostic plots will be saved.

> Parameter
[-f][--FILTER]: filter of image to be crossmatched. g/G/r/R for Dragonfly.
[-b][--IMAGE_BOUNDS]: bounds of the region to be processed in pixel coordinate. [Xmin, Ymin, Xmax, Ymax]
[-I][--IMAGE]: path of image.
[-n][--N_COMP]: number of multi-power law component (default: 2).
[-r][--R_SCALE]: radius at which normalziation is measured, in pixel.
[-m][--MAG_THRE]: magnitude thresholds used to select [medium, very] bright stars (default: [14,10.5]). 
[-M][--MASK_TYPE]: mask core by "radius" or "brightness" (default: "radius").
[-c][--R_CORE]: inner masked radius for [medium, very] bright stars, in pixel (default: 24). A single value can be passed for both.
[-s][--SB_FIT_THRE]: inner masked surface brightness, in mag/arcsec^2 (default: 26)
[-B][--BRIGHTEST_ONLY]: whether to fit brightest stars only.
[-L]: whether to fit a 1st-order Legendre polynomial for the background.
[--PARALLEL]: whether to draw meidum bright stars in parallel.
[--N_CPU]: number of CPU used in nested sampling (default: n_cpu-1).
[--NO_PRINT]: if yes, suppress progress print.
[--DIR_MEASURE]: directory name where normalization measurements are saved.
[--DIR_NAME]: directory name for saving fitting outputs.

> Example Usage
1. In jupyter notebook / lab
%matplotlib inline
%run -i Run_Fitting.py -f 'G' -b '[3100, 1400, 4100, 2400]' -n 3 -r 12 -B
2. In bash

"""

import sys
import getopt
from utils import *
from modeling import *
from plotting import *


def main(argv):
    # Image Parameter (default)
    band = "G"                
    pixel_scale = 2.5  # arcsec/pixel
    image_bounds0 = [3100, 1400, 4100, 2400]
    # image_bounds0 = [1800, 2400, 2800, 3400]
    
    # Fitting Setup (default)
    n_cpu = 4
    n_spline = 2
    draw_real = True
    brightest_only = False
    parallel = False 
    leg2d = False
    fit_frac = False
    
    # Fitting Option (default)
    print_progress = True
    draw = True
    save = True

    # Measure Parameter
    r_scale = 12
    mag_threshold = [14,10.5]
    
    # Mask Setup
    mask_type = 'radius'
    r_core = [24, 24]
    r_out = None
    SB_fit_thre = 26
    wid_strip, n_strip = 24, 48
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv, "f:b:n:r:m:c:s:M:I:BLFCP",
                                       ["FILTER=", "IMAGE=", "IMAGE_BOUNDS=",
                                        "N_COMP=", "R_SCALE=", "MAG_THRE=",
                                        "MASK_TYPE=", "R_CORE=", "SB_FIT_THRE=" 
                                        "N_CPU=", "PARALLEL", "BRIGHTEST_ONLY", 
                                        "NO_PRINT", "W_STRIP=", "N_STRIP=", "CONV",
                                        "NO_SAVE", "DIR_NAME=", "DIR_MEASURE="])
        opts = [opt for opt, arg in optlists]        
        
    except getopt.GetoptError as e:
        print(e)
        sys.exit('Wrong Option.')
    
    for opt, arg in optlists:
        if opt in ("-f", "--FILTER"):
            if arg in ["G", "R", "r", "g"]:
                band = arg.upper()
            else:
                sys.exit("Filter Not Available.")
    
    # Default File Path
    hdu_path = "./data/coadd_Sloan%s_NGC_5907.fits"%band
    dir_name = './fit-real'
    dir_measure = './Measure'
                
    for opt, arg in optlists:
        if opt in ("-I", "--IMAGE"):
            hdu_path = arg
        elif opt in ("-b", "--IMAGE_BOUNDS"):    
            image_bounds0 = np.array(re.findall(r'\d+', arg), dtype=int)
        elif opt in ("-n", "--N_COMP"):
            try:
                n_spline = np.int(arg)
            except ValueError:
                sys.exit("Model Not Available.")
        elif opt in ("-r", "--R_SCALE"):
            r_scale = np.float(arg)
        elif opt in ("-m", "--MAG_THRE"):    
            mag_threshold = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=float)
        elif opt in ("-M", "--MASK_TYPE"):    
            mask_type = arg
        elif opt in ("-c", "--R_CORE"):    
            r_core = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=float)
        elif opt in ("-s","--SB_FIT_THRE"):    
            SB_fit_thre = np.float(arg)
        elif opt in ("--W_STRIP"):
            wid_strip = np.float(arg)
        elif opt in ("--N_STRIP"):
            n_strip = np.float(arg)
        elif opt in ("--N_CPU"):
            n_cpu = np.int(arg)
        elif opt in ("--DIR_NAME"):
            dir_name = arg
        elif opt in ("--DIR_MEASURE"):
            dir_measure = arg 
        
    if '-L' in opts: leg2d = True
    if '-F' in opts: fit_frac = True
    if ('-B' in opts)|("--BRIGHTEST_ONLY" in opts): brightest_only = True
    if ('-C' in opts)|("--CONV" in opts): draw_real = False
    if ('-P' in opts)|("--PARALLEL" in opts): parallel = True
    if ("--NO_PRINT" in opts): print_progress = False
    if ("--NO_SAVE" in opts): save = False
    
    if mask_type=='radius':
        dir_name = os.path.join(dir_name, "NGC5907-%s-R%dM%dpix_X%dY%d"\
                                %(band, r_scale, r_core[0], image_bounds0[0], image_bounds0[1]))
    elif mask_type=='count':
        dir_name = os.path.join(dir_name, "NGC5907-%s-R%dB%.1f_X%dY%d"\
                                %(band, r_scale, SB_fit_thre, image_bounds0[0], image_bounds0[1]))
    if save:
        check_save_path(dir_name, make_new=False)
    
    # Run Fitting!
    ds = Run_Fitting(hdu_path, image_bounds0, n_spline, band,
                     r_scale=r_scale, mag_threshold=mag_threshold, 
                     mask_type=mask_type, SB_fit_thre=SB_fit_thre,
                     r_core=r_core, r_out=r_out, leg2d=leg2d,
                     pixel_scale=pixel_scale, n_cpu=n_cpu,
                     wid_strip=wid_strip, n_strip=n_strip,
                     brightest_only=brightest_only, draw_real=draw_real,
                     parallel=parallel, print_progress=print_progress, 
                     draw=draw, dir_measure=dir_measure,
                     save=save, dir_name=dir_name)

    return opts

    
def Run_Fitting(hdu_path, image_bounds0,
                n_spline=2, band="G",
                r_scale=12, mag_threshold=[14,11], 
                mask_type='radius', SB_fit_thre=24.5,
                r_core=[24,24], r_out=None,
                fit_frac=False, leg2d=False,
                pad=100, pixel_scale=2.5, 
                wid_strip=24, n_strip=48, 
                n_cpu=None, parallel=False, 
                brightest_only=True, draw_real=True,
                draw=True, print_progress=True,
                save=False, dir_name='./',
                dir_measure='./Measure'):
    
    ############################################
    # Setup PSF
    ############################################
    
    patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = image_bounds0
    image_size = (patch_Xmax0 - patch_Xmin0) - 2 * pad
    
    # PSF Parameters (some from fitting stacked PSF)
    frac = 0.3                  # fraction of aureole
    beta = 10                   # moffat beta, in arcsec
    fwhm = 2.3 * pixel_scale    # moffat fwhm, in arcsec

    n0 = 3.1                    # estimated true power index
    theta_0 = 5.                # radius at which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([n0, 4])                 # power index
    theta_s = np.array([theta_0, 1200])     # transition radius in arcsec
    
    if n_spline == 1:
        # Single-power PSF
        params_pow = {"fwhm":fwhm, "beta":beta, "frac":frac, "n":n0, 'theta_0':theta_0}
        psf = PSF_Model(params=params_pow, aureole_model='power')
    else:
        # Multi-power PSF
        params_mpow = {"fwhm":fwhm, "beta":beta, "frac":frac, "n_s":n_s, 'theta_s':theta_s}
        psf = PSF_Model(params=params_mpow, aureole_model='multi-power')

    # Build grid of image for drawing
    psf.make_grid(image_size, pixel_scale=pixel_scale)

    # Generate core and aureole PSF
    psf_c = psf.generate_core()
    psf_e, psf_size = psf.generate_aureole(contrast=1e6, psf_range=image_size)
    star_psf = (1-frac) * psf_c + frac * psf_e
    
    psf_tri = psf.copy()

    ############################################
    # Read
    ############################################
    
    # Read hdu
    if os.path.isfile(hdu_path) is False:
        sys.exit("Image does not exist. Check path.")
    with fits.open(hdu_path) as hdul:
        print("Read Image :", hdu_path)
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)
    
    # Background level and Zero Point
    try:
        mu, ZP = np.array([header["BACKVAL"], header["REFZP"]]).astype(float)
        print("BACKVAL: %.2f , ZP: %.2f , PIXSCALE: %.2f\n" %(mu, ZP, pixel_scale))

    except KeyError:
        print("BKG / ZP / PIXSCALE missing in header --->")
        ZP = np.float(input("Input a value of ZP :"))
        mu = np.float(input("Manually set a value of background :"))
        data += mu

    # Crop image
    image_bounds = (patch_Xmin0+pad, patch_Ymin0+pad, patch_Xmax0-pad, patch_Ymax0-pad)
    patch_Xmin, patch_Ymin = image_bounds[0], image_bounds[1]

    patch0, seg_patch0 = crop_image(data, image_bounds0, draw=False)

    # Magnitude name
    b_name = band.lower()
    mag_name = b_name+'MeanPSFMag' if 'PS' in dir_measure else b_name+'mag'
    
    # Read measurement for faint stars from catalog
    fname_catalog = os.path.join(dir_measure, "NGC5907-%s-catalog_PS_%s_all.txt"%(band, b_name))
    if os.path.isfile(fname_catalog):
        tab_catalog = Table.read(fname_catalog, format="ascii")
    else:
        sys.exit("Table %s does not exist. Exit."%fname_catalog)
    
    tab_faint = tab_catalog[(tab_catalog[mag_name]>=15) & (tab_catalog[mag_name]<23)]
    tab_faint = crop_catalog(tab_faint, keys=("X_IMAGE_PS", "Y_IMAGE_PS"),
                             bounds=image_bounds)

    # Read measurement for bright stars
    fname_res_Rnorm = os.path.join(dir_measure, "NGC5907-%s-norm_%dpix_%s15mag_X%dY%d.txt"\
                                   %(band, r_scale, b_name, patch_Xmin0, patch_Ymin0))
    if os.path.isfile(fname_res_Rnorm):
        table_res_Rnorm = Table.read(fname_res_Rnorm, format="ascii")
    else:
        sys.exit("Table %s does not exist. Exit."%fname_res_Rnorm)
    
    table_res_Rnorm = crop_catalog(table_res_Rnorm, bounds=image_bounds0)
    Iflag = table_res_Rnorm["Iflag"]
    table_res_Rnorm = table_res_Rnorm[Iflag==0]    

    ############################################
    # Setup Stars
    ############################################
    # Positions & Flux of faint stars from SE
    try:
        ma = tab_faint['FLUX_AUTO'].data.mask
    except AttributeError:
        ma = np.isnan(tab_faint['FLUX_AUTO'])
    star_pos1 = np.vstack([tab_faint['X_IMAGE_PS'].data[~ma],
                           tab_faint['Y_IMAGE_PS'].data[~ma]]).T - [patch_Xmin, patch_Ymin]
    Flux1 = np.array(tab_faint['FLUX_AUTO'].data[~ma])
        

    # Positions & Flux (estimate) of faint stars from measured norm
    star_pos2 = np.vstack([table_res_Rnorm['X_IMAGE_PS'],
                           table_res_Rnorm['Y_IMAGE_PS']]).T - [patch_Xmin, patch_Ymin]

    Flux2 = 10**((table_res_Rnorm["MAG_AUTO_corr"]-ZP)/(-2.5))
    
    # Estimate of brightness I at r_scale (I = Intensity - BKG) (and flux)
    # Background is slightly lowered (1 ADU). Note this is not affecting fitting.
    z_norm = table_res_Rnorm['Imean'].data - (mu-1)
    
    Flux_threshold = 10**((mag_threshold - ZP) / (-2.5))
    SB_threshold = psf.Flux2SB(Flux_threshold, BKG=mu, ZP=ZP, r=r_scale)
    
    print('Magnitude Thresholds:  {0}, {1} mag'.format(*mag_threshold))
    print("(<=> Flux Thresholds: {0}, {1} ADU)".format(*np.around(Flux_threshold,2)))
    print("(<=> Surface Brightness Thresholds: {0}, {1} mag/arcsec^2 at {2} pix)\n"\
          .format(*np.around(SB_threshold,1),r_scale))

    # Combine 
    star_pos = np.vstack([star_pos1, star_pos2])
    Flux = np.concatenate([Flux1, Flux2])
    stars_all = Stars(star_pos, Flux, Flux_threshold=Flux_threshold)
    stars_all.plot_flux_dist(label='All', color='plum')

    # Bright stars in model
    stars0 = Stars(star_pos2, Flux2, Flux_threshold=Flux_threshold,
                   z_norm=z_norm, r_scale=r_scale, BKG=mu)
    stars0 = stars0.remove_outsider(image_size, d=[36, 12], verbose=True)
    stars0.plot_flux_dist(label='Model', color='orange', ZP=ZP, save=save, dir_name=dir_name)
    
    # Maximum amplitude from estimate
    Amp_m = psf.Flux2Amp(Flux).max()

    ############################################
    # Setup Image
    ############################################

    # Make fixed background of dim stars
    image_base = make_base_image(image_size, stars_all, psf_base=star_psf,
                                 psf_size=64, pad=pad, verbose=True)

    # Cutout
    image0 = patch0.copy()
    image = image0[pad:-pad,pad:-pad]

    ############################################
    # Make Mask
    ############################################
    mask = Mask(image0, stars0, image_size, pad=pad, mu=mu)

    seg_base = "./Measure/Seg_PS_X%dY%d.fits" %(patch_Xmin0, patch_Ymin0)
    
    if mask_type=='count':
        count = SB2Intensity(SB_fit_thre, mu, ZP, pixel_scale)[0]
    else:
    count = None
    mask.make_mask_map_deep(by=mask_type, seg_base=seg_base,
                            r_core=r_core, r_out=r_out, count=count,
                            sn_thre=2.5, n_dilation=5, draw=True, 
                            save=save, dir_name=dir_name)
    
    if stars0.n_verybright > 0:
        # S/N + Strip + Cross mask
        mask.make_mask_strip(wid_strip=wid_strip, n_strip=n_strip,
                             dist_strip=image_size, dist_cross=72,
                             clean=True, draw=draw_real,
                             save=save, dir_name=dir_name)
        stars_b = mask.stars_new
        mask_fit = mask.mask_comb
        
    else:
        # S/N mask only
        stars_b = stars0
        mask_fit = mask.mask_deep
        
    plt.show()

    # Choose if only use brightest stars
    if brightest_only:
        stars_vb = Stars(stars_b.star_pos_verybright,
                         stars_b.Flux_verybright,
                         Flux_threshold=stars_b.Flux_threshold,
                         z_norm=stars_b.z_norm_verybright,
                         BKG=stars_b.BKG, r_scale=r_scale)
        stars = stars_vb # stars for fit
        print("\nOnly model brightest stars in the field.\n")
    else:
        stars = stars_b # for fit

    z_norm_stars = stars.z_norm.copy()
    stars_tri = stars.copy()

    ############################################
    # Estimates
    ############################################
    X = np.array([psf.xx,psf.yy])
    Y = image[~mask_fit].copy().ravel()
    
    # Compute Sky Poisson Noise
    std_poi = compute_poisson_noise(Y, header=header)
    
    # Estimated mu and sigma used as prior
    Y_sky = sigma_clip(image[~mask.mask_base], sigma=3, maxiters=10)
    mu_patch, std_patch = np.mean(Y_sky), np.std(Y_sky)
    print("Estimate of Background: (%.3f, %.3f)"%(mu_patch, std_patch))

    ############################################
    # Priors and Likelihood Models for Fitting
    ############################################

    # Make Priors
    prior_tf = set_prior(n_est=n0, mu_est=mu, std_est=std_patch,
                         n_spline=n_spline, leg2d=leg2d,
                         n_min=1, theta_in=50, theta_out=300)
                         
    if n_spline==1:
        labels = [r'$n0$', r'$\mu$', r'$\log\,\sigma$']
                         
    elif n_spline==2:
        labels = [r'$n0$', r'$n1$', r'$\theta_1$', r'$\mu$', r'$\log\,\sigma$']
            
    elif n_spline==3:
        labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$',
                  r'$\mu$', r'$\log\,\sigma$']
        
    else:
        labels = [r'$n_%d$'%d for d in range(n_spline)] \
               + [r'$\theta_%d$'%(d+1) for d in range(n_spline-1)] \
               + [r'$\mu$', r'$\log\,\sigma$']
        
    if leg2d:
        labels = np.insert(labels, -2, [r'$\log\,A_{01}$', r'$\log\,A_{10}$'])
        
#     if fit_frac:
#         labels = np.insert(labels, -2, [r'$f_{pow}$'])

    ndim = len(labels)
    method = str(n_spline)+'p'
    
    loglike = set_likelihood(Y, mask_fit, psf_tri, stars_tri, 
                             n_spline=n_spline, psf_range=[360,720], norm='brightness',
                             z_norm=z_norm_stars, image_base=image_base,
                             brightest_only=brightest_only, leg2d=leg2d,
                             parallel=parallel, draw_real=draw_real)
    
    ############################################
    # Run & Plot
    ############################################
    ds = DynamicNestedSampler(loglike, prior_tf, ndim,
                              sample='auto', n_cpu=n_cpu)
    ds.run_fitting(nlive_init=7*ndim, nlive_batch=2*ndim+2, 
                   maxbatch=2, print_progress=print_progress)
    
    if save:
        fit_info = {'n_spline':n_spline, 'image_size':image_size,
                    'image_bounds0':image_bounds0, 'leg2d':leg2d,
                    'r_core':r_core, 'r_scale':r_scale}
        
        fname='NGC5907-%s-fit_best_X%dY%d_%s'%(band, patch_Xmin0, patch_Ymin0, method)
        if leg2d: fname+='l'
        if brightest_only: fname += 'b'
            
        ds.save_result(fname+'.res', fit_info, dir_name=dir_name)
    
    ds.cornerplot(labels=labels, figsize=(18, 16), save=save, dir_name=dir_name, suffix='_'+method)
    
    # Plot recovered PSF
    ds.plot_fit_PSF1D(psf, n_bootstrap=800, Amp_max=Amp_m, r_core=r_core, leg2d=leg2d,
                      save=save, dir_name=dir_name, suffix='_'+method)
    
    # Recovered PSF
    psf_fit, params = ds.generate_fit(psf, stars, image_base,
                                      brightest_only=brightest_only,
                                      leg2d=leg2d, n_out=4, theta_out=1200)
    
    # Calculate Chi^2
    cal_reduced_chi2((ds.image_fit[~mask_fit]).ravel(), Y, params)
    
    # Draw compaison
    ds.draw_comparison_2D(image, mask, vmin=mu-psf_fit.bkg_std, vmax=mu+25*psf_fit.bkg_std,
                          save=save, dir_name=dir_name, suffix='_'+method)
    
    if leg2d:
        ds.draw_background(self, save=save, dir_name=dir_name, suffix='_'+method)

    return ds
    
if __name__ == "__main__":
    main(sys.argv[1:])
