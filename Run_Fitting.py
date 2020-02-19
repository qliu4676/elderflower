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
[-m][--MAG_THRE]: magnitude thresholds used to select [medium, very] bright stars (default: [14,11]). 
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
%run -i Run_Fitting.py -f 'G' -b '[3000, 1300, 4000, 2300]' -n 2 -r 12 -B
2. In bash

"""

import sys
import getopt
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
    mag_threshold = [14, 11]
    
    # Mask Setup
    mask_type = 'radius'
    r_core = 24
    r_out = None
    SB_fit_thre = 26
    wid_strip, n_strip = 24, 48
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv, "f:b:n:r:m:c:s:M:I:BLFCP",
                                       ["FILTER=", "IMAGE=", "IMAGE_BOUNDS=",
                                        "N_COMP=", "R_SCALE=", "MAG_THRE=",
                                        "MASK_TYPE=", "R_CORE=", "SB_FIT_THRE=", 
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
                                %(band, r_scale, r_core, image_bounds0[0], image_bounds0[1]))
    elif mask_type=='brightness':
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
    # Setup PSF
    ############################################
    from modeling import PSF_Model
    
    image_size = (image_bounds0[2] - image_bounds0[0]) - 2 * pad
    
    # PSF Parameters (some from fitting stacked PSF)
    frac = 0.3                  # fraction of aureole
    beta = 10                   # moffat beta, in arcsec
    fwhm = 2.3 * pixel_scale    # moffat fwhm, in arcsec

    n0 = 3.2                    # estimated true power index
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
    # Read Image and Table
    ############################################
    from image import Image
    DF_Image = Image("./data/coadd_Sloan%s_NGC_5907.fits"%band, image_bounds0)

    from utils import read_measurement_tables
    table_faint, table_res_Rnorm = read_measurement_tables(dir_measure, image_bounds0)
    
    ############################################
    # Setup Stars
    ############################################    
    from utils import assign_star_props
    stars_0, stars_all = assign_star_props(table_faint, table_res_Rnorm, DF_Image, 
                                           r_scale=r_scale, verbose=True,
                                           draw=True, save=save, save_dir=dir_name)
    
    ############################################
    # Setup Image
    ############################################
    from modeling import make_base_image
    # Make fixed background of dim stars
    image_base = make_base_image(image_size, stars_all, psf_base=star_psf,
                                 psf_size=64, pad=pad, verbose=True)

    image = DF_Image.image

    ############################################
    # Make Mask
    ############################################
    from utils import SB2Intensity
    from mask import Mask
    
    mask = Mask(DF_Image, stars0)

    if mask_type=='brightness':
        count = SB2Intensity(SB_fit_thre, mu, ZP, pixel_scale)[0]
    else:
        count = None
    
    # Primary combined mask
    mask.make_mask_map_deep(by=mask_type, dir_measure='./Measure-PS/',
                            r_core=r_core, r_out=None,
                            sn_thre=2.5, n_dilation=5, draw=True)
    
    # Supplementary mask
    if stars_0.n_verybright > 0:
        # S/N + Strip + Cross mask
        mask.make_mask_strip(wid_strip=wid_strip, n_strip=n_strip,
                             dist_strip=image_size, dist_cross=72,
                             clean=True, draw=draw_real,
                             save=save, save_dir=dir_name)
        stars_b = mask.stars_new
        mask_fit = mask.mask_comb
        
    else:
        # S/N mask only
        stars_b = stars_0
        mask_fit = mask.mask_deep
        
    # Make stars for fit. Choose if only use brightest stars
    if brightest_only:
        stars = stars_b.use_verybright()
    else:
        stars = stars_b # for fit

    z_norm_stars = stars.z_norm.copy()
    stars_tri = stars.copy()

    ############################################
    # Estimates
    ############################################
    from utils import compute_poisson_noise
    from astropy.stats import sigma_clip
    
    Y = image[~mask_fit].copy().ravel()
    
    # Compute Sky Poisson Noise
    std_poi = compute_poisson_noise(Y, header=header)
    
    # Sigma-clipped sky
    Y_sky = sigma_clip(image[~mask.mask_base], sigma=3)
    
    # Estimated mu and sigma used as prior
    mu_patch, std_patch = np.mean(Y_sky), np.std(Y_sky)
    print("Estimate of Background: (%.3f +/- %.3f)"%(mu_patch, std_patch))

    ############################################
    # Setup Priors and Likelihood Models for Fitting
    ############################################
    from container import Container

    container = Container(n_spline=2, leg2d=False,
                          fit_sigma=True, fit_frac=False,
                          brightest_only=False,
                          parallel=False, draw_real=True)
    # Set Priors
    container.set_prior(n0, mu, std_patch, n_min=1, theta_in=50, theta_out=240)

    # Set Likelihood
    container.set_likelihood(Y, mask_fit, psf_tri, stars_tri, 
                             psf_range=[None, None], 
                             norm='brightness', z_norm=z_norm_stars,
                             image_base=image_base)
    
    ndim = container.ndim
    
    ############################################
    # Run Sampling
    ############################################
    from sampler import DynamicNestedSampler
    
    ds = DynamicNestedSampler(container, sample='auto', n_cpu=n_cpu)
    
    ds.run_fitting(nlive_init=ndim*10, nlive_batch=2*ndim+2,
                   maxbatch=2, print_progress=print_progress)
    
    if save:
        fit_info = {'n_spline':n_spline, 'image_size':image_size,
                    'image_bounds0':image_bounds0, 'leg2d':leg2d,
                    'r_core':r_core, 'r_scale':r_scale}
        
        method = str(n_spline)+'p'
        fname='NGC5907-%s-fit_best_X%dY%d_%s'\
                    %(band, image_bounds0[0], image_bounds0[1], method)
        if leg2d: fname+='l'
        if brightest_only: fname += 'b'
            
        ds.save_results(fname+'.res', fit_info, save_dir=dir_name)
        
    ############################################
    # Plot Results
    ############################################
    from utils import cal_reduced_chi2
    
    ds.cornerplot(figsize=(18, 16),
                  save=save, save_dir=dir_name, suffix='_'+method)
    
    # Plot recovered PSF
    ds.plot_fit_PSF1D(psf, n_bootstrap=800, r_core=r_core, leg2d=leg2d,
                      save=save, save_dir=dir_name, suffix='_'+method)
    
    # Recovered PSF
    psf_fit, params = ds.generate_fit(psf, stars, image_base,
                                      brightest_only=brightest_only,
                                      leg2d=leg2d, n_out=4, theta_out=1200)
    
    # Calculate Chi^2
    cal_reduced_chi2((ds.image_fit[~mask_fit]).ravel(), Y, params)
    
    # Draw compaison
    ds.draw_comparison_2D(image, mask, vmin=mu-psf_fit.bkg_std, vmax=mu+25*psf_fit.bkg_std,
                          save=save, save_dir=dir_name, suffix='_'+method)
    
    if leg2d:
        ds.draw_background(save=save, save_dir=dir_name, suffix='_'+method)

    return ds
    
if __name__ == "__main__":
    main(sys.argv[1:])
