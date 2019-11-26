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
    hdu_path = "./data/coadd_Sloan%s_NGC_5907.fits"%band
    
    # Fitting Setup (default)
    n_cpu = 3
    method = '3p'
    draw_real = True
    brightest_only = True
    parallel = False 
    leg2d = False
    
    # Fitting Option (default)
    print_progress = True
    draw = True
    save = False

    # Measure Parameter
    r_scale = 12
    mag_threshold = 15
    
    # Mask Setup
    r_core = [36, 24]
    r_out = None
    wid_strip, n_strip = 24, 48
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv, "f:b:n:r:c:I:BLCP",
                                       ["FILTER=", "IMAGE=", "IMAGE_BOUNDS=", "N_COMP=",
                                        "R_SCALE=", "R_CORE=", "W_STRIP=", "N_STRIP=",
                                        "BRIGHTEST_ONLY", "CONV", "PARALLEL", "N_CPU=", "NO_PRINT"])
        opts = [opt for opt, arg in optlists]        
        
    except getopt.GetoptError:
        print('Wrong Option.')
        sys.exit(2)
    
    for opt, arg in optlists:
        if opt in ("-f", "--FILTER"):
            if arg in ["G", "R", "r", "g"]:
                band = arg.upper()
            else:
                print("Filter Not Available.")
                sys.exit(1)
        elif opt in ("-I", "--IMAGE"):
            hdu_path = arg
        elif opt in ("-b", "--IMAGE_BOUNDS"):    
            image_bounds = np.array(re.findall(r'\d+', arg), dtype=int).reshape(-1,4)
        elif opt in ("-n", "--N_COMP"):
            n_comp = np.int(arg)
            if n_comp == 1:
                method = 'p'
            else:
                if n_comp == 2:
                    method = '2p'
                if n_comp == 3:
                    method = '3p'
        elif opt in ("-r", "--r_SCALE"):
            R_norm = np.float(arg)
        elif opt in ("-c", "--R_CORE"):    
            r_core = np.array(re.findall(r'\d+', arg), dtype=float)
        elif opt in ("--W_STRIP"):
            wid_strip = np.float(arg)
        elif opt in ("--N_STRIP"):
            n_strip = np.float(arg)
        elif opt in ("--N_CPU"):
            n_cpu = np.int(arg)
        
    if '-L' in opts: leg2d = True
    if not (('-B' in opts)|("--BRIGHTEST_ONLY" in opts)): brightest_only = False
    if ('-C' in opts)|("--CONV" in opts): draw_real = False
    if ('-P' in opts)|("--PARALLEL" in opts): parallel = True
    if ("--NO_PRINT" in opts): print_progress = True
    
    dir_name = "./fit-real/NGC5907-%s-R%dm%dpix"%(band, r_scale, r_core[0])
    if save:
        check_save_path(dir_name, make_new=False)
    
    ds = Run_Fitting(hdu_path, image_bounds0, method, band=band,
                     r_scale=r_scale, r_core=r_core, r_out=r_out,
                     pixel_scale=pixel_scale, n_cpu=n_cpu,
                     wid_strip=wid_strip, n_strip=n_strip,
                     parallel=parallel, draw_real=draw_real,
                     brightest_only=brightest_only, leg2d=leg2d,
                     print_progress=print_progress, draw=draw,
                     save=save, dir_name=dir_name)

    return opts

    
def Run_Fitting(hdu_path,
                image_bounds0,
                method='2p',
                dir_measure='./Measure',
                band="G",
                r_scale=12, r_core=[36,36], r_out=None,
                pad=100, pixel_scale=2.5, n_cpu=None,
                wid_strip=24, n_strip=48, 
                parallel=False, draw_real=True,
                brightest_only=True, leg2d=False,
                print_progress=True, draw=True,
                save=False, dir_name='./'):
    
    ############################################
    # Setup PSF
    ############################################
    
    patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = image_bounds0
    image_size = (patch_Xmax0 - patch_Xmin0) - 2 * pad
    
    # PSF Parameters
    frac = 0.3                  # fraction of aureole (from fitting stacked PSF)
    beta = 10                                     # moffat beta, in arcsec
    fwhm = 2.28 * pixel_scale                     # moffat fwhm, in arcsec

    n0 = 3.1                    # estimated true power index
    theta_0 = 5.                # radius at which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([n0, 2.8, 2.4, 1.8, 1.2, 3])                 # power index
    theta_s = np.array([theta_0, 72, 150, 240, 300, 1200])      # transition radius in arcsec

    if method == 'p':
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

    ############################################
    # Read
    ############################################
    
    # Read hdu
    with fits.open(hdu_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)

    # Backgroundlevel and estimated std
    mu, ZP = np.float(header["BACKVAL"]), np.float(header["REFZP"])
    print("BACKVAL: %.2f , ZP: %.2f , PIX_SCALE: %.2f" %(mu, ZP, pixel_scale))

    # Crop image
    image_bounds = (patch_Xmin0+pad, patch_Ymin0+pad, patch_Xmax0-pad, patch_Ymax0-pad)
    patch_Xmin, patch_Ymin = image_bounds[0], image_bounds[1]

    patch0, seg_patch0 = crop_image(data, image_bounds0, draw=False)

    # Read measurement for faint stars from catalog
    fname_catalog = os.path.join(dir_measure, "NGC5907-%s-catalog_PS_r15mag.txt"%band)
    tab_catalog = Table.read(fname_catalog, format="ascii")
    
    tab_faint = tab_catalog[(tab_catalog["rmag"]>=15) & (tab_catalog["rmag"]<22)]
    tab_faint = crop_catalog(tab_faint, keys=("X_IMAGE_PS", "Y_IMAGE_PS"),
                             bounds=image_bounds)
    tab_faint["FLUX_AUTO"] = 10**((tab_faint["MAG_AUTO"]-ZP)/(-2.5))

    # Read measurement for bright stars
    fname_res_Rnorm = os.path.join(dir_measure, "NGC5907-%s-norm_%dpix_r15mag_X%dY%d.txt"\
                                   %(band, r_scale, patch_Xmin0, patch_Ymin0))
    table_res_Rnorm = Table.read(fname_res_Rnorm, format="ascii")
    
    table_res_Rnorm = crop_catalog(table_res_Rnorm, bounds=image_bounds0)

    ############################################
    # Setup Stars
    ############################################
    # Positions & Flux of faint stars from SE
    ma = tab_faint['FLUX_AUTO'].data.mask
    star_pos1 = np.vstack([tab_faint['X_IMAGE_PS'].data[~ma],
                           tab_faint['Y_IMAGE_PS'].data[~ma]]).T - [patch_Xmin, patch_Ymin]
    Flux1 = np.array(tab_faint['FLUX_AUTO'].data[~ma])

    # Positions & Flux (estimate) of faint stars from measured norm
    star_pos2 = np.vstack([table_res_Rnorm['X_IMAGE'],
                           table_res_Rnorm['Y_IMAGE']]).T - [patch_Xmin, patch_Ymin]

    # Estimate of brightness I at r_scale (I = Intensity - BKG) and flux
    z_norm = table_res_Rnorm['Imean'].data - mu
    z_norm[z_norm<=0] = z_norm[z_norm>0].min()/10
    Flux2 = psf.I2Flux(z_norm, r=r_scale)

    # Thresholds
    SB_threshold = np.array([27.5, 23.5])
    Flux_threshold = psf.SB2Flux(SB_threshold, BKG=mu, ZP=ZP, r=r_scale)
    print("\nSurface Brightness Thresholds: %r mag/arcsec^2 at %d pix"%(SB_threshold, r_scale))
    print("(<=> Flux Thresholds: %r)\n"%np.around(Flux_threshold,2))

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

    # Make sky background and dim stars
    image_base = make_base_image(image_size, stars_all, psf_base=star_psf,
                                 psf_size=64, pad=pad, verbose=True)

    # Cutout
    image0 = patch0.copy()
    image = image0[pad:-pad,pad:-pad]

    ############################################
    # Make Mask
    ############################################
    mask = Mask(image0, stars0, image_size, pad=pad, mu=mu)

    mask_base = "./Measure/Seg_PS_X%dY%d.fits" %(patch_Xmin0, patch_Ymin0)
    mask.make_mask_map_dual(r_core, r_out=r_out, mask_base=mask_base,
                            sn_thre=2.5, n_dilation=5, draw=True, save=save, dir_name=dir_name)
    
    if stars0.n_verybright > 0:
        # Strip + Cross mask
        mask.make_mask_strip(wid_strip, n_strip, dist_strip=1000,
                             clean=True, draw=draw_real, save=save, dir_name=dir_name)
        stars_b = mask.stars_new
        mask_strip = True
        subtract_external = True
    else:
        stars_b = stars0
        mask_strip = False
        subtract_external = False
        
    plt.show()

    # Choose if only use brightest stars
    if brightest_only:
        stars_vb = Stars(stars_b.star_pos_verybright,
                         stars_b.Flux_verybright,
                         Flux_threshold=stars_b.Flux_threshold,
                         z_norm=stars_b.z_norm_verybright,
                         BKG=stars_b.BKG, r_scale=r_scale)
        stars = stars_vb
        print("\nOnly model brightest stars in the field.\n")
    else:
        stars = stars_b

    z_norm_stars = stars.z_norm.copy()
    stars_tri = stars.copy()

    ############################################
    # Estimates
    ############################################
    if mask_strip is True:
        mask_fit = mask.mask_comb
    else:
        mask_fit = mask.mask_deep

    X = np.array([psf.xx,psf.yy])
    Y = image[~mask_fit].copy().ravel()

    # Estimated mu and sigma used as prior
    Y_clip = sigma_clip(image[~mask.mask_deep], sigma=3, maxiters=10)
    mu_patch, std_patch = np.mean(Y_clip), np.std(Y_clip)
    print("Estimate of Background: (%.3f +/- %.3f)"%(mu_patch, std_patch))

    ############################################
    # Priors and Likelihood Models for Fitting
    ############################################

    # Make Priors
    prior_tf = set_prior(n_est=n0, mu_est=mu_patch, std_est=std_patch, leg2d=leg2d,
                         n_min=0.8, theta_in=60, theta_out=300, method=method)

    # Not working in script! 
#     loglike_2p = set_likelihood(Y, mask_fit, psf, stars_tri, psf_range=[320,640],
#                                 image_base=image_base, z_norm=z_norm_stars, 
#                                 brightest_only=brightest_only, leg2d=False,
#                                 method=method, parallel=False, draw_real=True)
    if leg2d:
        cen = psf.cen
        x_grid = y_grid = np.linspace(0,image_size-1,image_size)
        H10 = leggrid2d((x_grid-cen[1])/image_size, (y_grid-cen[0])/image_size, c=[[0,1],[0,0]])
        H01 = leggrid2d((x_grid-cen[1])/image_size, (y_grid-cen[0])/image_size, c=[[0,0],[1,0]])
        
    def loglike_2p(v):
        _n_s = np.append(v[:2], 3)
        _theta_s = np.append([theta_0, 10**v[2]], 1200)
        _mu, _sigma = v[-2], 10**v[-1]

        psf.update({'n_s':_n_s, 'theta_s':_theta_s})

        # I varies with sky background
        stars_tri.z_norm = z_norm_stars + (stars_tri.BKG - _mu)

        image_tri = generate_image_by_znorm(psf, stars_tri, psf_range=[320,640],
                                            brightest_only=brightest_only,
                                            subtract_external=subtract_external,
                                            psf_scale=pixel_scale,
                                            parallel=False, draw_real=draw_real)
        image_tri = image_tri + image_base + _mu
        
        if leg2d:
            A10, A01 = 10**v[-3], 10**v[-4]
            image_tri += A10 * H10 + A01 * H01
                
        ypred = image_tri[~mask_fit].ravel()
        residsq = (ypred - Y)**2 / _sigma**2
        loglike = -0.5 * np.sum(residsq + math.log(2 * np.pi * _sigma**2))

        if not np.isfinite(loglike):
            loglike = -1e100

        return loglike
    
    def loglike_3p(v):
        _n_s = np.append(v[:3], 3)
        _theta_s = np.append([theta_0, 10**v[3], 10**v[4]], 1200)
        _mu, _sigma = v[-2], 10**v[-1]

        psf.update({'n_s':_n_s, 'theta_s':_theta_s})

        # I varies with sky background
        stars_tri.z_norm = z_norm_stars + (stars_tri.BKG - _mu)

        image_tri = generate_image_by_znorm(psf, stars_tri, psf_range=[360,720],
                                            subtract_external=subtract_external,
                                            brightest_only=brightest_only,
                                            psf_scale=pixel_scale,
                                            parallel=parallel, draw_real=draw_real)
        image_tri = image_tri + image_base + _mu

        if leg2d:
            A10, A01 = 10**v[-3], 10**v[-4]
            image_tri += A10 * H10 + A01 * H01
            
        ypred = image_tri[~mask_fit].ravel()
        residsq = (ypred - Y)**2 / _sigma**2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * _sigma**2))

        if not np.isfinite(loglike):
            loglike = -1e100

        return loglike

    if method=='2p':
        labels = [r'$n0$', r'$n1$', r'$\theta_1$', r'$\mu$', r'$\log\,\sigma$']
        if leg2d:
            labels = [r'$n0$', r'$n1$', r'$\theta_1$',
                      r'$\log\,A_{01}}$', r'$\log\,A_{10}}$', r'$\mu$', r'$\log\,\sigma$']
            
        loglike = loglike_2p
        
    elif method=='3p':
        labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$', r'$\mu$', r'$\log\,\sigma$']
        if leg2d:
            labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$',
                      r'$\log\,A_{01}}$', r'$\log\,A_{10}}$', r'$\mu$', r'$\log\,\sigma$']
            
        loglike = loglike_3p
        
    ndim = len(labels)
    
    ############################################
    # Run & Plot
    ############################################
    ds = DynamicNestedSampler(loglike, prior_tf, ndim, n_cpu=n_cpu)
    ds.run_fitting(nlive_init=60, nlive_batch=20, maxbatch=2, print_progress=print_progress)

    ds.cornerplot(labels=labels, figsize=(18, 16), save=save, dir_name=dir_name)
    ds.plot_fit_PSF1D(psf, n_bootstrap=500, leg2d=leg2d,
                      Amp_max=Amp_m, r_core=r_core, save=save, dir_name=dir_name)

    if save:
        fit_info = {'method':method, 'image_size':image_size,
                    'image_bounds0':image_bounds0, 'leg2d':leg2d,
                    'r_core':r_core, 'r_scale':r_scale}
        
        fname='NGC5907-%s-fit_best_X%dY%d_%s.res'%(band, patch_Xmin0, patch_Ymin0, method)
        if leg2d: fname+='l'
        if brightest_only: fname += 'b'
            
        ds.save_result(fname, fit_info, dir_name=dir_name)
        
    psf_fit, params = make_psf_from_fit(ds.results, psf, n_out=3, theta_out=1200, leg2d=leg2d)
    image_fit, noise_fit, bkg_fit = generate_image_fit(psf_fit, stars0, image_base, leg2d=leg2d)

    cal_reduced_chi2((image_fit[~mask_fit]).ravel(), Y, params)
    draw_comparison_2D(image_fit, image, mask_fit, noise_fit, vmin=458, save=save, dir_name=dir_name)
    
    if leg2d:
        im=plt.imshow(bkg_fit, vmin=-0.03, vmax=+0.03)
        colorbar(im)
        if save:
            plt.savefig(os.path.join(dir_name,'%s-Legendre2D_X%dY%d_%s.png'\
                                     %(band, patch_Xmin0, patch_Ymin0, method)), dpi=80)


    return ds
    
if __name__ == "__main__":
    main(sys.argv[1:])