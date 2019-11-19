from utils import *
from modeling import *
from plotting import *

# Fitting Parameter
n_cpu = 3
RUN_FITTING = True
print_progress = True
draw = True
draw_real = True
save = False
brightest_only = True
wid_strip, n_strip = 16, 48

dir_name = "./real"
aureole_model = 'multi-power'
method = '3p'

# Image Parameter
image_bounds0 = (3100, 1400, 4100, 2400) 
# image_bounds0 = [1000, 1400, 1800, 2200]
    
pixel_scale = 2.5     # arcsec/pixel
R_scale = 10

# Core mask
r_core_s = [36, 36]
r_out_s = None
    
def run_fitting(image_bounds0, R_scale=10, pad=100, pixel_scale=2.5):
    
    ############################################
    # Setup PSF
    ############################################
    
    patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = image_bounds0
    image_size = (patch_Xmax0 - patch_Xmin0) - 2 * pad
    
    # PSF Parameters
    beta = 10                                     # moffat beta, in arcsec
    fwhm = 2.28 * pixel_scale                     # moffat fwhm, in arcsec

    n0 = 3.1                    # estimated true power index
    theta_0 = 5.                # radius at which power law is flattened, in arcsec (arbitrary)

    n_s = np.array([n0, 2.8, 2.4, 1.8, 1.2, 3])                 # power index
    theta_s = np.array([theta_0, 72, 150, 240, 300, 1200])      # transition radius in arcsec

    if aureole_model == 'power':
        # Single-power PSF
        frac = 0.3              # fraction of power law component (from fitting stacked PSF)
        params_pow = {"fwhm":fwhm, "beta":beta, "frac":frac, "n":n0, 'theta_0':theta_0}
        psf = PSF_Model(params=params_pow, aureole_model='power')

    elif aureole_model == 'multi-power':
        # Multi-power PSF
        frac = 0.3  
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
    hdu = fits.open("./data/coadd_SloanG_NGC_5907.fits")[0]
    data = hdu.data
    header = hdu.header

    # Backgroundlevel and estimated std
    mu, ZP = np.float(hdu.header["BACKVAL"]), np.float(hdu.header["REFZP"])
    print("mu: %.2f , ZP: %.2f , pix_scale: %.2f" %(mu, ZP, pixel_scale))

    # Crop image
    image_bounds = (patch_Xmin0+pad, patch_Ymin0+pad, patch_Xmax0-pad, patch_Ymax0-pad)
    patch_Xmin, patch_Ymin = image_bounds[0], image_bounds[1]

    patch0, seg_patch0 = crop_image(data, image_bounds0, draw=True)

    # Read measurement for faint stars from catalog
    tab_catalog = Table.read("./Measure/NGC5907-G-catalog_PS_r15mag.txt", format="ascii")
    tab_faint = tab_catalog[(tab_catalog["rmag"]>=15) & (tab_catalog["rmag"]<22)]
    tab_faint = crop_catalog(tab_faint, keys=("X_IMAGE_PS", "Y_IMAGE_PS"),
                             bounds=image_bounds)
    tab_faint["FLUX_AUTO"] = 10**((tab_faint["MAG_AUTO"]-ZP)/(-2.5))

    # Read measurement for bright stars
    table_res_Rnorm = Table.read("./Measure/NGC5907-G-norm_%dpix_r15mag_X%dY%d.txt"\
                                 %(R_scale, patch_Xmin0, patch_Ymin0), format="ascii")
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
    Flux2 = psf.I2Flux(z_norm, r=R_scale)

    # Thresholds
    SB_threshold = np.array([27.5,23.5])
    Flux_threshold = psf.SB2Flux(SB_threshold, BKG=mu, ZP=ZP, r=R_scale)
    print("Threosholds:", SB_threshold, Flux_threshold)

    # Combine 
    star_pos = np.vstack([star_pos1, star_pos2])
    Flux = np.concatenate([Flux1, Flux2])
    stars_all = Stars(star_pos, Flux, Flux_threshold=Flux_threshold)
    stars_all.plot_flux_dist(label='All', color='plum')

    # Bright stars in model
    stars0 = Stars(star_pos2, Flux2, Flux_threshold=Flux_threshold,
                   z_norm=z_norm, r_scale=R_scale, BKG=mu)
    stars0 = stars0.remove_outsider(image_size, d=[36, 12], verbose=True)
    stars0.plot_flux_dist(label='Model', color='orange', ZP=ZP, save=save, dir_name=dir_name)
    
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
    mask.make_mask_map_dual(r_core_s, r_out=r_out_s, mask_base=mask_base,
                            sn_thre=3, n_dilation=5, draw=True, save=save, dir_name=dir_name)
    
    if stars0.n_verybright > 0:
        # Strip + Cross mask
        mask.make_mask_strip(wid_strip, n_strip, dist_strip=1000, clean=True, draw=draw_real)
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
                         z_norm=stars_b.z_norm_verybright, BKG=stars_b.BKG, r_scale=R_scale)
        stars = stars_vb
        print("Only model brightest stars in the field.")
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
    print("Estimate of Background: (%.3f, %.3f)"%(mu_patch, std_patch))

    ############################################
    # Priors and Likelihood Models for Fitting
    ############################################

    # Make Priors
    prior_tf = set_prior(n_est=3.1, mu_est=mu_patch, std_est=std_patch,
                         n_min=0.8, theta_in=60, theta_out=300, method=method)

    # Not working in script! 
#     loglike_2p = set_likelihood(Y, mask_fit, psf, stars_tri, method='2p', 
#                                 psf_range=[320,640], image_base=image_base, z_norm=z_norm_stars, 
#                                 brightest_only=brightest_only, leg2d=False,
#                                 parallel=False, draw_real=True)

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
                                            parallel=False, draw_real=True)
        image_tri = image_tri + image_base + _mu

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

        image_tri = generate_image_by_znorm(psf, stars_tri, psf_range=[320,640],
                                            subtract_external=subtract_external,
                                            brightest_only=brightest_only,
                                            psf_scale=pixel_scale,
                                            parallel=False, draw_real=True)
        image_tri = image_tri + image_base + _mu

        ypred = image_tri[~mask_fit].ravel()
        residsq = (ypred - Y)**2 / _sigma**2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * _sigma**2))

        if not np.isfinite(loglike):
            loglike = -1e100

        return loglike

    if method=='2p':
        labels = [r'$n0$', r'$n1$', r'$\theta_1$', r'$\mu$', r'$\log\,\sigma$']
        loglike = loglike_2p
    elif method=='3p':
        labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$', r'$\mu$', r'$\log\,\sigma$']
        loglike = loglike_3p
        
    ndim = len(labels)
    
    ############################################
    # Run & Plot
    ############################################
    ds = DynamicNestedSampler(loglike, prior_tf, ndim)
    ds.run_fitting(nlive_init=50, nlive_batch=15, maxbatch=2)

    ds.cornerplot(labels=labels, figsize=(18, 16), save=save, dir_name=dir_name)
    ds.plot_fit_PSF(psf, n_bootstrap=500,
                    Amp_max=Amp_m, r_core=r_core_s, save=save, dir_name=dir_name)

    
if __name__ == "__main__":
    run_fitting(image_bounds0, R_scale=R_scale)