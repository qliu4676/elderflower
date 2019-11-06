from modeling import *

from matplotlib import rcParams
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'gnuplot2'
plt.rcParams['font.serif'] = 'Times New Roman'
rcParams.update({'xtick.major.pad': '5.0'})
rcParams.update({'xtick.major.size': '4'})
rcParams.update({'xtick.major.width': '1.'})
rcParams.update({'xtick.minor.pad': '5.0'})
rcParams.update({'xtick.minor.size': '4'})
rcParams.update({'xtick.minor.width': '0.8'})
rcParams.update({'ytick.major.pad': '5.0'})
rcParams.update({'ytick.major.size': '4'})
rcParams.update({'ytick.major.width': '1.'})
rcParams.update({'ytick.minor.pad': '5.0'})
rcParams.update({'ytick.minor.size': '4'})
rcParams.update({'ytick.minor.width': '0.8'})
rcParams.update({'axes.labelsize': 14})
rcParams.update({'font.size': 14})

def draw_mask_map(image, seg_map, mask_deep, stars,
                  r_core=24, vmin=884, vmax=1e3,
                  pad=0, save=False, dir_name='./'):
    """ Visualize mask map """
    from matplotlib import patches
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    im1 = ax1.imshow(image, cmap='gray', norm=norm1, vmin=vmin, vmax=1e4, aspect='auto')
    ax1.set_title("Mock")
    colorbar(im1)

    ax2.imshow(seg_map)
    ax2.set_title("Deep Mask")

    image2 = image.copy()
    image2[mask_deep] = 0
    im3 = ax3.imshow(image2, norm=norm2, vmin=vmin, vmax=vmax, aspect='auto') 
    ax3.set_title("'Sky'")
    colorbar(im3)
    
    if np.ndim(r_core) == 0:
        r_core = [r_core, r_core]
    
    star_pos_A = stars.star_pos_verybright + pad
    star_pos_B = stars.star_pos_medbright + pad
    
    aper = CircularAperture(star_pos_A, r=r_core[0])
    aper.plot(color='lime',lw=2,label="",alpha=0.9, axes=ax3)
    aper = CircularAperture(star_pos_B, r=r_core[1])
    aper.plot(color='c',lw=2,label="",alpha=0.7, axes=ax3)
    
    patch_size = image.shape[0] - pad * 2
    rec = patches.Rectangle((pad, pad), patch_size, patch_size,
                            facecolor='none', edgecolor='w', linestyle='--',alpha=0.8)
    ax3.add_patch(rec)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(dir_name, "Mask_dual.png"), dpi=150)
        plt.close()

def draw_mask_map_strip(image, seg_comb, mask_comb, stars,
                        ma_example=None,
                        r_core=24, vmin=884, vmax=1e3,
                        pad=0, save=False, dir_name='./'):
    """ Visualize mask map w/ strips """
    
    from matplotlib import patches
    
    star_pos_A = stars.star_pos_verybright + pad
    star_pos_B = stars.star_pos_medbright + pad
    
    if ma_example is not None:
        mask_strip, mask_cross = ma_example
    
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    mask_strip[mask_cross.astype(bool)]=0.5
    ax1.imshow(mask_strip, cmap="gray_r")
    ax1.plot(star_pos_A[0][0], star_pos_A[0][1], "r*",ms=18)
    ax1.set_title("Mask Strip")

    ax2.imshow(seg_comb)
    ax2.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=18)
    ax2.set_title("Mask Comb.")

    image3 = image.copy()
    image3[mask_comb] = 0
    im3 = ax3.imshow(image3, norm=norm1, aspect='auto', vmin=vmin, vmax=vmax) 
    ax3.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=18)
    ax3.set_title("'Sky'")
    colorbar(im3)

    aper = CircularAperture(star_pos_A, r=r_core[0])
    aper.plot(color='lime',lw=2,label="",alpha=0.9, axes=ax3)
    aper = CircularAperture(star_pos_B, r=r_core[1])
    aper.plot(color='c',lw=2,label="",alpha=0.7, axes=ax3)

    size = image.shape[0] - pad * 2
    rec = patches.Rectangle((pad, pad), size, size,
                            facecolor='none', edgecolor='w', linestyle='--',alpha=0.8)
    ax3.add_patch(rec)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir_name, "Mask_strip.png"), dpi=150)
        plt.close()
        
def Fit_background_distribution(image, mask_deep):
    # Check background, fit with gaussian and exp-gaussian distribution
    import seaborn as sns
    plt.figure(figsize=(6,4))
    z_sky = image[~mask_deep]
    sns.distplot(z_sky, label='Data', hist_kws={'alpha':0.3})

    mu_fit, std_fit = stats.norm.fit(z_sky)
    print(mu_fit, std_fit)
    d_mod = stats.norm(loc=mu_fit, scale=std_fit)
    x = np.linspace(d_mod.ppf(0.001), d_mod.ppf(0.999), 100)
    plt.plot(x, d_mod.pdf(x), 'g-', lw=2, alpha=0.6, label='Norm Fit')

    K_fit, mu_fit, std_fit = stats.exponnorm.fit(z_sky)
    print(K_fit, mu_fit, std_fit)
    d_mod2 = stats.exponnorm(loc=mu_fit, scale=std_fit, K=K_fit)
    x = np.linspace(d_mod2.ppf(0.001), d_mod2.ppf(0.9999), 100)
    plt.plot(x, d_mod2.pdf(x), 'r-', lw=2, alpha=0.6, label='Exp-Norm Fit')

    plt.legend(fontsize=12)
    
def plot_PSF_model_1D(frac, f_core, f_aureole,
                      psf_range=400, yunit='Intensity',
                      ZP=27.1, pixel_scale=2.5):
    
    r = np.logspace(0, np.log10(psf_range), 100)
    
    I_core = (1-frac) * f_core(r)
    I_aureole = frac * f_aureole(r)
    I_tot = I_core + I_aureole
    
    if yunit=='Intensity':
        plt.semilogx(r, np.log10(I_tot),
                 ls="-", lw=3,alpha=0.9, zorder=5, label='combined')
        plt.semilogx(r, np.log10(I_core),
                 ls="--", lw=3, alpha=0.9, zorder=1, label='core')
        plt.semilogx(r, np.log10(I_aureole),
                 ls="--", lw=3, alpha=0.9, label='aureole')
        plt.ylabel('log Intensity', fontsize=14)
        plt.ylim(np.log10(I_aureole).min(), -0.5)
        
    elif yunit=='SB':
        plt.semilogx(r, -14.5+Intensity2SB(I=I_tot, BKG=0,
                                           ZP=27.1, pixel_scale=pixel_scale),
                     ls="-", lw=3,alpha=0.9, zorder=5, label='combined')
        plt.semilogx(r, -14.5+Intensity2SB(I=I_core, BKG=0,
                                           ZP=27.1, pixel_scale=pixel_scale),
                     ls="--", lw=3, alpha=0.9, zorder=1, label='core')
        plt.semilogx(r, -14.5+Intensity2SB(I=I_aureole, BKG=0,
                                           ZP=27.1, pixel_scale=pixel_scale),
                     ls="--", lw=3, alpha=0.9, label='aureole')
        plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
        plt.ylim(31,17)

    plt.legend(loc=1, fontsize=12)
    plt.xlabel('r [pix]', fontsize=14)
    
def plot_PSF_model_galsim(psf_inner, psf_outer, params,
                          image_size, pixel_scale,
                          contrast=None, save=False, dir_name='.',
                          aureole_model="Power"):
    
    frac, gamma_pix, beta = params['frac'], params['gamma'] / pixel_scale, params['beta']
    
    c_mof2Dto1D = C_mof2Dto1D(gamma_pix, beta)
    
    if aureole_model == "power":
        n, theta_0_pix = params['n'], params['theta_0'] / pixel_scale
        c_aureole_2Dto1D = C_pow2Dto1D(n, theta_0_pix)
        profile = lambda r: trunc_power1d_normed(r, n, theta_0_pix)
        
    if aureole_model == "multi-power":
        n_s, theta_s_pix = params['n_s'], params['theta_s'] / pixel_scale
        c_aureole_2Dto1D = C_mpow2Dto1D(n_s, theta_s_pix)
        profile = lambda r: multi_power1d_normed(r, n_s, theta_s_pix)
        
    star_psf = (1-frac) * psf_inner + frac * psf_outer
    
    img_outer = psf_outer.drawImage(nx=201, ny=201, scale=pixel_scale, method="no_pixel").array
    img_inner = psf_inner.drawImage(scale=pixel_scale, method="no_pixel").array
    img_star = star_psf.drawImage(nx=image_size, ny=image_size, scale=pixel_scale, method="no_pixel").array
    
    plt.figure(figsize=(7,6))

    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(frac*img_outer, color="g",
                                                  pixel_scale=pixel_scale, seeing=2.5, 
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label=aureole_model)
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d((1-frac)*img_inner, color="orange",
                                                  pixel_scale=pixel_scale, seeing=2.5, 
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label="Moffat")
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(img_star,
                                                  pixel_scale=pixel_scale, seeing=2.5,  
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label="Combined")

    r = np.logspace(0, np.log10(image_size), 100)
    
    comp1 = moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D
    comp2 = profile(r) / c_aureole_2Dto1D
    
    plt.legend(loc=1, fontsize=12)
    
    plt.plot(r, np.log10((1-frac) * comp1 + comp2 * frac), ls="-", lw=3, zorder=5)
    plt.plot(r, np.log10((1-frac) * comp1), ls="--", lw=3, zorder=1)
    plt.plot(r, np.log10(comp2 * frac), ls="--", lw=3)
    
    if aureole_model == "multi-power":
        for t in theta_s_pix:
            plt.axvline(t, ls="--", color="k",alpha=0.3, zorder=1)
        
    if contrast is not None:
        plt.axhline(np.log10(comp1.max()/contrast),color="k",ls="--")
        
    plt.title("Model PSF",fontsize=14)
    plt.ylim(-9, -0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir_name, "Model_PSF.png"), dpi=120)
        plt.close()
    
def plot_flux_dist(Flux, Flux_thresholds,
                   save=False, dir_name='.', **kwargs):
    import seaborn as sns
    F_bright, F_verybright = Flux_thresholds
    plt.axvline(np.log10(F_bright), color="k", ls="-",alpha=0.7, zorder=1)
    plt.axvline(np.log10(F_verybright), color="k", ls="--",alpha=0.7, zorder=1)
    plt.axvspan(1, np.log10(F_bright),
                color='gray', alpha=0.15, zorder=0)
    plt.axvspan(np.log10(F_bright), np.log10(F_verybright),
                color='seagreen', alpha=0.15, zorder=0)
    plt.axvspan(np.log10(F_verybright), 9,
                color='steelblue', alpha=0.15, zorder=0)
    sns.distplot(np.log10(Flux), kde=False, **kwargs)
    plt.yscale('log')
    plt.xlabel('Estimated Total Flux/Mag', fontsize=15)
    plt.ylabel('# of stars', fontsize=15)
    plt.legend(loc=1)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir_name, "Flux_dist.png"), dpi=120)
        plt.close()

def draw_prior(priors, xlabels=None, plabels=None, save=False, dir_name='./'):
    
    x_s = [np.linspace(d.ppf(0.01), d.ppf(0.99), 100) for d in priors]
    
    fig, axes = plt.subplots(1, len(priors), figsize=(15,4))
    for k, ax in enumerate(axes):
        ax.plot(x_s[k], priors[k].pdf(x_s[k]),'-', lw=5, alpha=0.6, label=plabels[k])
        ax.legend()
        if xlabels is not None:
            ax.set_xlabel(xlabels[k], fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir_name, "Prior.png"), dpi=100)
        plt.close()

        
def plot1D_fit_vs_truth_PSF_mpow(res, psf, labels, n_bootstrap=400,
                                 Amp_max=None, r_core=None,
                                 plot_truth=True, save=False, dir_name="."):
    """ Compare 1D fit and truth PSF """
    image_size = psf.image_size
    pixel_scale = psf.pixel_scale
    
    plt.figure(figsize=(7,6))
    r = np.logspace(0., np.log10(image_size), 100)
    
    # read fitting results
    pmed, pmean, pcov, samples_eq = get_params_fit(res, return_sample=True)
    print("Fitting (mean) : ", np.around(pmean,3))
    print("Fitting (median) : ", np.around(pmed,3))
    
    from astropy.stats import bootstrap
    samples_eq_bs = bootstrap(samples_eq, bootnum=1, samples=n_bootstrap)[0]
    
    # truth psf params
    gamma_pix, beta, frac = psf.gamma_pix, psf.beta, psf.frac
    n_s, theta_s_pix = psf.n_s, psf.theta_s_pix
    print('Truth : ', n_s)
    
    # 2D - 1D conversion for visual
    c_mof2Dto1D =  C_mof2Dto1D(gamma_pix, beta)
    c_mpow2Dto1D = C_mpow2Dto1D(n_s=n_s, theta_s=theta_s_pix)
    
    # 1D truth psf
    comp1 = moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D
    comp2 = multi_power1d_normed(r, n_s, theta_s_pix) / c_mpow2Dto1D
    
    # plot truth
    if plot_truth:
        plt.semilogy(r, (1-frac) * comp1 + frac * comp2,
                     label="Truth", color="steelblue", lw=4, zorder=2)
    
    N_n = len([lab for lab in labels if "n" in lab])
    N_theta = len([lab for lab in labels if "theta" in lab])
    
    for sample_k in samples_eq_bs:
        
        n_s_k = sample_k[:N_n]
        
        if N_theta > 0:
            theta_s_pix_k = np.append([theta_s_pix[0]],
                                      10**sample_k[N_n:-2] / pixel_scale) 
        else:
            theta_s_pix_k = theta_s_pix 
            
        comp2_k = multi_power1d_normed(r, n_s_k, theta_s_pix_k) / c_mpow2Dto1D

        plt.semilogy(r, (1-frac) * comp1 + frac * comp2_k,
                     color="lightblue", lw=2, alpha=0.1, zorder=1)
        
    for fits, c, ls, lab in zip([pmed, pmean], ["royalblue", "b"],
                              ["-.","--"], ["mean", "med"]):

        n_s_fit = fits[:N_n]

        if N_theta > 0:
            theta_s_pix_fit = np.append([theta_s_pix[0]],
                                        10**fits[N_n:-2] / pixel_scale)
        else:
            theta_s_pix_fit = theta_s_pix 

        comp2 = multi_power1d_normed(r,n_s=n_s_fit, theta_s=theta_s_pix_fit) / c_mpow2Dto1D
        y_fit = (1-frac) * comp1 + frac * comp2

        plt.semilogy(r, y_fit, color=c, lw=2.5, ls=ls,
                     alpha=0.8, label=lab+' comb.', zorder=4)

        if lab=="med":
            plt.semilogy(r, (1-frac) * comp1,
                         color="orange", lw=2, ls="--",
                         alpha=0.7, label="med core", zorder=4)
            plt.semilogy(r, frac * comp2,
                         color="seagreen", lw=2, ls="--",
                         alpha=0.7, label="med aureole", zorder=4)

        std_fit = 10**fits[-1]
        contrast = Amp_max / std_fit
        plt.axhline(y_fit.max()/contrast, color="k", ls="--", alpha=0.9)
            
    for theta in theta_s_pix:
        plt.axvline(theta, ls=":", color="gray",alpha=0.3)
    if N_theta > 0:    
        for logtheta_fit in fits[N_n:-2]:
            plt.axvline(10**logtheta_fit / pixel_scale, ls="-", color="k",alpha=0.7)
    
    if r_core is not None:
        r_max = r[np.argmin(abs(y_fit-y_fit.max()/contrast))]
        plt.axvspan(np.atleast_1d(r_core).max(), r_max,
                    color='steelblue', alpha=0.15, zorder=1)
        plt.axvspan(plt.gca().get_xlim()[0], np.atleast_1d(r_core).min(),
                    color='gray', alpha=0.15, zorder=1)
    plt.legend(loc=1, fontsize=12)    
    plt.xlabel(r"$\rm r\,[pix]$",fontsize=18)
    plt.ylabel(r"$\rm Intensity$",fontsize=18)
    plt.title("Recovered PSF from Fitting vs Truth",fontsize=18)
    plt.text(1, y_fit.max()/contrast*1.5, '1 $\sigma$', fontsize=10)
    plt.xscale("log")
    plt.ylim(1e-9, 0.5)    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(dir_name, "Fit_PSF.png"), dpi=120)
        plt.close()
        
def draw2D_fit_vs_truth_PSF_mpow(res,  psf, stars, labels,
                                 image_base=None, vmin=None, vmax=None,
                                 avg_func='median', save=False, dir_name="."):
    """ Compare 2D fit and truth image """
    N_n = len([lab for lab in labels if "n" in lab])
    N_theta = len([lab for lab in labels if "theta" in lab])
    
    pmed, pmean, pcov = get_params_fit(res)
    fits = pmed if avg_func=='median' else pmean
    print("Fitting (mean) : ", np.around(pmean,3))
    print("Fitting (median) : ", np.around(pmed,3))
    
    n_s_fit = fits[:N_n]
    if N_theta > 0:
        theta_s_fit = np.append([psf.theta_s[0]], 10**fits[N_n:-2])
    else:
        theta_s_fit = psf.theta_s
    
    mu_fit, sigma_fit = fits[-2], 10**fits[-1]
    noise_fit = make_noise_image(psf.image_size, sigma_fit)
    
    psf_fit = psf.copy()
    psf_fit.update({'n_s':n_s_fit, 'theta_s': theta_s_fit})
    
    image_fit = generate_mock_image(psf_fit, stars, draw_real=True)
    image_fit = image_fit + mu + noise_fit
    
    if image_base is not None:
        image_fit += image_base
        
    if vmin is None:
        vmin = mu_fit - 0.3 * sigma_fit
    if vmax is None:
        vmax = vmin + 11
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))
    im = ax1.imshow(image_fit, vmin=vmin, vmax=vmax, norm=norm1); colorbar(im)
    im = ax2.imshow(image, vmin=vmin, vmax=vmax, norm=norm1); colorbar(im)
    Diff = (image_fit-image)/image
    im = ax3.imshow(Diff, vmin=-0.1, vmax=0.1, cmap='seismic'); colorbar(im)
    ax1.set_title("Fit: I$_f$")
    ax2.set_title("Original: I$_0$")
    ax3.set_title("Frac.Diff: (I$_f$ - I$_0$) / I$_0$")
    
    plt.tight_layout()   
    if save:
        plt.savefig(os.path.join(dir_name,
                                 "Fit_vs_truth_image.png"), dpi=120)
        plt.close()
        
def draw_comparison_fit_data(image_fit, data, 
                             noise_fit, mask_fit, vmin=None, vmax=None,
                             save=False, dir_name=".", suffix=""):
    # Difference between real vs conv
    
    image_fit = image_fit + noise_fit
    
    if vmin is None:
        vmin = np.median(image_fit[~mask_fit]) - 1
    if vmax is None:
        vmax = vmin + 150
        
    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))
    im = ax1.imshow(image_fit, vmin=vmin, vmax=vmax, norm=norm1, cmap="gnuplot2")
    ax1.set_title("Fitted I$_f$"); colorbar(im)
    
    im = ax2.imshow(data, vmin=vmin, vmax=vmax, norm=norm1, cmap="gnuplot2")    
    ax2.set_title("Data I$_0$"); colorbar(im)
    
    diff = (image_fit-data)/data
    im = ax3.imshow(diff, vmin=-0.1, vmax=0.1, cmap="seismic")
    diff = (image_fit-data)/data
    ax3.set_title("(I$_f$ - I$_0$) / I$_0$"); colorbar(im)
    
    diff[mask_fit] = np.nan
    im = ax4.imshow(diff, vmin=-0.05, vmax=0.05, cmap="seismic")
    ax4.set_title("(I$_f$ - I$_0$) / I$_0$  (w/ mask)"); colorbar(im)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir_name, "Comparison_fit_data%s.png"%suffix), dpi=120)
        plt.close()
    
# def plot_fit_PSF(res, image_size=image_size, 
#                  n_bootstrap=200, save=False, dir_name="./"):

#     samples = res.samples                                 # samples
#     weights = np.exp(res.logwt - res.logz[-1])            # normalized weights 
#     pmean, pcov = dyfunc.mean_and_cov(samples, weights)     # weighted mean and covariance
#     samples_eq = dyfunc.resample_equal(samples, weights)  # resample weighted samples
#     pmed = np.median(samples_eq,axis=0)                    # median sample
    
#     print("Fitting (mean): ", pmean)
#     print("Fitting (median): ", pmed)
    
#     from astropy.stats import bootstrap
#     samples_eq_bs = bootstrap(samples_eq, bootnum=1, samples=n_bootstrap)[0]
    
#     c_mof2Dto1D =  C_mof2Dto1D(gamma_pix, beta_psf)
#     c_pow2Dto1D = C_pow2Dto1D(n, theta_t_pix)
    
#     r = np.logspace(0., np.log10(image_size//2), 100)
#     comp1 = moffat1d_normed(r, gamma=gamma_pix, alpha=beta_psf) / c_mof2Dto1D
    
#     plt.figure(figsize=(7,6))
    
#     for (logfrac_k, n_k, _, _) in samples_eq_bs:
#         frac_k = 10**logfrac_k
#         comp2_k = trunc_power1d_normed(r, n_k, theta_t_pix) / c_pow2Dto1D

#         plt.semilogy(r, (1-frac_k) * comp1 + frac_k * comp2_k,
#                      color="lightblue", lw=1.5,alpha=0.1,zorder=1)
#     else:
#         for fits, c, ls, l in zip([pmed, pmean], ["royalblue", "b"],
#                                   ["-.","--"], ["mean", "med"]):
#             f_fit = 10**fits[0]
#             comp2 = trunc_power1d_normed(r, fits[1], theta_t_pix) / c_pow2Dto1D
#             y_fit = (1-f_fit) * comp1 + f_fit * comp2
            
#             plt.semilogy(r, y_fit, color=c, lw=2.5, ls=ls, alpha=0.8, label=l, zorder=4)
#             if l=="med":
#                 plt.semilogy(r, (1-f_fit) * comp1,
#                              color="orange", lw=2, ls="--", alpha=0.7, label="med mof",zorder=4)
#                 plt.semilogy(r, f_fit * comp2,
#                              color="seagreen", lw=2, ls="--", alpha=0.7, label="med pow",zorder=4)
                
#             std_fit = fits[-1]
#             Amp_max = moffat2d_Flux2Amp(gamma_pix, beta_psf, Flux=(1-f_fit)*Flux.max())
#             contrast = Amp_max/(std_fit)
#             y_min_contrast = y_fit.max()/contrast
#             plt.axhline(y_min_contrast, color="k", ls="-.", alpha=0.5)
#             plt.axhline(y_min_contrast*2, color="k", ls=":", alpha=0.5)
            
#     plt.axvspan(12, 24, color="seagreen",  alpha=0.1)
#     r_max = r[np.argmin(abs(y_min_contrast-y_fit))]
#     plt.axvspan(24, r_max, color="plum", alpha=0.1)
#     plt.text(1, y_min_contrast*1.15, "1 $\sigma$", fontsize=9)
#     plt.text(1, y_min_contrast*2.35, "2 $\sigma$", fontsize=9)
#     plt.axvspan(1, 12, color="gray", alpha=0.2)
    
#     plt.legend(fontsize=12)    
#     plt.xlabel(r"$\rm r\,[pix]$",fontsize=18)
#     plt.ylabel(r"$\rm Intensity$",fontsize=18)
#     plt.title("Recovered PSF from Fitting",fontsize=18)
#     plt.xscale("log")
#     plt.xlim(0.9, 3*r_max)    
#     plt.ylim(1e-8, 0.5)    
#     plt.tight_layout()
#     if save:
#         plt.savefig("%s/Fit_PSF.png"%dir_name,dpi=150)
#         plt.close()
        
    
def plot_fit_res_SB(params, psf, r=np.logspace(0.03,2.5,100), mags=[15,12,9], ZP=27.1):
    
    print("Fit results: ", params)
    # PSF Parameters
    n_fit = params[1]                     # true power index
    f_fit = 10**params[0]                 # fraction of power law component
    
    c_mof2Dto1D =  C_mof2Dto1D(psf.gamma_pix, psf.beta)
    c_pow2Dto1D = C_pow2Dto1D(n_fit, psf.theta_0_pix)

    I_A, I_B, I_C = [10**((mag-ZP)/-2.5) for mag in mags]
    comp1 = moffat1d_normed(r, gamma=psf.gamma_pix, alpha=psf.beta) / c_mof2Dto1D 
    comp2 = trunc_power1d_normed(r, n_fit, psf.theta_0_pix) / c_pow2Dto1D

    [I_tot_A, I_tot_B, I_tot_C] = [Intensity2SB(((1-f_fit) * comp1 + comp2 * f_fit) * I,
                                                0, ZP, psf.pixel_scale)
                                   for I in [I_A, I_B, I_C]]
    return I_tot_A, I_tot_B, I_tot_C