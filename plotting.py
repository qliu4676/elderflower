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
                  pad=0, save=False, save_path='./'):
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
    
    star_pos_A = stars.star_pos_verybright
    star_pos_B = stars.star_pos_medbright
    
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
        plt.savefig(os.path.join(save_path, "Mask.png"), dpi=150)
        plt.close()

def draw_mask_map_strip(image, seg_comb, mask_comb, stars,
                        ma_example=None,
                        
                        r_core=24, vmin=884, vmax=1e3,
                        pad=0, save=False, save_path='./'):
    """ Visualize mask map w/ strips """
    
    from matplotlib import patches
    
    star_pos_A = stars.star_pos_verybright
    star_pos_B = stars.star_pos_medbright
    
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
        plt.savefig(os.path.join(save_path, "Mask_strip.png"), dpi=150)
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
                      psf_range=400):
    
    plt.figure(figsize=(6,5))

    r = np.logspace(0, np.log10(psf_range), 100)
    comp1, comp2 = f_core(r), f_aureole(r)

    plt.plot(r, np.log10((1-frac) * comp1 + comp2 * frac),
             ls="-", lw=3,alpha=0.9, zorder=5, label='combined')
    plt.plot(r, np.log10((1-frac) * comp1),
             ls="--", lw=3, alpha=0.9, zorder=1, label='core')
    plt.plot(r, np.log10(comp2 * frac),
             ls="--", lw=3, alpha=0.9, label='aureole')

    plt.legend(loc=1, fontsize=12)
    plt.xscale('log')
    plt.xlabel('r [pix]', fontsize=14)
    plt.ylabel('log Intensity', fontsize=14)
    plt.ylim(-9, -0.5)
    
def plot_PSF_model_galsim(psf_inner, psf_outer, params,
                          image_size, pixel_scale,
                          contrast=None, aureole_model="Power"):
    
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

    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(frac*img_outer, pix_scale=pixel_scale, seeing=2.5, 
                                                  core_undersample=True, color="g",
                                                  xunit="pix", yunit="intensity", label=aureole_model)
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d((1-frac)*img_inner, pix_scale=pixel_scale, seeing=2.5, 
                                                  core_undersample=True, color="orange",
                                                  xunit="pix", yunit="intensity", label="Moffat")
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(img_star, pix_scale=pixel_scale, seeing=2.5,  
                                                  core_undersample=True, 
                                                  xunit="pix", yunit="intensity", label="Combined")

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
    
def plot_flux_dist(Flux, Flux_thresholds):
    import seaborn as sns
    F_bright, F_verybright = Flux_thresholds
    sns.distplot(np.log10(Flux), label="log Flux (tot)", color='plum')
    plt.axvline(np.log10(F_bright), color="k", ls="-",alpha=0.8)
    plt.axvline(np.log10(F_verybright), color="k", ls="--",alpha=0.8)
    plt.axvspan(plt.gca().get_xlim()[0], np.log10(F_bright),
                color='gray', alpha=0.2, zorder=1)
    plt.axvspan(np.log10(F_bright), np.log10(F_verybright),
                color='seagreen', alpha=0.2, zorder=1)
    plt.axvspan(np.log10(F_verybright), plt.gca().get_xlim()[1],
                color='steelblue', alpha=0.2, zorder=1)
    plt.xlabel('Estimated Total Flux', fontsize=15)
    plt.show()