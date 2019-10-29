from modeling import *


def draw_mask_map(image, seg_map, mask_deep,
                  star_pos_A, star_pos_B,
                  r_core=24, vmin=884, vmax=1e3,
                  pad=0, save=False, save_path='./'):
    """ Visualize mask map """
    from matplotlib import patches
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    im1 = ax1.imshow(image, cmap='gray', norm=norm1, vmin=vmin, vmax=1e4, aspect='auto')
    ax1.set_title("Mock")
    colorbar(im1)

    ax2.imshow(seg_map, cmap="gnuplot2")
    ax2.set_title("Deep Mask")

    image2 = image.copy()
    image2[mask_deep] = 0
    im3 = ax3.imshow(image2, cmap='gnuplot2', norm=norm2, vmin=vmin, vmax=vmax, aspect='auto') 
    ax3.set_title("'Sky'")
    colorbar(im3)
    
    if np.ndim(r_core) == 0:
        r_core = [r_core, r_core]
    
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

def draw_mask_map_strip(image, seg_comb, mask_comb,
                        mask_strip, mask_cross,
                        star_pos_A, star_pos_B,
                        r_core=24, vmin=884, vmax=1e3,
                        pad=0, save=False, save_path='./'):
    """ Visualize mask map w/ strips """
    
    from matplotlib import patches
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    mask_strip[mask_cross.astype(bool)]=0.5
    ax1.imshow(mask_strip, cmap="gray_r")
    ax1.plot(star_pos_A[-1][0], star_pos_A[-1][1], "r*",ms=15)
    ax1.set_title("Mask Strip")

    ax2.imshow(seg_comb, cmap="gnuplot2")
    ax2.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=15)
    ax2.set_title("Mask Comb.")

    image3 = image.copy()
    image3[mask_comb] = 0
    im3 = ax3.imshow(image3, cmap='gnuplot2', norm=norm1, aspect='auto', vmin=vmin, vmax=vmax) 
    ax3.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=15)
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
        
def display_background_distribution(image, mask_deep):
    # Check background, fit with gaussian and exp-gaussian distribution
    import seaborn as sns
    plt.figure(figsize=(6,4))
    z_sky = image[~mask_deep]
    sns.distplot(z_sky, label='Data', hist_kws={'alpha':0.3})

    mu_fit, std_fit = stats.norm.fit(z_sky)
    print(mu_fit, std_fit)
    d_mod = stats.norm(loc=mu_fit, scale=std_fit)
    x = np.linspace(d_mod.ppf(0.001), d_mod.ppf(0.999), 100)
    plt.plot(x, d_mod.pdf(x), 'g-', lw=2, alpha=0.6, label='Normal')

    K_fit, mu_fit, std_fit = stats.exponnorm.fit(z_sky)
    print(K_fit, mu_fit, std_fit)
    d_mod2 = stats.exponnorm(loc=mu_fit, scale=std_fit, K=K_fit)
    x = np.linspace(d_mod2.ppf(0.001), d_mod2.ppf(0.9999), 100)
    plt.plot(x, d_mod2.pdf(x), 'r-', lw=2, alpha=0.6, label='exp norm')

    plt.legend(fontsize=12)
    
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