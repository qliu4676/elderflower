import os
import numpy as np

from copy import deepcopy

try: 
    import seaborn as sns
    seaborn_plot = True
except ImportError:
    import warnings
    warnings.warn("Seaborn is not installed. Plot with matplotlib.")
    seaborn_plot = False

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from matplotlib import rcParams
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'gnuplot2'
plt.rcParams["font.serif"] = "Times New Roman"
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
rcParams.update({'axes.labelsize': 16})
rcParams.update({'font.size': 16})

from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, AsinhStretch, HistEqStretch
from astropy.stats import mad_std

from photutils import CircularAperture


### Plotting Helpers ###

def LogNorm():
    return ImageNormalize(stretch=LogStretch())

def AsinhNorm(a=0.1):
    return ImageNormalize(stretch=AsinhStretch(a=a))

def HistEqNorm(data):
    return ImageNormalize(stretch=HistEqStretch(data))

def vmin_3mad(img):
    """ lower limit of visual imshow defined by 3 mad above median """ 
    return np.median(img)-3*mad_std(img)

def vmax_2sig(img):
    """ upper limit of visual imshow defined by 2 sigma above median """ 
    return np.median(img)+2*np.std(img)

def colorbar(mappable, pad=0.2, size="5%", loc="right", color_nan='gray', **args):
    """ Customized colorbar """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    
    if loc=="bottom":
        orent = "horizontal"
        pad = 1.5*pad
        rot = 75
    else:
        orent = "vertical"
        rot = 0
    
    cax = divider.append_axes(loc, size=size, pad=pad)
    
    cb = fig.colorbar(mappable, cax=cax, orientation=orent, **args)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(),rotation=rot)
    
    cmap = cb.mappable.get_cmap()
    cmap.set_bad(color=color_nan, alpha=0.3)
    
    return cb

def make_rand_cmap(n_label, rand_state = 12345):
    from photutils.utils import make_random_cmap
    rand_cmap = make_random_cmap(n_label, random_state=rand_state)
    rand_cmap.set_under(color='black')
    rand_cmap.set_over(color='white')
    return rand_cmap

def make_rand_color(n_color, seed=1234,
                    colour = ["indianred", "plum", "seagreen", "lightcyan",
                              "orchid", 'gray', 'orange', 'yellow', "brown" ]):
    import random
    random.seed(seed)
    rand_colours = [random.choice(colour) for i in range(n_color)]
    return rand_colours

def draw_mask_map(image, seg_map, mask_deep, stars,
                  r_core=None, r_out=None, vmin=None, vmax=None,
                  pad=0, save=False, save_dir='./'):
    """ Visualize mask map """
    
    from matplotlib import patches
    
    mu = np.nanmedian(image)
    std = mad_std(image)
    
    if vmin is None:
        vmin = mu - std
    if vmax is None:
        vmax = mu + 10*std
    
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    
    im1 = ax1.imshow(image, cmap='gray', norm=LogNorm(), vmin=vmin, vmax=1e4)
    ax1.set_title("Image")
    
    n_label = seg_map.max()
    ax2.imshow(seg_map, vmin=1, vmax=n_label-2, cmap=make_rand_cmap(n_label))
    ax2.set_title("Deep Mask")

    image2 = image.copy()
    image2[mask_deep] = 0
    im3 = ax3.imshow(image2, norm=LogNorm(), vmin=vmin, vmax=vmax, aspect='auto') 
    ax3.set_title("'Sky'")
    colorbar(im3)
    
    if r_core is not None:
        if np.ndim(r_core) == 0:
            r_core = [r_core, r_core]
    
        star_pos_A = stars.star_pos_verybright + pad
        star_pos_B = stars.star_pos_medbright + pad

        aper = CircularAperture(star_pos_A, r=r_core[0])
        aper.plot(color='lime',lw=2,label="",alpha=0.9, axes=ax3)
        
        aper = CircularAperture(star_pos_B, r=r_core[1])
        aper.plot(color='c',lw=2,label="",alpha=0.7, axes=ax3)
    
        if r_out is not None:
            aper = CircularAperture(star_pos_A, r=r_out[0])
            aper.plot(color='lime',lw=1.5,label="",alpha=0.9, axes=ax3)
            aper = CircularAperture(star_pos_B, r=r_out[1])
            aper.plot(color='c',lw=1.5,label="",alpha=0.7, axes=ax3)
    
    patch_size = image.shape[0] - pad * 2
    rec = patches.Rectangle((pad, pad), patch_size, patch_size, facecolor='none',
                            edgecolor='w', linewidth=2, linestyle='--',alpha=0.8)
    ax3.add_patch(rec)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(save_dir, "Mask_dual.png"), dpi=120)
        plt.show()
        plt.close()
    else:
        plt.show()

def draw_mask_map_strip(image, seg_comb, mask_comb, stars,
                        ma_example=None, r_core=None, vmin=None, vmax=None,
                        pad=0, save=False, save_dir='./'):
    """ Visualize mask map w/ strips """
    
    from matplotlib import patches
    
    star_pos_A = stars.star_pos_verybright + pad
    star_pos_B = stars.star_pos_medbright + pad
    
    if ma_example is not None:
        mask_strip, mask_cross = ma_example
    
    mu = np.nanmedian(image)
    std = mad_std(image)
    if vmin is None:
        vmin = mu - std
    if vmax is None:
        vmax = mu + 10*std
        
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    
    mask_strip[mask_cross.astype(bool)]=0.5
    ax1.imshow(mask_strip, cmap="gray_r")
    ax1.plot(star_pos_A[0][0], star_pos_A[0][1], "r*",ms=18)
    ax1.set_title("Strip/Cross")
    
    n_label = seg_comb.max()
    ax2.imshow(seg_comb, vmin=1, vmax=n_label-3, cmap=make_rand_cmap(n_label))
    ax2.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=18)
    ax2.set_title("Mask Comb.")

    image3 = image.copy()
    image3[mask_comb] = 0
    im3 = ax3.imshow(image3, norm=LogNorm(), aspect='auto', vmin=vmin, vmax=vmax) 
    ax3.plot(star_pos_A[:,0], star_pos_A[:,1], "r*",ms=18)
    ax3.set_title("'Sky'")
    colorbar(im3)
    
    if r_core is not None:
        if np.ndim(r_core) == 0:
            r_core = [r_core, r_core]
            
        aper = CircularAperture(star_pos_A, r=r_core[0])
        aper.plot(color='lime',lw=2,label="",alpha=0.9, axes=ax3)
        
        aper = CircularAperture(star_pos_B, r=r_core[1])
        aper.plot(color='c',lw=2,label="",alpha=0.7, axes=ax3)

    size = image.shape[0] - pad * 2
    rec = patches.Rectangle((pad, pad), size, size, facecolor='none',
                            edgecolor='w', linewidth=2, linestyle='--',alpha=0.8)
    ax3.add_patch(rec)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "Mask_strip.png"), dpi=120)
        plt.show()
        plt.close()
    else:
        plt.show()

        
def Fit_background_distribution(image, mask_deep):
    # Check background, fit with gaussian and exp-gaussian distribution
    from scipy import stats
    
    plt.figure(figsize=(6,4))
    z_sky = image[~mask_deep]
    if seaborn_plot:
        sns.distplot(z_sky, label='Data', hist_kws={'alpha':0.3})
    else:
        plt.hist(z_sky, label='Data', alpha=0.3)

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
    
def plot_PSF_model_1D(frac, f_core, f_aureole, psf_range=400,
                      yunit='Intensity', label='combined', log_scale=True,
                      ZP=27.1, pixel_scale=2.5, decompose=True):
    from .utils import Intensity2SB
    
    r = np.logspace(0, np.log10(psf_range), 100)
    
    I_core = (1-frac) * f_core(r)
    I_aureole = frac * f_aureole(r)
    I_tot = I_core + I_aureole
    
    if log_scale:
        I_core, I_aureole, I_tot = np.log10(I_core), np.log10(I_aureole), np.log10(I_tot) 
    
    if yunit=='Intensity':
        plt.semilogx(r, I_tot,
                 ls="-", lw=3,alpha=0.9, zorder=5, label=label)
        if decompose:
            plt.semilogx(r, I_core,
                     ls="--", lw=3, alpha=0.9, zorder=1, label='core')
            plt.semilogx(r, I_aureole,
                     ls="--", lw=3, alpha=0.9, label='aureole')
        plt.ylabel('log Intensity', fontsize=14)
        plt.ylim(I_aureole.min(), I_tot.max()+0.25)
        
    elif yunit=='SB':
        plt.semilogx(r, -14.5+Intensity2SB(I=I_tot, BKG=0,
                                           ZP=27.1, pixel_scale=pixel_scale),
                     ls="-", lw=3,alpha=0.9, zorder=5, label=label)
        if decompose:
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

    
def plot_PSF_model_galsim(psf, image_size=800, contrast=None,
                          figsize=(7,6), save=False, save_dir='.'):
    """ Plot and 1D PSF model and Galsim 2D model averaged in 1D """
    from .utils import Intensity2SB, cal_profile_1d
    
    pixel_scale = psf.pixel_scale
    
    frac = psf.frac
    psf_core = psf.psf_core
    psf_aureole = psf.psf_aureole
    
    psf_star = psf.psf_star
    
    img_core = psf_core.drawImage(scale=pixel_scale, method="no_pixel")
    img_aureole = psf_aureole.drawImage(nx=201, ny=201, scale=pixel_scale, method="no_pixel")
    img_star = psf_star.drawImage(nx=image_size, ny=image_size, scale=pixel_scale, method="no_pixel")
    
    if figsize is not None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(frac*img_aureole.array, color="g",
                                                  pixel_scale=pixel_scale,
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label=psf.aureole_model)
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d((1-frac)*img_core.array, color="orange",
                                                  pixel_scale=pixel_scale, 
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label="Moffat")
    r_rbin, z_rbin, logzerr_rbin = cal_profile_1d(img_star.array,
                                                  pixel_scale=pixel_scale, 
                                                  core_undersample=True, mock=True,
                                                  xunit="pix", yunit="Intensity",
                                                  label="Combined")

    plt.legend(loc=1, fontsize=12)
    
    r = np.logspace(0, np.log10(image_size), 100)
    comp1 = psf.f_core1D(r)
    comp2 = psf.f_aureole1D(r)
    
    plt.plot(r, np.log10((1-frac) * comp1 + comp2 * frac), ls="-", lw=3, zorder=5)
    plt.plot(r, np.log10((1-frac) * comp1), ls="--", lw=3, zorder=1)
    plt.plot(r, np.log10(comp2 * frac), ls="--", lw=3)
    
    if psf.aureole_model == "multi-power":
        for t in psf.theta_s_pix:
            plt.axvline(t, ls="--", color="k",alpha=0.3, zorder=1)
        
    if contrast is not None:
        plt.axhline(np.log10(comp1.max()/contrast),color="k",ls="--")
        
    plt.title("Model PSF",fontsize=14)
    plt.ylim(-8.5, -0.5)
    plt.xlim(r_rbin.min()*0.8, r_rbin.max()*1.2)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "Model_PSF.png"), dpi=120)
        plt.close()
        
    return img_star
    
def plot_flux_dist(Flux, Flux_thresholds, ZP=None,
                   save=False, save_dir='.', figsize=None, **kwargs):
    import seaborn as sns
    F_bright, F_verybright = Flux_thresholds
    if figsize is not None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    plt.axvline(np.log10(F_bright), color="k", ls="-",alpha=0.7, zorder=1)
    plt.axvline(np.log10(F_verybright), color="k", ls="--",alpha=0.7, zorder=1)
    plt.axvspan(1, np.log10(F_bright),
                color='gray', alpha=0.15, zorder=0)
    plt.axvspan(np.log10(F_bright), np.log10(F_verybright),
                color='seagreen', alpha=0.15, zorder=0)
    plt.axvspan(np.log10(F_verybright), 9,
                color='steelblue', alpha=0.15, zorder=0)
    
    if seaborn_plot:
        sns.distplot(np.log10(Flux), kde=False, **kwargs)
    else:
        plt.hist(np.log10(Flux), alpha=0.5)
    
    plt.yscale('log')
    plt.xlabel('Estimated log Flux$_{tot}$ / Mag', fontsize=15)
    plt.ylabel('# of stars', fontsize=15)
    plt.legend(loc=1)
    
    if ZP is not None:
        ax1 = plt.gca()
        xticks1 = ax1.get_xticks()
        ax2 = ax1.twiny()
        ax2.set_xticks(xticks1)
        ax2.set_xticklabels(np.around(-2.5*xticks1+ZP ,1))
        ax2.set_xbound(ax1.get_xbound())
            
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "Flux_dist.png"), dpi=80)
        plt.show()
        plt.close()

def draw_independent_priors(priors, xlabels=None, plabels=None,
                            save=False, save_dir='./'):
    
    x_s = [np.linspace(d.ppf(0.01), d.ppf(0.99), 100) for d in priors]
    
    fig, axes = plt.subplots(1, len(priors), figsize=(15,4))
    for k, ax in enumerate(axes):
        ax.plot(x_s[k], priors[k].pdf(x_s[k]),'-', lw=5, alpha=0.6, label=plabels[k])
        ax.legend()
        if xlabels is not None:
            ax.set_xlabel(xlabels[k], fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "Prior.png"), dpi=100)
        plt.close()

        
def draw_cornerplot(results, ndim, labels=None, truths=None, figsize=(16,14),
                    save=False, save_dir='.', suffix=''):
    from dynesty import plotting as dyplot
    
    fig = plt.subplots(ndim, ndim, figsize=figsize)
    dyplot.cornerplot(results, truths=truths, labels=labels, 
                      color="royalblue", truth_color="indianred",
                      title_kwargs={'fontsize':18, 'y': 1.04},
                      label_kwargs={'fontsize':16}, 
                      show_titles=True, fig=fig)
    
    if save:
        plt.savefig(os.path.join(save_dir, "Cornerplot%s.png"%suffix), dpi=150)
        plt.show()
        plt.close()
    else:
        plt.show()

        
def draw2D_fit_vs_truth_PSF_mpow(results,  psf, stars, labels, image,
                                 image_base=None, vmin=None, vmax=None,
                                 avg_func='median', save=False, save_dir="."):
    """ Compare 2D fit and truth image """
    from .sampler import get_params_fit
    
    N_n = len([lab for lab in labels if "n" in lab])
    N_theta = len([lab for lab in labels if "theta" in lab])
    
    pmed, pmean, pcov = get_params_fit(results)
    fits = pmed if avg_func=='median' else pmean
    print("Fitting (mean) : ", np.around(pmean,3))
    print("Fitting (median) : ", np.around(pmed,3))
    
    n_s_fit = fits[:N_n]
    if N_theta > 0:
        theta_s_fit = np.append([psf.theta_s[0]], 10**fits[N_n:N_n+N_theta])
    else:
        theta_s_fit = psf.theta_s
    
    mu_fit, sigma_fit = fits[-2], 10**fits[-1]
    noise_fit = make_noise_image(psf.image_size, sigma_fit)
    
    psf_fit = psf.copy()
    psf_fit.update({'n_s':n_s_fit, 'theta_s': theta_s_fit})
    
    psf_range = psf.image_size * psf.pixel_scale
    image_fit = generate_image_by_flux(psf_fit, stars, draw_real=True,
                                       psf_range=[psf_range//2, psf_range])
    image_fit = image_fit + mu_fit + noise_fit
    
    if image_base is not None:
        image_fit += image_base
        
    if vmin is None:
        vmin = mu_fit - 0.3 * sigma_fit
    if vmax is None:
        vmax = vmin + 11
        
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))
    im = ax1.imshow(image_fit, vmin=vmin, vmax=vmax, norm=LogNorm()); colorbar(im)
    im = ax2.imshow(image, vmin=vmin, vmax=vmax, norm=LogNorm()); colorbar(im)
    Diff = (image_fit-image)/image
    im = ax3.imshow(Diff, vmin=-0.1, vmax=0.1, cmap='seismic'); colorbar(im)
    ax1.set_title("Fit: I$_f$")
    ax2.set_title("Original: I$_0$")
    ax3.set_title("Frac.Diff: (I$_f$ - I$_0$) / I$_0$")
    
    plt.tight_layout()   
    if save:
        plt.savefig(os.path.join(save_dir,
                                 "Fit_vs_truth_image.png"), dpi=120)
        plt.close()
        
def draw_comparison_2D(image_fit, data, mask, image_star,
                       noise_fit=0, r_core=None,
                       vmin=None, vmax=None,
                       cmap='gnuplot2', norm=AsinhNorm(0.01),
                       save=False, save_dir=".", suffix=""):
                       
    """ Compare data and fit in 2D """
    
    mask_fit = getattr(mask, 'mask_comb', mask.mask_deep)
    
    if vmin is None:
        vmin = np.median(image_fit[~mask_fit]) - 5
    if vmax is None:
        vmax = vmin + 200
        
    norm2 = deepcopy(norm)
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=(16,9))
    
    im = ax1.imshow(data, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
    ax1.set_title("Data [I$_0$]", fontsize=15); colorbar(im)
    
    im = ax2.imshow(image_fit+noise_fit, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
    ax2.set_title("Fit [I$_f$]", fontsize=15); colorbar(im)
    
    im = ax3.imshow(image_star, vmin=0, vmax=vmax-vmin, norm=norm2, cmap=cmap)    
    ax3.set_title("Bright Stars [I$_{f,B}$]", fontsize=15); colorbar(im)
    
    frac_diff = (image_fit-data)/data
#     frac_diff[mask_fit] = 0
    im = ax4.imshow(frac_diff, vmin=-0.1, vmax=0.1, cmap="seismic")
    ax4.set_title("Frac. Diff. [(I$_f$ - I$_0$)/I$_0$]", fontsize=15); colorbar(im)
    
#     noise = np.sqrt((data/0.37/618)**2+(2/0.37/618)**2)
#     chi = (image_fit-data)/noise
#     im = ax4.imshow(chi, vmin=-10, vmax=10, cmap="seismic")
#     ax4.set_title("Chi. [(I$_f$ - I$_0$)/$\sigma_0$]", fontsize=15); colorbar(im)
    
    residual = (data-image_star)
    im = ax5.imshow(residual, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
    ax5.set_title("Bright Subtracted [I$_0$ - I$_{f,B}$]", fontsize=15); colorbar(im)
    
    residual[mask_fit] = 0
    im = ax6.imshow(residual, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
    ax6.set_title("Bright Subtracted (masked)"); colorbar(im)
    
    if r_core is not None:
        if np.ndim(r_core) == 0:
            r_core = [r_core,r_core]
        aper1 = CircularAperture(mask.stars.star_pos_verybright, r=r_core[0])
        aper1.plot(color='lime',lw=2,alpha=0.9, axes=ax6)
        aper2 = CircularAperture(mask.stars.star_pos_medbright, r=r_core[1])
        aper2.plot(color='skyblue',lw=2,label="",alpha=0.7, axes=ax6)
        
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, "Comparison_fit_data2D%s.png"%suffix), dpi=120)
        plt.show()
        plt.close()
    else:
        plt.show()
    
    
def plot_fit_PSF1D(results, psf,
                   image_size=800, n_spline=2,
                   n_bootstrap=500, truth=None,  
                   Amp_max=None, r_core=None,
                   save=False, save_dir="./",
                   suffix='', figsize=(7,6)):

    from astropy.stats import bootstrap
    from .sampler import get_params_fit
    
    pixel_scale = psf.pixel_scale
    
    frac = psf.frac
    
    if figsize is not None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    
    if truth is not None:
        print("Truth : ", psf.params)
        psf.plot1D(psf_range=600, decompose=False, label='Truth')
    
    # read fitting results
    pmed, pmean, pcov, samples_eq = get_params_fit(results, return_sample=True)
    print("Fitting (mean) : ", np.around(pmean,3))
    print("Fitting (median) : ", np.around(pmed,3))
    
    samples_eq_bs = bootstrap(samples_eq, bootnum=1, samples=n_bootstrap)[0]
    
    # Number of n and theta in the fitting
    if psf.aureole_model != "moffat":
        theta_0 = psf.theta_0
        N_n = n_spline
        N_theta = n_spline - 1
    
    psf_fit = psf.copy()
    
    r = np.logspace(0., np.log10(image_size), 100)
    comp1 = psf.f_core1D(r)
    
    # Sample distribution from joint PDF
    for sample in samples_eq_bs:
        frac_k = frac
        
        if psf.aureole_model == "moffat":
            gamma1_k = sample[0]
            beta1_k = sample[1]
            psf_fit.update({'gamma1':gamma1_k, 'beta1':beta1_k})
            
        else:
            n_c = psf.n_c
            theta_c = psf.theta_c
            
            if psf.aureole_model == "power":
                n_k = sample[0]
                psf_fit.update({'n':n_k})

            elif psf.aureole_model == "multi-power":
                n_s_k = np.concatenate([sample[:N_n], [n_c]])
                theta_s_k = np.concatenate([[theta_0],
                                            np.atleast_1d(10**sample[N_n:N_n+N_theta]),
                                            [theta_c]])
                psf_fit.update({'n_s':n_s_k, 'theta_s':theta_s_k})
            
        comp2_k = psf_fit.f_aureole1D(r)
            
        plt.semilogy(r, (1-frac_k) * comp1 + frac_k * comp2_k,
                     color="lightblue", lw=2,alpha=0.1,zorder=1)
        
    # Median and mean fitting
    for fits, c, ls, lab in zip([pmed, pmean], ["royalblue", "b"],
                              ["-.","-"], ["mean", "med"]):
        
        if psf.aureole_model == "moffat":
            gamma1_fit = fits[0]
            beta1_fit = fits[1]
            psf_fit.update({'gamma1':gamma1_k, 'beta1':beta1_k})
            
        else:
            if psf.aureole_model == "power":
                n_fit = fits[0]
                psf_fit.update({'n':n_fit})

            elif psf.aureole_model == "multi-power":
                n_s_fit = np.concatenate([fits[:N_n], [n_c]])
                theta_s_fit = np.concatenate([[theta_0],
                                              np.atleast_1d(10**fits[N_n:N_n+N_theta]),
                                              [theta_c]])
                psf_fit.update({'n_s':n_s_fit, 'theta_s':theta_s_fit})
            
        comp2 = psf_fit.f_aureole1D(r)

        y_fit = (1-frac) * comp1 + frac * comp2

        plt.semilogy(r, y_fit, color=c, lw=2.5, ls=ls, alpha=0.8, label=lab+' comb.', zorder=4)

        if lab=="med":
            plt.semilogy(r, (1-frac) * comp1,
                         color="orange", lw=2, ls="--", alpha=0.7, label="med core",zorder=4)
            plt.semilogy(r, frac * comp2,
                         color="seagreen", lw=2, ls="--", alpha=0.7, label="med aureole",zorder=4)

#         if Amp_max is not None:
#             std_fit = 10**fits[-1]
#             contrast = Amp_max/(std_fit)
#             y_min_contrast = y_fit.max()/contrast
            
#             plt.axhline(y_min_contrast, color="k", ls="-.", alpha=0.5)
#             plt.axhline(y_min_contrast*2, color="k", ls=":", alpha=0.5)
#             plt.text(1, y_fit.max()/contrast*1.2, '1 $\sigma$', fontsize=10)
#             plt.text(1, y_fit.max()/contrast*2.5, '2 $\sigma$', fontsize=10)
            
#             r_max = r[np.argmin(abs(y_fit-y_fit.max()/contrast))]
#             plt.xlim(0.9, 5*r_max)  
                
    # Draw boundaries etc.
    if r_core is not None:
        
        if figsize is not None:
            plt.axvspan(np.atleast_1d(r_core).max(), theta_c/pixel_scale,
                        color='steelblue', alpha=0.15, zorder=1)
            plt.axvspan(np.atleast_1d(r_core).min(), np.atleast_1d(r_core).max(),
                        color='seagreen', alpha=0.15, zorder=1)
            plt.axvspan(plt.gca().get_xlim()[0], np.atleast_1d(r_core).min(),
                        color='gray', alpha=0.15, zorder=1)
            
        if psf.aureole_model != "moffat":
            for t in psf_fit.theta_s_pix:
                plt.axvline(t, lw=2, ls='--', color='k', alpha=0.5)        
        
    plt.legend(loc=1, fontsize=12)    
    plt.xlabel(r"$\rm r\,[pix]$",fontsize=18)
    plt.ylabel(r"$\rm Intensity$",fontsize=18)
    plt.title("Recovered PSF from Fitting",fontsize=18)
    plt.ylim(3e-9, 0.5)    
    plt.xscale("log")
    plt.tight_layout()
    
    if save:
        plt.savefig("%s/Fit_PSF1D%s.png"%(save_dir, suffix),dpi=150)
        plt.show()
        plt.close()
    
def plot_bright_star_profile(tab_target, table_res_Rnorm, res_thumb,
                             bkg_sky=460, std_sky=2, pixel_scale=2.5, ZP=27.1,
                             mag_name='MAG_AUTO_corr', figsize=(8,6)):
    
    from .utils import Intensity2SB, cal_profile_1d
    
    r = np.logspace(0.03,3,100)
    
    z_mean_s, z_med_s = table_res_Rnorm['Imean'], table_res_Rnorm['Imed'] 
    z_std_s, sky_mean_s = table_res_Rnorm['Istd'], table_res_Rnorm['Isky']
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    # adaptive colormap
    cmap = plt.cm.plasma(np.linspace(0.01, 0.99, len(res_thumb)+np.sum(tab_target[mag_name]<10)+1))
    ax.set_prop_cycle(plt.cycler('color', cmap))

    for i, (num, sky_m, mag) in enumerate(zip(list(res_thumb.keys())[::-1],
                                              sky_mean_s[::-1],tab_target[mag_name][::-1])):
        
        if num in tab_target["NUMBER"]:
            alpha = min(0.05*(18-mag), 0.8) 
            errorbar = True if mag<10 else False
            ms = max((15-mag), 0)
            lw = max((12-mag), 1.5)
        else:
            alpha = 0.5; errorbar=False
            ms, lw = 3, 3
            
        img, ma, cen = res_thumb[num]['image'], res_thumb[num]['mask'], res_thumb[num]['center']
        
        r_rbin, I_rbin, _ = cal_profile_1d(img, cen=cen, mask=ma, dr=1.25,
                                           ZP=ZP, sky_mean=bkg_sky, sky_std=std_sky,
                                           xunit="pix", yunit="SB", errorbar=errorbar,
                                           core_undersample=False, 
                                           color=None, lw=lw, markersize=ms, alpha=alpha)
        if i==0:
            plt.text(3, I_rbin[np.argmin(abs(r_rbin-10))], '%s mag'%np.around(mag, 1))
    
    plt.text(14, I_rbin[np.argmin(abs(r_rbin-10))], '%s mag'%np.around(mag, 1))

    I_sky = Intensity2SB(std_sky, 0, ZP=ZP, pixel_scale=pixel_scale)
    plt.axhline(I_sky, color="k", ls="-.", alpha=0.5)
    plt.text(1.1, I_sky+0.5, '1 $\sigma$', fontsize=10)
    plt.ylim(30.5,16.5)
    plt.xlim(1.,3e2)
    plt.xscale('log')
    plt.show()
