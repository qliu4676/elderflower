import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
from astropy.stats import SigmaClip, mad_std, gaussian_fwhm_to_sigma
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())
from skimage import morphology

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def power1d(x, n, theta0, I_theta0): 
    x[x<=0] = x[x>0].min()
    a = I_theta0/(theta0)**(-n)
    y = a * np.power(x, -n)
    return y

def multi_power1d(x, n0, theta0, I_theta0, n_s, theta_s):
    ind_slice = [np.argmin(x<theta_i) for theta_i in theta_s]
    a0 = I_theta0/(theta0)**(-n0)
    y = a0 * np.power(x, -n0)
    I_theta_i = a0 * np.power(1.*theta_s[0], -n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        y_i = a_i * np.power(x, -n_i)
        y[x>theta_i] = y_i[x>theta_i]
        try:
            I_theta_i = a_i * np.power(1.*theta_s[i+1], -n_i)
        except IndexError:
            pass
    return y

def power2d(x, y, n, theta0, I_theta0, cen): 
    r = np.sqrt((x-cen[0])**2+(y-cen[1])**2)
    r[r<=0] = r[r>0].min()
    a = I_theta0/(theta0)**(-n)
    z = a * np.power(r, -n) 
    return z 

def multi_power2d(x, y, n0, theta0, I_theta0, n_s, theta_s, cen):
    r = np.sqrt((x-cen[0])**2+(y-cen[1])**2)
    r[r<=0] = r[r>0].min()
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(r, -n0) 
    I_theta_i = a0 * np.power(1.*theta_s[0], -n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(r, -n_i)
        z[r>theta_i] = z_i[r>theta_i]
        try:
            I_theta_i = a_i * np.power(1.*theta_s[i+1], -n_i)
        except IndexError:
            pass
    return z


def vmin_3mad(img):
    # lower limit of visual imshow defined by 3 mad above median
    return np.median(img)-3*mad_std(img)

def vmax_2sig(img):
    # upper limit of visual imshow defined by 2 sigma above median
    return np.median(img)+2*np.std(img)

def colorbar(mappable, pad=0.2):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    return fig.colorbar(mappable, cax=cax)


def background_sub_SE(field, mask=None, b_size=64, f_size=3, n_iter=5):
    # Subtract background using SE estimator with mask
    from photutils import Background2D, SExtractorBackground
    Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                       box_size=(b_size, b_size), filter_size=(f_size, f_size),
                       sigma_clip=SigmaClip(sigma=3., maxiters=n_iter))
    back = Bkg.background
    back_rms = Bkg.background_rms
    if mask is not None:
        back *= ~mask
    field_sub = field - back
    return back, back_rms

def display_background_sub(field, back):
    norm1 = ImageNormalize(stretch=LogStretch())
    norm2 = ImageNormalize(stretch=LogStretch())
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(17,6))
    ax1.imshow(field, origin="lower", aspect="auto", cmap="gray", vmin=vmin_3mad(field), vmax=vmax_2sig(field),norm=norm1)
    im2 = ax2.imshow(back, origin='lower', aspect="auto", cmap='gray')
    colorbar(im2)
    ax3.imshow(field - back, origin='lower', aspect="auto", cmap='gray', vmin=0., vmax=vmax_2sig(field - back),norm=norm2)
    plt.tight_layout()

def source_detection(data, sn=2, b_size=120, k_size=3, fwhm=3, 
                     morph_oper=morphology.dilation,
                     sub_background=True, mask=None):
    from astropy.convolution import Gaussian2DKernel 
    from photutils import detect_sources

    if sub_background:
        back, back_rms = background_sub_SE(data, b_size=b_size)
        threshold = back + (sn * back_rms)
    else:
        back = np.zeros_like(data)
        threshold = np.nanstd(data)
    sigma = fwhm * gaussian_fwhm_to_sigma    # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=k_size, y_size=k_size)
    kernel.normalize()
    segm_sm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel, mask=mask)
    
    segm_sm = morph_oper(segm_sm.data)
    data_ma = data.copy() - back
    data_ma[segm_sm!=0] = np.nan

    return data_ma, segm_sm

def make_mask_map(image, sn_thre=2.5, b_size=25, n_dilation = 5):    
    from photutils import detect_sources 
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=5)

    from skimage import morphology
    segmap = segm0.data.copy()
    for i in range(n_dilation):
        segmap = morphology.dilation(segmap)
    segmap2 = segm0.data.copy()
    segmap2[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap2!=0)
    
    return mask_deep, segmap2

def save_nested_fitting_result(res, filename='fit.res'):
    import dill
    with open(filename,'wb') as file:
        dill.dump(res, file)
        
def plot_fitting_vs_truth_PSF(res, true_pars, n_bootsrap=100, save=False, version=""):
    from dynesty import utils as dyfunc
    
    samples = res.samples  # samples
    weights = np.exp(res.logwt - res.logz[-1])  # normalized weights
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    # Resample weighted samples.
    samples_eq = dyfunc.resample_equal(samples, weights)
    
    from astropy.stats import bootstrap
    samples_eq_bs = bootstrap(samples_eq, bootnum=1, samples=n_bootsrap)[0]
    
    gamma, alpha, n, theta, mu, sigma = true_pars.values()
    
    Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    
    
    plt.figure(figsize=(8,6))
    r = np.logspace(0.,3,100)
    plt.semilogy(r, Mof_mod_1d(r) + power1d(r, n, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)),
                 label="Truth", color="steelblue", lw=3, zorder=2)
    for n_k, theta_k in zip(samples_eq_bs[:,0].T, samples_eq_bs[:,1].T):
        plt.semilogy(r, Mof_mod_1d(r) + power1d(r, n_k, theta0=theta_k*gamma, I_theta0=Mof_mod_1d(theta_k*gamma)),
                     color="lightblue",alpha=0.1,zorder=1)
    else:
        plt.semilogy(r, Mof_mod_1d(r) + power1d(r, mean[0], theta0=mean[1]*gamma, I_theta0=Mof_mod_1d(mean[1]*gamma)),
                     label="Fit", color="mediumblue",ls="--",lw=2,alpha=0.75,zorder=2)
    plt.semilogy(r, Mof_mod_1d(r), label="Moffat", ls=":", color="orange", lw=2, alpha=0.7,zorder=1)
    plt.semilogy(r, power1d(r, n, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)),
                 label="Power", color="g", lw=2, ls=":",alpha=0.7,zorder=1)
#     plt.axhline(mu/1e6,color="k",ls=":")
    plt.ylim(3e-9,3)
    plt.xlabel(r"$\rm r\,[pix]$",fontsize=18)
    plt.ylabel(r"$\rm Intensity$",fontsize=18)
    plt.xscale("log")
    plt.tight_layout()
    plt.legend(fontsize=12)
    if save:
        plt.savefig("./tmp/PSF%s.png"%version,dpi=150)
    plt.show()