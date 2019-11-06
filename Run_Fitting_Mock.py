import galsim
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from utils import *
from modeling import *
from plotting import *

import matplotlib
matplotlib.use('Agg')

# Option
save = True
dir_name = id_generator()
check_save_path(dir_name)
# dir_name = 'mp_mock'
print_progress = True
method = '2p'
n_thread = None
n_cpu = 3


############################################
# Setting
############################################

# Meta-parameter
n_star = 300
wid_strip, n_strip = 8, 32
mu = 884
sigma = 1e-1

# Image Parameter
image_size = 601
pixel_scale = 2.5                                # arcsec/pixel

# PSF Parameters
beta = 10                                        # moffat beta, in arcsec
fwhm = 2.28 * pixel_scale                        # moffat FWHM, in arcsec

n0 = 3.3                                         # true power index
frac = 0.1                                       # fraction of power law component
theta_0 = 5.                                     # radius at which power law is flattened, in arcsec

n_s = np.array([n0, 3, 2.4, 2., 1.4, 4])
theta_s = np.array([theta_0, 60, 120, 200, 320, 900])      # transition radius in arcsec

# Multi-power PSF
params_mpow = {"fwhm":fwhm, "beta":beta, "frac":frac, "n_s":n_s, 'theta_s':theta_s}
psf = PSF_Model(params=params_mpow, aureole_model='multi-power')

# Build grid of image for drawing
psf.make_grid(image_size, pixel_scale=pixel_scale)


############################################
# Star Distribution (position, flux)
############################################

# Generate randomn star positions
np.random.seed(626)
star_pos = (image_size-2) * np.random.random(size=(n_star,2)) + 1

# Read SE measurement based on APASS
SE_cat_full = Table.read("./SE_APASS/coadd_SloanR_NGC_5907.cat", format="ascii.sextractor").to_pandas()
Flux_Auto_SE = SE_cat_full[SE_cat_full['FLAGS']<8]["FLUX_AUTO"]

# Star flux sampling from SE catalog
np.random.seed(888)
Flux = Flux_Auto_SE.sample(n=n_star).values

Flux[Flux>5e5] *= 10

# Flux Thresholds 
F_bright = 1e5
F_verybright = 3e6

stars = Stars(star_pos, Flux, Flux_threshold=[F_bright, F_verybright], verbose=True)
stars.plot_flux_dist(label='flux', save=True, dir_name=dir_name)

############################################
# Generate mock truth and base
############################################

# Generate core and (initial) aureole PSF
psf_c = psf.generate_core()
psf_e, psf_size = psf.generate_aureole(contrast=1e6, psf_range=image_size)
star_psf = (1-frac) * psf_c + frac * psf_e
psf0 = psf.copy()

# Galsim 2D model averaged in 1D
Amp_m = psf.Flux2Amp(Flux).max()
contrast = Amp_m / sigma
psf.plot_model_galsim(psf_c, psf_e, image_size,
                      contrast=contrast, save=True, dir_name=dir_name)

# Make noise image
noise_image = make_noise_image(image_size, sigma)

# Make sky background and dim stars
image_base = make_base_image(image_size, stars, psf_base=star_psf)

# Make truth image
image = make_truth_image(psf, stars)
image = image + image_base + mu + noise_image

# Masking
mask = Mask(image, stars, image_size, mu=mu)

# Core mask
r_core_s = [36, 36]
mask.make_mask_map_dual(r_core_s, sn_thre=2.5, n_dilation=3,
                        draw=True, save=True, dir_name=dir_name)

# Strip + Cross mask
mask.make_mask_strip(wid_strip, n_strip, dist_strip=320, clean=True,
                     draw=True, save=True, dir_name=dir_name)
stars = mask.stars_new


# Fitting Preparation
############################################ 
mask_fit = mask.mask_comb

X = np.array([psf.xx,psf.yy])
Y = image[~mask_fit].copy().ravel()

# Estimated mu and sigma used as prior
Y_clip = sigma_clip(Y, sigma=3, maxiters=10)
mu_patch, std_patch = np.mean(Y_clip), np.std(Y_clip)
print("\nEstimate of Background: (%.3f, %.3f)"%(mu_patch, std_patch))


############################################
# Priors and Likelihood Models for Fitting
############################################

def prior_tf_2p(u):
    v = u.copy()
    v[0] = u[0] * 0.6 + 3             # n0 : 3-3.6
    v[1] = u[1] * (v[0]-1.2) + 1.2    # n1 : 1.2-n0
    v[2] = u[2] * 0.8 + 1.7           # log theta1 : 50-300  # in arcsec
    v[-2] = stats.truncnorm.ppf(u[-2], a=-2, b=0.1,
                                loc=mu_patch, scale=std_patch)         # mu
    v[-1] = stats.truncnorm.ppf(u[-1], a=-1, b=0.1,
                                loc=np.log10(std_patch), scale=0.5)    # log sigma 
    return v

def loglike_2p(v):
    n_s = v[:2]
    theta_s = [theta_0, 10**v[2]]
    mu, sigma = v[-2], 10**v[-1]
    
    psf.update({'n_s':n_s, 'theta_s':theta_s})
    
    image_tri = generate_mock_image(psf, stars, brightest_only=True, parallel=False, draw_real=True)
    image_tri = image_tri + image_base + mu 
    
    ypred = image_tri[~mask_fit].ravel()
    residsq = (ypred - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike


def prior_tf_3p(u):
    v = u.copy()
    v[0] = u[0] * 0.8 + 2.7                 # n0 : 2.7-3.5
    v[1] = u[1] * 0.8 + (v[0]-1)            # n1 : n0-1 - n0-0.2
    v[2] = max(u[2] * 0.8 + (v[1]-1) ,1)    # n2 : [1,n1-1] - n1-0.2
    v[3] = u[3] * 0.8 + 1.7                 # log theta1 : 50-300  # in arcsec
    v[4] = u[4] * (2.7-2*v[3]) + 2*v[3]     # log theta2 : 2 * theta1 - 500  # in arcsec
    v[-2] = stats.truncnorm.ppf(u[-2], a=-2, b=0.1,
                                loc=mu_patch, scale=std_patch)         # mu
    v[-1] = stats.truncnorm.ppf(u[-1], a=-1, b=0.1,
                                loc=np.log10(std_patch), scale=0.5)    # log sigma 
    return v

def loglike_3p(v):
    n_s = v[:3]
    theta_s = [theta_0, 10**v[3], 10**v[4]]
    mu, sigma = v[-2], 10**v[-1]
    
    psf.update({'n_s':n_s, 'theta_s':theta_s})
    
    image_tri = generate_mock_image(psf, stars, brightest_only=True, parallel=False, draw_real=True)
    image_tri = image_tri + image_base + mu 
    
    ypred = image_tri[~mask_fit].ravel()
    residsq = (ypred - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike


# Choose fitting parameterization
if method == '2p':
    labels = [r'$n0$', r'$n1$', r'$\theta_1$', r'$\mu$', r'$\log\,\sigma$']
    loglike, prior_tf = loglike_2p, prior_tf_2p
    
elif method == '3p':
    labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$', r'$\mu$', r'$\log\,\sigma$']
    loglike, prior_tf = loglike_3p, prior_tf_3p

ndim = len(labels)
print("Labels: ", labels)


############################################
# Run & Plot
############################################
ds = DynamicNestedSampler(loglike, prior_tf, ndim,
                          n_cpu=n_cpu, n_thread=n_thread)

ds.run_fitting(nlive_init=min(20*ndim,100), nlive_batch=25, maxbatch=2,
               print_progress=print_progress)

ds.save_result(filename='Mock-fit_best_%s.res'%method,
               dir_name=dir_name)

ds.cornerplot(labels=labels, save=save, dir_name=dir_name)

draw2D_fit_vs_truth_PSF_mpow(ds.ressults, psf0, stars, labels,
                             save=save, dir_name=dir_name)

plot1D_fit_vs_truth_PSF_mpow(ds.ressults, psf0, labels,
                             n_bootstrap=800,
                             Amp_max=Amp_m, r_core=r_core_s,
                             save=save, dir_name=dir_name)