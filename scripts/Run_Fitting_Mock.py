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
dir_name = os.path.join('tmp', id_generator())
check_save_path(dir_name)
# dir_name = 'mp_mock'
print_progress = True
n_spline = 3
n_thread = None
n_cpu = 4


############################################Z
# Setting
############################################

# Meta-parameter
n_star = 400
wid_strip, n_strip = 8, 32
mu = 884
sigma = 1e-1

# Image Parameter
image_size = 801
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
np.random.seed(62)
star_pos = (image_size-2) * np.random.random(size=(n_star,2)) + 1

# Read SE measurement based on APASS
SE_cat_full = Table.read("./SE_APASS/coadd_SloanR_NGC_5907.cat", format="ascii.sextractor").to_pandas()
Flux_Auto_SE = SE_cat_full[SE_cat_full['FLAGS']<8]["FLUX_AUTO"]

# Star flux sampling from SE catalog
np.random.seed(888)
Flux = Flux_Auto_SE.sample(n=n_star).values

Flux[Flux>5e5] *= 15

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
mask.make_mask_map_dual(r_core_s, sn_thre=2.5, n_dilation=5,
                        draw=True, save=True, dir_name=dir_name)

# Strip + Cross mask
mask.make_mask_strip(wid_strip, n_strip, dist_strip=320, clean=True,
                     draw=True, save=True, dir_name=dir_name)
stars = mask.stars_new


# Fitting Preparation
############################################ 
mask_fit = mask.mask_comb
mask_deep = mask.mask_deep

X = np.array([psf.xx,psf.yy])
Y = image[~mask_fit].copy().ravel()

# Estimated mu and sigma used as prior
Y_clip = sigma_clip(image[~mask_deep].ravel(), sigma=3, maxiters=10)
mu_patch, std_patch = np.mean(Y_clip), np.std(Y_clip)
print("\nEstimate of Background: (%.3f, %.3f)"%(mu_patch, std_patch))

# Choose fitting parameterization
prior_tf = set_prior(3.3, mu_patch, std_patch,
                     theta_in=90, theta_out=500, n_spline=n_spline)

loglike = set_likelihood(Y, mask_fit, psf, stars, image_base,
                         psf_range=[320, 640], n_spline=n_spline, norm='flux',
                         brightest_only=True, parallel=False, draw_real=False)

if n_spline==2:
    labels = [r'$n0$', r'$n1$', r'$\theta_1$', r'$\mu$', r'$\log\,\sigma$']

elif n_spline==3:
    labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$',
              r'$\mu$', r'$\log\,\sigma$']

if leg2d:
    labels = np.insert(labels, -2, [r'$\log\,A_{01}$', r'$\log\,A_{10}$'])

if fit_frac:
    labels = np.insert(labels, -2, [r'$f_{pow}$'])

    
ndim = len(labels)
print("Labels: ", labels)


############################################
# Run & Plot
############################################
ds = DynamicNestedSampler(loglike, prior_tf, ndim,
                          n_cpu=n_cpu, n_thread=n_thread)

ds.run_fitting(nlive_init=min(20*(ndim-1),100), nlive_batch=25, maxbatch=2,
               print_progress=print_progress)

ds.save_result(filename='Mock-fit_best_%s.res'%(n_spline+'p'),
               dir_name=dir_name)

ds.cornerplot(labels=labels, save=save, dir_name=dir_name, figsize=(22, 20))

ds.plot_fit_PSF1D(psf0, labels, n_bootstrap=500,
                  Amp_max=Amp_m, r_core=r_core_s, save=save, dir_name=dir_name)

draw2D_fit_vs_truth_PSF_mpow(ds.results, psf0, stars, labels, image,
                             save=save, dir_name=dir_name)

