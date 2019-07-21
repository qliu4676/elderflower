import os
import time
import numpy as np
from astropy.modeling import models, fitting
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import stats
from photutils.datasets import make_noise_image
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())
norm3 = ImageNormalize(stretch=SqrtStretch())
from utils import *

# Params of Run
SHOW_MOCK, SHOW_MASK = True, True
MOCK_CONV, RUN_FITTING = True, True
LIN_BKG = True
version = '_linbkg'
n_cpu = 11 #os.cpu_count()

tshape = (1001, 1001)
psf_size = tshape[0]
gamma, alpha = 3, 3
n, theta = 2, 5

np.random.seed(512)
n_star = 50

if LIN_BKG:
    mu, sigma = 885, 5
    Amps = 10**stats.lognorm.rvs(s=0.58, loc=1.66, scale=1.15, size=n_star)
    cut = Amps > 1e7
    Amps[cut] = 10**stats.lognorm.rvs(s=0.5, loc=1., scale=1.15, size=len(Amps[cut]))
else:
    mu, sigma = 1e-2, 1e-2
    Amps = np.random.lognormal(mean=np.log(20),sigma=1.5, size=n_star)

###-----------------------------------------------------###

cen = ((tshape[0]-1)/2., (tshape[0]-1)/2.)

# Generate noise background
np.random.seed(39)
noise =  make_noise_image(tshape, type='gaussian', mean=mu, stddev=sigma)

Add_Poisson_Noise = False
if Add_Poisson_Noise: 
    poisson_noise = np.sum([L*make_noise_image(tshape, type='poisson', mean=1e-3) 
                      for L in np.random.normal(loc=1e-2,scale=3e-3,size=20)], axis=0)
    noise += poisson_noise
    shot = poisson_noise[poisson_noise!=0]
    poisson_frac = shot.ravel().size/noise.size
    poisson_med = np.median(shot)
    poisson_std =  np.std(shot)
    print("outlier fraction: %.3f med: %.2g std: %.2g"%(poisson_frac, poisson_med, poisson_std))

# Generate stars w/ known amplitude and position
np.random.seed(42)
star_pos = tshape[0]*np.random.random(size=(n_star,2))

yy, xx = np.mgrid[:tshape[0], :tshape[1]]
dist_maps = [np.sqrt((xx-x0)**2+(yy-y0)**2) for (x0,y0) in star_pos]


# Generate Mock Image
if MOCK_CONV: 
    start=time.time()
    # Build PSF Model
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
   
    Mof_mod_2d = models.Moffat2D(amplitude=1, x_0=cen_psf[0], y_0=cen_psf[1], gamma=gamma, alpha=alpha) 
    Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    psf_model = Mof_mod_2d(xx_psf, yy_psf) + \
                power2d(xx_psf, yy_psf, n=n, cen=cen_psf, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)) 
   
    from astropy.convolution import Kernel,convolve_fft
    psf_kernel = Kernel(psf_model)
    psf_kernel.normalize(mode="peak")
    
    # Impose Spots with R=2 pixel 
    image_spot = np.zeros(tshape)
    spot2d_s = [models.Disk2D(amplitude=amp, x_0=x0, y_0=y0, R_0=2) 
                  for (amp,(x0,y0)) in zip(Amps, star_pos)]

    for s2d in spot2d_s:
        spot = s2d(xx,yy)
    #     spot /= 1.0 * len(spot[np.nonzero(spot)])
        image_spot += spot

    # Convolve Image with PSF
    image_conv = convolve_fft(image_spot, psf_kernel)
    image_conv += noise
    image = image_conv.copy()
    end=time.time()
    print("Time to generate the image: %.3gs"%(end-start))
    
else:
    start=time.time()
    # 2D & 1D Moffat
    moffat2d_s = [models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha) 
                  for (amp,(x0,y0)) in zip(Amps, star_pos)]
    moffat1d_s = [models.Moffat1D(amplitude=amp, x_0=0, gamma=gamma, alpha=alpha) 
                  for (amp,(x0,y0)) in zip(Amps, star_pos)]
    ###mask  = ~np.logical_and.reduce([d_maps>5*gamma for d_maps in dist_maps])

    image = noise + np.sum([m2d(xx,yy) for m2d in moffat2d_s] ,axis=0) \
                  + np.sum([power2d(xx, yy, n, cen=(x0,y0),
                                    theta0=theta*gamma, I_theta0=m1d(theta*gamma)) 
                              for ((x0,y0),m1d) in zip(star_pos, moffat1d_s)], axis=0)
    end=time.time()
    print("Time to generate the image: %.3gs"%(end-start))    

if SHOW_MOCK:
    plt.figure(figsize=(7,6))
    ax = plt.subplot(111)
    plt.imshow(image, cmap='gray', aspect='equal', norm=norm1, vmin=mu, vmax=mu+100, origin='lower')
    plt.colorbar(fraction=0.045, pad=0.05) 

    from photutils import CircularAperture, CircularAnnulus
    aper = CircularAperture(star_pos, r=5*gamma)
    aper.plot(color='gold',lw=1,label="",alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig("Mock%s.png"%version,dpi=150)
    plt.close()

print("Build Mock Image...Done!")
    
# Source Extraction and Masking
mask_deep = make_mask_map(image, sn_thre=2.5, b_size=25, n_dilation = 3)

if SHOW_MASK:
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    im1 = ax1.imshow(image, origin='lower', cmap='gray', norm=norm1, vmin=mu, vmax=mu+100, aspect='auto')
    colorbar(im1)

    ax2.imshow(segmap2, origin="lower", cmap="gnuplot2")

    image2 = image.copy()
    image2[mask_deep] = 0
    im3 = ax3.imshow(image2, cmap='gnuplot2', norm=norm2, vmin=mu, vmax=mu+10*sigma, origin='lower', aspect='auto') 
    colorbar(im3)

    plt.tight_layout()
    plt.savefig("Seg+Mask%s.png"%version,dpi=150)

###-----------------------------------------------------------------###

# Plotting Params
import dynesty
from dynesty import plotting as dyplot

from matplotlib import rcParams
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
rcParams.update({'font.size': 14})

# Build Prior Transformation and Likelihood Model
if LIN_BKG:
    truths = n, theta, mu, sigma
    labels = [r'$n$', r'$\theta$', r'$\mu$', r'$\sigma$']
else:
    truths = n, theta, np.log10(mu), np.log10(sigma)
    labels = [r'$n$', r'$\theta$', r'$\log\,\,\mu$', r'$\log\,\,\sigma$']
    

def prior_transform(u):
    """Priors for the Params of Outer PSF"""
    v = u.copy()
    v[0] = u[0] * 4 + 1  # n : 1-5
    v[1] = u[1] * 7. + 3  # theta : 3-10
    v[2] = u[2] * 2 - 3  # mu : 1e-3 - 1e-1
    v[3] = u[3] * 2 - 3  # sigma : 1e-3 - 1e-1
    return v

def prior_transform_linbkg(u):
    v = u.copy()
    v[0] = u[0] * 4 + 1  # n : 1-5
    v[1] = u[1] * 7. + 3  # theta : 3-10
    v[2] = u[2] * 15 + 875  # mu : 875 - 890
    v[3] = u[3] * 9 + 1  # sigma : 1 - 10
    return v

"""Fix Moffat component while fitting"""
if RUN_FITTING:
    mask_fit = mask_deep.copy()
    X = np.array([xx,yy])
    Y = image[~mask_fit].ravel()
    xx, yy = X[0], X[1]
    
    if MOCK_CONV is False:
        mof_comp = np.sum([m(xx,yy) for m in moffat2d_s] ,axis=0)
    else:
        Mof_mod_2d = models.Moffat2D(amplitude=1, x_0=cen_psf[0], y_0=cen_psf[1], gamma=gamma, alpha=alpha) 
        Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)

    
def loglike_sum(v):
    """Log Likelihood for Outer PSF in summation"""
    n, theta, log10_mu, log10_sigma = v
    mu, sigma = (10**log10_mu, 10**log10_sigma)
    pow_comp = np.sum([power2d(xx, yy, n=n, cen=(x0,y0), 
                               theta0=theta*gamma, I_theta0=m1d(theta*gamma))
                       for ((x0,y0),m1d) in zip(star_pos, moffat1d_s)], axis=0)
    ypred = (mof_comp + pow_comp)[~mask_fit].ravel()
    residsq = (ypred  + mu - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike

def loglike_conv(v):
    """Log Likelihood for Outer PSF in FFT"""
    n, theta, log10_mu, log10_sigma = v
    mu, sigma = (10**log10_mu, 10**log10_sigma)
    
    psf_model = Mof_mod_2d(xx_psf, yy_psf) + \
                power2d(xx_psf, yy_psf, n=n, cen=cen_psf, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)) 
    
    psf_kernel = Kernel(psf_model)
    psf_kernel.normalize(mode="peak")
    
    image_conv = convolve_fft(image_spot, psf_kernel)
    
    ypred = convolve_fft(image_spot, psf_kernel)[~mask_fit].ravel()
    residsq = (ypred + mu - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike

def loglike_conv_linbkg(v):
    n, theta, mu, sigma = v
    
    psf_model = Mof_mod_2d(xx_psf, yy_psf) + \
                power2d(xx_psf, yy_psf, n=n, cen=cen_psf, theta0=theta*gamma, I_theta0=Mof_mod_1d(theta*gamma)) 
    
    psf_kernel = Kernel(psf_model)
    psf_kernel.normalize(mode="peak")
    
    image_conv = convolve_fft(image_spot, psf_kernel)
    
    ypred = image_conv[~mask_fit].ravel()
    residsq = (ypred + mu - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike

###-----------------------------------------------------###

# Run in multiprocess
import multiprocess as mp

def Run_Nested_Fitting(truths=truths, nlive_init=200, nlive_batch=200, maxbatch=4, ndim=4):
    if MOCK_CONV: 
        if LIN_BKG:
            print("Fit Background in Linear.")
            prior_transform = prior_transform_linbkg
            loglike = loglike_conv_linbkg   
        else:
            loglike = loglike_conv
    else:
        print("Fit Models in Real Space.")
        loglike = loglike_sum
        
    with mp.Pool(n_cpu-1) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu-1))
        pool.size = n_cpu-1

        dlogz = 1e-3 * (nlive_init - 1) + 0.01

        start = time.time()
            pdsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim,
                                                      pool=pool, use_pool={'update_bound': False})
            pdsampler.run_nested(nlive_init=nlive_init, nlive_batch=nlive_batch, maxbatch=maxbatch,
                                  dlogz_init=dlogz, wt_kwargs={'pfrac': 0.8})
        end = time.time()

    print("Total time elapsed: %.3gs"%(end-start))

    pdres = pdsampler.results

    # Plot Result
    fig, axes = dyplot.cornerplot(pdres, truths=truths, show_titles=True, 
                                  color="royalblue", truth_color="indianred",
                                  title_kwargs={'fontsize':24, 'y': 1.04}, labels=labels,
                                  label_kwargs={'fontsize':22},
                                  fig=plt.subplots(ndim, ndim, figsize=(18, 16)))
    plt.savefig("Result%s.png"%version,dpi=150)
    plt.close()
    
    return pdsampler

if RUN_FITTING:
    pdsampler = Run_Nested_Fitting(truths=truths)
    pdres = pdsampler.results
    
save_nested_fitting_result(pdres, filename='fit%s.res'%version)