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
SHOW_MOCK, SHOW_MASK = False, False
MOCK_CONV, RUN_FITTING = True, True
version = 2.4
n_cpu = os.cpu_count()

tshape = (201, 201)
psf_size = tshape[0]//2+1
n_star = 50
mu, sigma = 1e-2, 1e-2

Amps = np.random.lognormal(mean=np.log(20),sigma=1.5, size=n_star)
gamma, alpha = 3, 3
n, theta = 2, 5

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
    print("outlier fraction: %.3f"%(poisson_frac))

    
# Generate stars w/ known amplitude and position
np.random.seed(42)
star_pos = tshape[0]*np.random.random(size=(n_star,2))

yy, xx = np.mgrid[:tshape[0], :tshape[1]]
dist_maps = [np.sqrt((xx-x0)**2+(yy-y0)**2) for (x0,y0) in star_pos]


# Generate Mock Image
if not MOCK_CONV: 
    start=time.time()
    # 2D & 1D Moffat
    np.random.seed(512)
    moffat2d_s = [models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha) 
                  for (amp,(x0,y0)) in zip(Amps, star_pos)]
    moffat1d_s = [models.Moffat1D(amplitude=amp, x_0=0, gamma=gamma, alpha=alpha) 
                  for (amp,(x0,y0)) in zip(Amps, star_pos)]
    ###mask  = ~np.logical_and.reduce([d_maps>5*gamma for d_maps in dist_maps])



    image = noise + np.sum([m2d(xx,yy) for m2d in moffat2d_s] ,axis=0) \
                  + np.sum([power2d(xx, yy, n, cen=(x0,y0),
                                    theta=theta*gamma, I_theta=m1d(theta*gamma)) 
                              for ((x0,y0),m1d) in zip(star_pos, moffat1d_s)], axis=0)
    end=time.time()
    print("Time to generate the image: %.3gs"%(end-start))
    
else:
    start=time.time()
    # Build PSF Model
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
   
    Mof_mod_2d = models.Moffat2D(amplitude=1, x_0=cen_psf[0], y_0=cen_psf[1], gamma=gamma, alpha=alpha) 
    Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    psf_model = Mof_mod_2d(xx_psf, yy_psf) + \
                power2d(xx_psf, yy_psf, n, cen=cen_psf, theta=theta*gamma, I_theta=Mof_mod_1d(theta*gamma)) 
   
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
    
    
print("Building Mock Image...Done!")

if SHOW_MOCK:
    plt.figure(figsize=(7,6))
    ax = plt.subplot(111)
    plt.imshow(image, cmap='gray', aspect='equal', norm=norm1, vmin=0, vmax=100, interpolation='None',origin='lower') # saturate at 100 to emphasize background
    plt.colorbar(fraction=0.045, pad=0.05) 

    from photutils import CircularAperture, CircularAnnulus
    aper1 = CircularAperture(star_pos, r=3*gamma)
    aper2 = CircularAperture(star_pos, r=5*gamma)
    aper1.plot(color='gold',ls="--",lw=2,alpha=0.5)
    aper2.plot(color='gold',lw=1,label="",alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig("Mock.png"+version,dpi=150)
    plt.close()

# Source Extraction and Masking
from photutils import detect_sources
back, back_rms = background_sub_SE(image, b_size=25)
threshold = back + (2.5 * back_rms)
segm0 = detect_sources(image, threshold, npixels=5)

from skimage import morphology
segmap = segm0.data.copy()
for i in range(5):
    segmap = morphology.dilation(segmap)
segmap2 = segm0.data.copy()
segmap2[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
mask_deep = (segmap!=0)

if SHOW_MASK:
    
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    im1 = ax1.imshow(image, origin='lower', cmap='gray', norm=norm1, vmin=0., vmax=100, aspect='auto')
    colorbar(im1)

    ax2.imshow(segmap2, origin="lower", cmap="gnuplot2")

    image2 = image.copy()
    image2[mask_deep] = 0
    im3 = ax3.imshow(image2, cmap='gnuplot2', norm=norm2, vmin=1e-2, vmax=0.07, origin='lower', aspect='auto') 
    colorbar(im3)

    plt.tight_layout()
    plt.savefig("Seg+Mask.png"+version,dpi=150)

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

"""Fix Moffat component while fitting"""
if RUN_FITTING:
    mask_fit = mask_deep.copy()
    X = np.array([xx,yy])
    Y = image[~mask_fit].ravel()
    xx, yy = X[0], X[1]
    
    if not MOCK_CONV: 
        mof_comp = np.sum([m(xx,yy) for m in moffat2d_s] ,axis=0)
    else:
        Mof_mod_2d = models.Moffat2D(amplitude=1, x_0=cen_psf[0], y_0=cen_psf[1], gamma=gamma, alpha=alpha) 
        Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)

    
def loglike_sum(v):
    """Log Likelihood for Outer PSF in summation"""
    n, theta, log10_mu, log10_sigma = v
    mu, sigma = (10**log10_mu, 10**log10_sigma)
    pow_comp = np.sum([power2d(xx, yy, n, cen=(x0,y0), 
                               theta=theta*gamma, I_theta=m1d(theta*gamma))
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
                power2d(xx_psf, yy_psf, n=n, cen=cen_psf, theta=theta*gamma, I_theta=Mof_mod_1d(theta*gamma)) 
    
    psf_kernel = Kernel(psf_model)
    psf_kernel.normalize(mode="peak")
    
    image_conv = convolve_fft(image_spot, psf_kernel)
    
    ypred = convolve_fft(image_spot, psf_kernel)[~mask_fit].ravel()
    residsq = (ypred + mu - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike

###-----------------------------------------------------###

# Run in multiprocess
import multiprocess as mp

def Run_Nested_Fitting():
    if not MOCK_CONV: 
        loglike = loglike_sum
    else:
        loglike = loglike_conv
        
    with mp.Pool(n_cpu-1) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu-1))
        pool.size = n_cpu-1
        
        dlogz = 1e-3 * (200 - 1) + 0.01
        
        start = time.time()
        pdsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, 4,
                                                  pool=pool, use_pool={'update_bound': False})
        pdsampler.run_nested(nlive_init=200, nlive_batch=100, maxbatch=3,
                              dlogz_init=dlogz, wt_kwargs={'pfrac': 0.8})
        end = time.time()

    print("Total time elapsed: %.3gs"%(end-start))

    pdres = pdsampler.results

    # Plot Result
    fig, axes = dyplot.cornerplot(pdres, truths=truths, show_titles=True, 
                                  color="royalblue", truth_color="indianred",
                                  title_kwargs={'fontsize':24, 'y': 1.04}, labels=labels,
                                  label_kwargs={'fontsize':22},
                                  fig=plt.subplots(4, 4, figsize=(18, 16)))
    plt.savefig("Result%s.png"+version,dpi=150)
    plt.close()

if RUN_FITTING:
    Run_Nested_Fitting()