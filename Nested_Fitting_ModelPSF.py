import numpy as np
from astropy.modeling import models, fitting
from astropy.table import Table
from matplotlib import rcParams
rcParams['font.size'] = 12
import matplotlib.pyplot as plt
from scipy import stats
from photutils.datasets import (make_random_gaussians_table,
                                make_noise_image,
                                make_gaussian_sources_image)
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())
norm3 = ImageNormalize(stretch=SqrtStretch())
from utils import *

###-----------------------------------------------------###

# Generate noise background
np.random.seed(39)
tshape = (201, 201)
cen = ((tshape[0]-1)/2., (tshape[0]-1)/2.)
mu, sigma = 1e-2, 1e-2
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
n_star = 50
star_pos = tshape[0]*np.random.random(size=(n_star,2))

yy, xx = np.mgrid[:tshape[0], :tshape[1]]
dist_maps = [np.sqrt((xx-x0)**2+(yy-y0)**2) for (x0,y0) in star_pos]

# 2D & 1D Moffat
np.random.seed(512)
Amps = np.random.lognormal(mean=np.log(20),sigma=1.5, size=n_star)
gamma, alpha = 3, 3
n, theta = 2, 5

moffat2d_s = [models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha) 
              for (amp,(x0,y0)) in zip(Amps, star_pos)]
moffat1d_s = [models.Moffat1D(amplitude=amp, x_0=0, gamma=gamma, alpha=alpha) 
              for (amp,(x0,y0)) in zip(Amps, star_pos)]
mask  = ~np.logical_and.reduce([d_maps>3*gamma for d_maps in dist_maps])

# Generate Mock Image
image = noise + np.sum([m2d(xx,yy) for m2d in moffat2d_s] ,axis=0) \
              + np.sum([power2d(xx, yy, n, cen=(x0,y0),
                                theta=theta*gamma, I_theta=m1d(theta*gamma)) 
                          for ((x0,y0),m1d) in zip(star_pos, moffat1d_s)], axis=0)

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
truths

def prior_transform(u):
    v = u.copy()
    v[0] = u[0] * 4 + 1  # n : 1-5
    v[1] = u[1] * 7. + 3  # theta : 3-10
    v[2] = u[2] * 2 - 3  # mu : 1e-3 - 1e-1
    v[3] = u[3] * 2 - 3  # sigma : 1e-3 - 1e-1
    return v

# Fix Moffat component while fitting
X = np.array([xx,yy])
Y = image[~mask].ravel()
xx, yy = X[0], X[1]
mof_comp = np.sum([m(xx,yy) for m in moffat2d_s] ,axis=0)
    
def loglike(v):
    n, theta, log10_mu, log10_sigma = v
    mu, sigma = (10**log10_mu, 10**log10_sigma)
    pow_comp = np.sum([power2d(xx, yy, n, cen=(x0,y0), 
                               theta=theta*gamma, I_theta=m1d(theta*gamma))
                       for ((x0,y0),m1d) in zip(star_pos, moffat1d_s)], axis=0)
    ypred = (mof_comp + pow_comp + mu)[~mask].ravel()
    residsq = (ypred - Y)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))
    
    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike

###-----------------------------------------------------###

# Run in multiprocessing
import multiprocess as mp
import time
import os

n_cpu = os.cpu_count()
mypool = mp.Pool(n_cpu-1)
mypool.size = n_cpu-1

dlogz = 1e-3 * (200 - 1) + 0.01
start = time.time()
pdsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, 4,
                                          pool=mypool, use_pool={'update_bound': False})
pdsampler.run_nested(nlive_init=200, nlive_batch=200, maxbatch=4,
                      dlogz_init=dlogz, wt_kwargs={'pfrac': 0.8})
end = time.time()

print("%.3gs"%(end-start))

pdres = pdsampler.results

# Plot Result
fig, axes = dyplot.cornerplot(pdres, truths=truths, show_titles=True, 
                              color="royalblue", truth_color="indianred",
                              title_kwargs={'fontsize':24, 'y': 1.04}, labels=labels,
                              label_kwargs={'fontsize':22},
                              fig=plt.subplots(4, 4, figsize=(18, 16)))
plt.savefig("Result.png",dpi=150)
plt.close()