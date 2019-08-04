import os
import time
import numpy as np
import multiprocess as mp
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from scipy import stats
from skimage.transform import resize
from utils import *


def generate_mock_image(image_shape, n_star=200, method="real", 
                        gamma=1.5, alpha=3, n=2.5, theta=4.,
                        pix_scale = 2.5, mu=886, sigma=5, 
                        display=True, savename="mock.png", parallel=False, n_cpu=4, seed=512):
    
    if method=="real":
        image_shape = (201, 201)
        cen = ((image_shape[1]-1)/2., (image_shape[0]-1)/2.)
        yy, xx = np.mgrid[:image_shape[0], :image_shape[1]]

        from photutils.datasets import make_noise_image
        noise =  make_noise_image(image_shape, type='gaussian', mean=mu, stddev=sigma, random_state=seed)
        Add_Poisson_Noise = False
        if Add_Poisson_Noise: 
            poisson_noise = np.sum([L*make_noise_image(image_shape, type='poisson', mean=1e-3) 
                                  for L in np.random.normal(loc=1,scale=0.3,size=20)], axis=0)
            noise += poisson_noise
            shot = poisson_noise[poisson_noise!=0]
            poisson_frac = shot.ravel().size/noise.size
            poisson_med = np.median(shot)
            poisson_std =  np.std(shot)
            print("outlier fraction: %.3f med: %.2g std: %.2g"%(poisson_frac, poisson_med, poisson_std))

        np.random.seed(seed)
        star_pos = image_shape[0]*np.random.random(size=(n_star,2))+1

        logAmp_dist = stats.lognorm(s=0.5, loc=1.66, scale=1.15)
        Amps = 10**logAmp_dist.rvs(size=n_star, random_state=seed)
        Amp_cutoff = 10**5.5
        while (Amps>Amp_cutoff).sum() > 0:
            Amps_resample = 10**logAmp_dist.rvs(size=(Amps>Amp_cutoff).sum(), random_state=seed+1)
            Amps[Amps>Amp_cutoff] = Amps_resample

        contrast = 0.5 * sigma*(1+(theta/gamma)**2)**alpha
        bright = Amps > contrast

        moffat2d_s = np.array([models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha) 
                      for (amp,(x0,y0)) in zip(Amps, star_pos)])
        moffat1d_s = np.array([models.Moffat1D(amplitude=amp, x_0=0, gamma=gamma, alpha=alpha) 
                      for (amp,(x0,y0)) in zip(Amps, star_pos)])


        image0 = np.sum([m2d(xx,yy) for m2d in moffat2d_s] ,axis=0)
        image = image0.copy()

        start=time.time()

        if parallel:
            with mp.Pool(processes=n_cpu) as pool:
                results = [pool.apply_async(power2d, (xx, yy, n, theta, m1d(theta), (x0,y0)))
                           for ((x0,y0),m1d) in zip(star_pos[bright], moffat1d_s[bright])]
                image += np.sum([res.get() for res in results], axis=0)        
        else:
            for ((x0,y0),m1d) in zip(star_pos[bright], moffat1d_s[bright]):
                image += power2d(xx, yy, n, cen=(x0,y0),
                                      theta0=theta, I_theta0=m1d(theta))

        image += noise  

        end=time.time()
        print("Time to generate the image: %.3gs"%(end-start))

        if display:
            plt.figure(figsize=(7,6))
            ax = plt.subplot(111)
            im = plt.imshow(image, cmap='gray', aspect='equal', norm=norm1, vmin=mu, vmax=mu+10*sigma, origin='lower')

            from photutils import CircularAperture, CircularAnnulus
            aper = CircularAperture(star_pos, r=theta)
            aper.plot(color='gold',ls="-",lw=2,alpha=0.5)
            plt.xlabel("X")
            plt.ylabel("Y")
            colorbar(im)
            plt.tight_layout()
            plt.savefig(savename, dpi=100)
            plt.close()
        
generate_mock_image((801,801), n_star=400, savename="./tmp/mock1")
generate_mock_image((801,801), n_star=400, parallel=True, savename="./tmp/mock2")