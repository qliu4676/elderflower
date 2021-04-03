import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.integrate import quad
from scipy.spatial import distance
from scipy.special import gamma as Gamma

from astropy import units as u
from astropy.io import fits, ascii
from astropy.modeling import models
from astropy.utils import lazyproperty

import galsim
from galsim import GalSimBoundsError

import warnings
from copy import deepcopy
from numpy.polynomial.legendre import leggrid2d
from itertools import combinations
from functools import partial, lru_cache

try: 
    from .parallel import parallel_compute
    parallel_enabled = True
    
except ImportError:
    import warnings
    warnings.warn("One of Joblib, psutil, multiprocessing, mpi4py is not installed. Parallelization is not enabled.")
    parallel_enabled = False

try:
    from numba import njit
    
except ImportError:
    def njit(*args, **kwargs):
        def dummy_decorator(func, *args, **kwargs):
            return func 
        return dummy_decorator
    
from .utils import fwhm_to_gamma, gamma_to_fwhm
from .utils import Intensity2SB, SB2Intensity
from .utils import round_good_fft, calculate_psf_size
from .image import DF_pixel_scale

############################################
# Functions for making PSF models
############################################

class PSF_Model:
    """ A PSF Model object """
    
    def __init__(self, params=None,
                 core_model='moffat',
                 aureole_model='power'):
        """
        Parameters
        ----------
        params : a dictionary containing keywords of PSF parameter
        core_model : model of PSF core (moffat)
        aureole_model : model of aureole ("moffat, "power" or "multi-power")    
        
        """
        self.core_model = core_model
        self.aureole_model = aureole_model
        
        self.cutoff = True   # with cutoff by default
        
        # Build attribute for parameters from dictionary keys 
        for key, val in params.items():
            if type(val) is list:
                params[key] = val = np.array(val)
            exec('self.' + key + ' = val')
            
        self.params = params
        
        if hasattr(self, 'fwhm'):
            self.gamma = fwhm_to_gamma(self.fwhm, self.beta)
            self.params['gamma'] = self.gamma
            
        if hasattr(self, 'gamma'):
            self.fwhm  = gamma_to_fwhm(self.gamma, self.beta)
            self.params['fwhm'] = self.fwhm
            
        self.gsparams = galsim.GSParams(folding_threshold=1e-10)
        
        if aureole_model == "power":
            self.n0 = params['n0']
            self.theta_0 = self.theta_0 = params['theta_0']

        elif aureole_model == "multi-power":
            self.n0 = params['n_s'][0]
            self.theta_0 = params['theta_s'][0]
            self.theta_s = np.array(self.theta_s)
        
    def __str__(self):
        return "A PSF Model Class"

    def __repr__(self):
        return " ".join([f"{self.__class__.__name__}", f"<{self.aureole_model}>"])
                
    def pixelize(self, pixel_scale=DF_pixel_scale):
        """ Build grid for drawing """
        self.pixel_scale = pixel_scale
        
        for key, val in self.params.items():
            if ('gamma' in key) | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def update(self, params):
        """ Update PSF parameters from dictionary keys """
        pixel_scale = self.pixel_scale
        for key, val in params.items():
            if np.ndim(val) > 0:
                val = np.array(val)
                
            exec('self.' + key + ' = val')
            self.params[key] = val
            
            if ('gamma' in key) | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def copy(self):
        """ A deep copy of the object """
        return deepcopy(self)            

    @property
    def f_core1D(self):
        """ 1D Core function *in pix* """
        gamma_pix, beta = self.gamma_pix, self.beta
        c_mof2Dto1D = C_mof2Dto1D(gamma_pix, beta)
        return lambda r: moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D

    @property
    def f_aureole1D(self):
        """ 1D Aureole function *in pix* """
        if self.aureole_model == "moffat":
            gamma1_pix, beta1 = self.gamma1_pix, self.beta1
            c_mof2Dto1D = C_mof2Dto1D(gamma1_pix, beta1)
            f_aureole = lambda r: moffat1d_normed(r, gamma1_pix, beta1) / c_mof2Dto1D
        
        elif self.aureole_model == "power":
            n0, theta_0_pix = self.n0, self.theta_0_pix
            c_aureole_2Dto1D = C_pow2Dto1D(n0, theta_0_pix)
            f_aureole = lambda r: trunc_power1d_normed(r, n0, theta_0_pix) / c_aureole_2Dto1D

        elif self.aureole_model == "multi-power":
            n_s, theta_s_pix = self.n_s, self.theta_s_pix
            c_aureole_2Dto1D = C_mpow2Dto1D(n_s, theta_s_pix)
            f_aureole = lambda r: multi_power1d_normed(r, n_s, theta_s_pix) / c_aureole_2Dto1D

        return f_aureole


    def plot1D(self, **kwargs):
        """ Plot 1D profile """
        from .plotting import plot_PSF_model_1D
        
        plot_PSF_model_1D(self.frac, self.f_core1D, self.f_aureole1D, **kwargs)
        
        if self.aureole_model == "multi-power":
            for t in self.theta_s_pix:
                plt.axvline(t, ls="--", color="k", alpha=0.3, zorder=1)
                
    def generate_core(self):
        """ Generate Galsim PSF of core. """
        gamma, beta = self.gamma, self.beta
        self.fwhm = fwhm = gamma * 2. * math.sqrt(2**(1./beta)-1)
        
        psf_core = galsim.Moffat(beta=beta, fwhm=fwhm,
                                 flux=1., gsparams=self.gsparams) # in arcsec
        self.psf_core = psf_core
        return psf_core
    
    def generate_aureole(self,
                         contrast=1e6,
                         psf_scale=None,
                         psf_range=None,
                         min_psf_range=60,
                         max_psf_range=1200,
                         interpolant="linear"):
        """
        Generate Galsim PSF of aureole.

        Parameters
        ----------
        contrast: Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given.
        psf_scale: Pixel scale of the PSF, <= pixel scale of data. In arcsec/pix.
        psf_range: Range of PSF. In arcsec.
        min_psf_range : Minimum range of PSF. In arcsec.
        max_psf_range : Maximum range of PSF. In arcsec.
        interpolant: Interpolant method in Galsim.
        
        Returns
        ----------
        psf_aureole: power law Galsim PSF, flux normalized to be 1.
        psf_size: Full image size of PSF used. In pixel.
        
        """
        
        if psf_scale is None:
            psf_scale = self.pixel_scale
            
        if self.aureole_model == "moffat":
            gamma1, beta1 = self.gamma1, self.beta1
            
            if psf_range is None:
                psf_range = max_psf_range
            psf_size = round_good_fft(2 * psf_range // psf_scale)   
    
        else:

            if psf_range is None:
                psf_size = calculate_psf_size(self.n0, self.theta_0, contrast,
                                              psf_scale, min_psf_range, max_psf_range)
            else:
                psf_size = round_good_fft(psf_range) 
        
            # Generate Grid of PSF and plot PSF model in real space onto it
            xx_psf, yy_psf, cen_psf = generate_psf_grid(psf_size)
        
        if self.aureole_model == "moffat":
            psf_aureole = galsim.Moffat(beta=beta1, scale_radius=gamma1,
                                        flux=1., gsparams=self.gsparams)
            
        else:
            if self.aureole_model == "power":
                theta_0_pix = self.theta_0 / psf_scale
                psf_model = trunc_power2d(xx_psf, yy_psf,
                                          self.n0, theta_0_pix, I_theta0=1, cen=cen_psf)

            elif self.aureole_model == "multi-power":
                theta_s_pix = self.theta_s / psf_scale
                psf_model =  multi_power2d(xx_psf, yy_psf,
                                           self.n_s, theta_s_pix, 1, cen=cen_psf) 

            # Parse the image to Galsim PSF model by interpolation
            image_psf = galsim.ImageF(psf_model)
#            self.image_psf = image_psf/image_psf.sum()
            psf_aureole = galsim.InterpolatedImage(image_psf, flux=1,
                                                   scale=psf_scale,
                                                   x_interpolant=interpolant,
                                                   k_interpolant=interpolant)
        self.psf_aureole = psf_aureole
        self.theta_out = max_psf_range
        
        return psf_aureole, psf_size   

        
    def Flux2Amp(self, Flux):
        """ Convert Flux to Astropy Moffat Amplitude (pixel unit) """
        
        Amps = [moffat2d_Flux2Amp(self.gamma_pix, self.beta, Flux=(1-self.frac)*F)
                for F in Flux]
        return np.array(Amps)
    
    
    def I2I0(self, I, r=12):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "moffat":
            return I2I0_mof(self.gamma1_pix, self.beta1, r, I=I)
        
        elif self.aureole_model == "power":
            return I2I0_pow(self.n0, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2I0_mpow(self.n_s, self.theta_s_pix, r, I=I)
        
    def I02I(self, I0, r=12):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "moffat":
            return I02I_mof(self.gamma1_pix, self.beta1, r, I0=I0)
        
        elif self.aureole_model == "power":
            return I02I_pow(self.n0, self.theta_0_pix, r, I0=I0)
        
        elif self.aureole_model == "multi-power":
            return I02I_mpow(self.n_s, self.theta_s_pix, r, I0=I0)
    
    def calculate_external_light(self, stars, n_iter=2):
        """ Calculate the integrated external scatter light that affects
        the flux scaling from very bright stars on the other stars.
        
        Parameters
        ----------
        stars : Star object
        n_iter : iteration time to do the calculation
        
        """
        
        I_ext = np.zeros(stars.n_bright)
        
        if self.aureole_model == "moffat":
            pass
        
        else:
            z_norm_verybright0 = stars.z_norm_verybright.copy()
            pos_source, pos_eval = stars.star_pos_verybright, stars.star_pos_bright

            if self.aureole_model == "power":
                cal_ext_light = partial(calculate_external_light_pow,
                                       n0=self.n0, theta0=self.theta_0_pix,
                                       pos_source=pos_source, pos_eval=pos_eval)
            elif self.aureole_model == "multi-power":
                cal_ext_light = partial(calculate_external_light_mpow,
                                        n_s=self.n_s, theta_s_pix=self.theta_s_pix,
                                        pos_source=pos_source, pos_eval=pos_eval)
            # Loop the subtraction    
            r_scale = stars.r_scale
            n_verybright = stars.n_verybright
            for i in range(n_iter):
                z_norm_verybright = z_norm_verybright0 - I_ext[:n_verybright]
                I0_verybright = self.I2I0(z_norm_verybright, r=r_scale)
                I_ext = cal_ext_light(I0_source=I0_verybright)
            
        return I_ext
    
    def I2Flux(self, I, r=12):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return I2Flux_mof(self.frac, self.gamma1_pix, self.beta1, r, I=I)
        
        elif self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n0, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2Flux_mpow(self.frac, self.n_s, self.theta_s_pix, r, I=I)
        
    def Flux2I(self, Flux, r=12):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return Flux2I_mof(self.frac, self.gamma1_pix, self.beta1, r, Flux=Flux)
        
        elif self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n0, self.theta_0_pix, r, Flux=Flux)
        
        elif self.aureole_model == "multi-power":
            return Flux2I_mpow(self.frac, self.n_s, self.theta_s_pix, r,  Flux=Flux)
        
    def SB2Flux(self, SB, BKG, ZP, r=12):
        """ Convert suface brightness SB at r to total flux, given background value and ZP. """
        # Intensity = I + BKG
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale) - BKG
        return self.I2Flux(I, r=r)
    
    def Flux2SB(self, Flux, BKG, ZP, r=12):
        """ Convert total flux to suface brightness SB at r, given background value and ZP. """
        I = self.Flux2I(Flux, r=r)
        return Intensity2SB(I+ BKG, BKG, ZP, self.pixel_scale)
    
    @property
    def psf_star(self):
        """ Galsim object of star psf (core+aureole) """
        frac = self.frac
        psf_core, psf_aureole = self.psf_core, self.psf_aureole
        return (1-frac) * psf_core + frac * psf_aureole

    def plot_PSF_model_galsim(self, contrast=None, save=False, save_dir='.'):
        """ Build and plot Galsim 2D model averaged in 1D """
        from .plotting import plot_PSF_model_galsim
        image_psf = plot_PSF_model_galsim(self, contrast=contrast,
                                          save=save, save_dir=save_dir)
        self.image_psf = image_psf / image_psf.array.sum()

        
    @staticmethod
    def write_psf_image(image_psf, filename='PSF_model.fits'):
        """ Write the 2D psf image to fits """
        hdu = fits.ImageHDU(image_psf)
        hdu.writeto(filename, overwrite=True)
    
    def draw_core2D_in_real(self, star_pos, Flux):
        """ 2D drawing function of the core in real space given positions and flux (of core) of target stars """
        
        gamma, alpha = self.gamma_pix, self.beta
        Amps = np.array([moffat2d_Flux2Amp(gamma, alpha, Flux=flux)
                       for flux in Flux])
        f_core_2d_s = np.array([models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0,
                                                gamma=gamma, alpha=alpha)
                                for ((x0,y0), amp) in zip(star_pos, Amps)])
            
        return f_core_2d_s
        
    def draw_aureole2D_in_real(self, star_pos, Flux=None, I0=None):
        """ 2D drawing function of the aureole in real space given positions and flux / amplitude (of aureole) of target stars """
        
        if self.aureole_model == "moffat":
            gamma1_pix, alpha1 = self.gamma1_pix, self.beta1
            
            # In this case I_theta0 is defined as the amplitude at gamma
            if I0 is None:
                I_theta0 = moffat2d_Flux2I0(gamma1_pix, alpha1, Flux=Flux)
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
                
            Amps = np.array([moffat2d_I02Amp(alpha1, I0=I0)
                             for I0 in I_theta0])
            
            f_aureole_2d_s = np.array([models.Moffat2D(amplitude=amp,
                                                       x_0=x0, y_0=y0,
                                                       gamma=gamma1_pix,
                                                       alpha=alpha1)
                                    for ((x0,y0), amp) in zip(star_pos, Amps)])
            
        elif self.aureole_model == "power":
            n0 = self.n0
            theta_0_pix = self.theta_0_pix
            
            if I0 is None:
                I_theta0 = power2d_Flux2Amp(n0, theta_0_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      trunc_power2d(xx, yy, cen=cen,
                                                    n=n0, theta0=theta_0_pix,
                                                    I_theta0=I)
                                      for (I, pos) in zip(I_theta0, star_pos)])

        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s_pix = self.theta_s_pix
            
            if I0 is None:
                I_theta0 = multi_power2d_Flux2Amp(n_s, theta_s_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      multi_power2d(xx, yy, cen=cen,
                                                    n_s=n_s, theta_s=theta_s_pix,
                                                    I_theta0=I)
                                      for (I, pos) in zip(I_theta0, star_pos)])
            
        return f_aureole_2d_s

    
class Stars:
    """
    Class storing positions & flux of faint/medium-bright/bright stars
    
    """
    def __init__(self, star_pos, Flux, 
                 Flux_threshold=[7e4, 2.7e6],
                 z_norm=None, r_scale=12, 
                 BKG=0, verbose=False):
        """
        Parameters
        ----------
        star_pos: positions of stars (in the region)
        Flux: flux of stars (in ADU)
        
        Flux_threshold : thereshold of flux
                (default: corresponding to [15, 11] mag for DF)
        z_norm : flux scaling measured at r_scale
        r_scale : radius at which to measure the flux scaling
        BKG : sky background value
                
        """
        self.star_pos = np.atleast_2d(star_pos)
        self.Flux = np.atleast_1d(Flux)
        self.Flux_threshold = Flux_threshold

        self.F_bright = Flux_threshold[0]
        self.F_verybright = Flux_threshold[1]
        
        self.n_tot = len(star_pos)
    
        self.bright = (self.Flux >= self.F_bright)
        self.verybright = (self.Flux >= self.F_verybright)
        self.medbright = self.bright & (~self.verybright)
        
        if z_norm is not None:
            self.z_norm = z_norm
            
        self.r_scale = r_scale
        self.BKG = BKG

        self.verbose = verbose
        
        if verbose:
            if len(Flux[self.medbright])>0:
                print("# of medium bright stars: %d (flux range:%.2g~%.2g) "\
                      %(self.n_medbright, Flux[self.medbright].min(), Flux[self.medbright].max()))
            
            if len(Flux[self.verybright])>0:
                print("# of very bright stars : %d (flux range:%.2g~%.2g)"\
                      %(self.n_verybright, Flux[self.verybright].min(), Flux[self.verybright].max()))
            
            # Rendering stars in parallel if number of bright stars exceeds 50
            if self.n_medbright < 50:
                print("Not many bright stars, will draw in serial.\n")
                self.parallel = False 
            else: 
                print("Crowded fields w/ bright stars > 50, will draw in parallel.\n")
                self.parallel = True
    
    def __str__(self):
        return "A Star Class"

    def __repr__(self):
        return ' N='.join([f"{self.__class__.__name__}", str(self.n_tot)])

    @classmethod            
    def from_znorm(cls, psf, star_pos, z_norm,
                   z_threshold=[10, 300], r_scale=12, 
                   verbose=False):
        """ Star object built from intensity at r_scale instead of flux """
        Flux = psf.I2Flux(z_norm, r_scale)
        Flux_threshold = psf.I2Flux(z_threshold, r=r_scale)
        
        return cls(star_pos, Flux, Flux_threshold,
                   z_norm=z_norm, r_scale=r_scale, verbose=verbose)
    
    def update_Flux(self, Flux):
        self.Flux = Flux
        

    @lazyproperty
    def n_faint(self):
        return np.sum(~self.bright)

    @lazyproperty
    def n_bright(self):
        return np.sum(self.bright)

    @lazyproperty
    def n_verybright(self):
        return np.sum(self.verybright)

    @lazyproperty
    def n_medbright(self):
        return np.sum(self.medbright)

    @property
    def Flux_faint(self):
        return self.Flux[~self.bright]

    @property
    def Flux_bright(self):
        return self.Flux[self.bright]

    @property
    def Flux_verybright(self):
        return self.Flux[self.verybright]

    @property
    def Flux_medbright(self):
        return self.Flux[self.medbright]

    @property
    def z_norm_bright(self):
        return self.z_norm[self.bright]
    
    @property
    def z_norm_verybright(self):
        return self.z_norm[self.verybright]
    
    @lazyproperty
    def star_pos_faint(self):
        return self.star_pos[~self.bright]

    @lazyproperty
    def star_pos_bright(self):
        return self.star_pos[self.bright]

    @lazyproperty
    def star_pos_verybright(self):
        return self.star_pos[self.verybright]

    @lazyproperty
    def star_pos_medbright(self):
        return self.star_pos[self.medbright]
            
    def plot_flux_dist(self, **kwargs):
        from .plotting import plot_flux_dist
        plot_flux_dist(self.Flux, [self.F_bright, self.F_verybright], **kwargs)

    def copy(self):
        return deepcopy(self) 
    
    def use_verybright(self):
        """ Crop the object into a new object only contains its very bright stars """
        if self.verbose:
            print("\nOnly model brightest stars in the field.\n")
            
        stars_vb = Stars(self.star_pos_verybright,
                         self.Flux_verybright,
                         Flux_threshold=self.Flux_threshold,
                         z_norm=self.z_norm_verybright,
                         r_scale=self.r_scale, BKG=self.BKG)
        
        return stars_vb
    
    def remove_outsider(self, image_shape, gap=[36,12], verbose=False):
        """ Remove out-of-field stars far from the edge. """
        
        star_pos = self.star_pos
        Flux = self.Flux
        
        def out(d):
            out_max = np.vstack([(star_pos[:,0]>image_shape[1]+d),
                                 (star_pos[:,1]>image_shape[0]+d)]).T
            out_min = (star_pos<-d)
            out = out_min | out_max
            return np.logical_or.reduce(out, axis=1)
            
        remove_A = out(gap[0]) & self.verybright
        remove_B = out(gap[1]) & self.medbright

        remove = remove_A | remove_B
        stars_new = Stars(star_pos[~remove], Flux[~remove],
                          self.Flux_threshold, self.z_norm[~remove],
                          r_scale=self.r_scale, BKG=self.BKG, verbose=verbose)
        return stars_new
     
    def save(self, name='stars', save_dir='./'):
        from .io import save_pickle
        save_pickle(self, os.path.join(save_dir, name+'.pkl'))
        
        

############################################
# Analytic Functions for models
############################################

### funcs on single element ###

def trunc_pow(x, n, theta0, I_theta0=1):
    """ Truncated power law for single element, I = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * x**(-n) if x > theta0 else I_theta0
    return y

# deprecated
def compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0):
    """ Compute normalization factor of each power law component """
    a_s = np.zeros(len(n_s))
    a0 = I_theta0 * theta0**(n0)

    I_theta_i = a0 * float(theta_s[0])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        try:
            a_s[i] = a_i
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass    
    return a0, a_s

@njit
def compute_multi_pow_norm(n_s, theta_s, I_theta0):
    """ Compute normalization factor A of each power law component A_i*(theta)^(n_i)"""
    n0, theta0 = n_s[0], theta_s[0]
    a0 = I_theta0 * theta0**(n0)
    a_s = np.zeros(len(n_s))   
    a_s[0] = a0
    
    I_theta_i = a0 * float(theta_s[1])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s[1:], theta_s[1:])):
#         if (i+2) == len(n_s):
#             break
        a_i = I_theta_i/(theta_s[i+1])**(-n_i)
        a_s[i+1] = a_i
        I_theta_i = a_i * float(theta_s[i+2])**(-n_i)
    return a_s

# deprecated
def multi_pow0(x, n0, n_s, theta0, theta_s, I_theta0, a0=None, a_s=None):
    """ Continuous multi-power law for single element """
    if a0 is None:
        a0, a_s = compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0)
        
    if x <= theta0:
        return I_theta0
    elif x<= theta_s[0]:
        y = a0 * x**(-n0)
        return y
    else:
        for k in range(len(a_s-1)):
            try:
                if x <= theta_s[k+1]:
                    y = a_s[k] * x**(-n_s[k])
                    return y
            except IndexError:
                pass
        else:
            y = a_s[k] * x**(-n_s[k])
            return y
        
def multi_pow(x, n_s, theta_s, I_theta0, a_s=None):
    """ Continuous multi-power law for single element """
    
    if a_s is None:
        a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
    
    if x <= theta0:
        return I_theta0
    elif x<= theta_s[1]:
        y = a0 * x**(-n0)
        return y
    else:
        for k in range(len(a_s)):
            try:
                if x <= theta_s[k+2]:
                    y = a_s[k+1] * x**(-n_s[k+1])
                    return y
            except IndexError:
                pass
        else:
            y = a_s[-1] * x**(-n_s[-1])
            return y

### 1D functions ###

def log_linear(x, k, x0, y0):
    """ linear function y ~ k * log x passing (x0,y0) """
    x_ = np.log10(x)
    return k * x_ + (y0-k*np.log10(x0))

def power1d(x, n, theta0, I_theta0):
    """ Power law for 1d array, I = I_theta0 at theta0, theta in pix """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x + 1e-6, -n)
    return y

def trunc_power1d(x, n, theta0, I_theta0=1): 
    """ Truncated power law for 1d array, I = I_theta0 at theta0, theta in pix """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x + 1e-6, -n) 
    y[x<=theta0] = I_theta0
    return y

# deprecated
def multi_power1d0(x, n0, theta0, I_theta0, n_s, theta_s):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a0, a_s = compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0)
    
    y = a0 * np.power(x + 1e-6, -n0)
    y[x<=theta0] = I_theta0
    for i, (n_i, a_i, theta_i) in enumerate(zip(n_s, a_s, theta_s)):
        y_i = a_i * np.power(x, -n_i)
        y[x>theta_i] = y_i[x>theta_i]
    return y

def multi_power1d(x, n_s, theta_s, I_theta0):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    theta0 = theta_s[0]
    
    y = np.zeros_like(x)
    y[x<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (x>theta_s[k]) & (x<=theta_s[k+1]) if k<len(a_s)-1 else (x>theta_s[k])  
        y[reg] = a_s[k] * np.power(x[reg], -n_s[k])
    return y


def moffat_power1d(x, gamma, alpha, n, theta0, A=1):
    """ Moffat + Power for 1d array, flux normalized = 1, theta in pix """
    Mof_mod_1d = models.Moffat1D(amplitude=A, x_0=0, gamma=gamma, alpha=alpha)
    y[x<=theta0] = Mof_mod_1d(x)
    y[x>theta0] = power1d(x[x>theta0], n, theta0, Mof_mod_1d(theta0))
    return y

def trunc_power1d_normed(x, n, theta0):
    """ Truncated power law for 1d array, flux normalized = 1, theta in pix """
    norm_pow = quad(trunc_pow, 0, np.inf, args=(n, theta0, 1))[0]
    y = trunc_power1d(x, n, theta0, 1) / norm_pow  
    return y

def moffat1d_normed(x, gamma, alpha):
    """ Moffat for 1d array, flux normalized = 1 """
    Mof_mod_1d = models.Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    norm_mof = quad(Mof_mod_1d, 0, np.inf)[0] 
    y = Mof_mod_1d(x) / norm_mof
    return y

def multi_power1d_normed(x, n_s, theta_s):
    """ Multi-power law for 1d array, flux normalized = 1, theta in pix """
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    norm_mpow = quad(multi_pow, 0, np.inf,
                     args=(n_s, theta_s, 1, a_s), limit=100)[0]
    y = multi_power1d(x, n_s, theta_s, 1) / norm_mpow
    return y

### 2D functions ###

def map2d(f, xx=None, yy=None):
    return f(xx,yy)

def map2d_k(k, func_list, xx=None, yy=None):
    return func_list[k](xx, yy)

@lru_cache(maxsize=16)
def generate_psf_grid(psf_size):
    # Generate Grid of PSF and plot PSF model in real space onto it
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
    return xx_psf, yy_psf, cen_psf

def power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    return z 

@njit
def trunc_power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Truncated power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2).ravel() + 1e-6
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    z[rr<=theta0] = I_theta0
    return z.reshape(xx.shape)

# deprecated
def multi_power2d_cover(xx, yy, n0, theta0, I_theta0, n_s, theta_s, cen):
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(rr, -n0) 
    z[rr<=theta0] = I_theta0
    
    I_theta_i = a0 * float(theta_s[0])**(-n0)
    
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(rr, -n_i)
        z[rr>theta_i] = z_i[rr>theta_i]
        try:
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass
    return z

@njit
def multi_power2d(xx, yy, n_s, theta_s, I_theta0, cen):
    """ Multi-power law for 2d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2).ravel()
    z = np.zeros(xx.size) 
    theta0 = theta_s[0]
    z[rr<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (rr>theta_s[k]) & (rr<=theta_s[k+1]) if k<len(a_s)-1 else (rr>theta_s[k])     
        z[reg] = a_s[k] * np.power(rr[reg], -n_s[k])
        
    return z.reshape(xx.shape)


### Flux/Amplitude Convertion ###

def moffat1d_Flux2Amp(r_core, beta, Flux=1):
    """ Calculate the (astropy) amplitude of 1d Moffat profile given the core width, power index, and total flux F.
    Note in astropy unit (x,y) the amplitude should be scaled with 1/sqrt(pi)."""
    Amp = Flux * Gamma(beta) / ( r_core * np.sqrt(np.pi) * Gamma(beta-1./2) ) # Derived scaling factor
    return  Amp

def moffat1d_Amp2Flux(r_core, beta, Amp=1):
    Flux = Amp / moffat1d_Flux2Amp(r_core, beta, Flux=1)
    return  Flux

def power1d_Flux2Amp(n, theta0, Flux=1, trunc=True):
    if trunc:
        I_theta0 = Flux * (n-1)/n / theta0
    else:
        I_theta0 = Flux * (n-1) / theta0
    return I_theta0

def power1d_Amp2Flux(n, theta0, Amp=1, trunc=True):
    if trunc:
        Flux = Amp * n/(n-1) * theta0
    else:
        Flux = Amp * 1./(n-1) * theta0
    return Flux

def moffat2d_Flux2Amp(r_core, beta, Flux=1):
    return Flux * (beta-1) / r_core**2 / np.pi

def moffat2d_Amp2Flux(r_core, beta, Amp=1):
    return Amp / moffat2d_Flux2Amp(r_core, beta, Flux=1)

def moffat2d_Flux2I0(r_core, beta, Flux=1):
    Amp = moffat2d_Flux2Amp(r_core, beta, Flux=Flux)
    return moffat2d_Amp2I0(beta, Amp=Amp)

def moffat2d_I02Amp(beta, I0=1):
    # Convert I0(r=r_core) to Amplitude
    return I0 * 2**(2*beta)

def moffat2d_Amp2I0(beta, Amp=1):
    # Convert I0(r=r_core) to Amplitude
    return Amp * 2**(-2*beta)

# def power2d_Flux2Amp(n, theta0, Flux=1, trunc=True):
#     if trunc:
#         I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
#     else:
#         I_theta0 = (1./np.pi) * Flux * (n-2)/2 / theta0**2
#     return I_theta0

# def power2d_Amp2Flux(n, theta0, Amp=1, trunc=True):
#     return Amp / power2d_Flux2Amp(n, theta0, Flux=1, trunc=trunc)

# def power2d_Flux2Amp(n, theta0, Flux=1, r_trunc=500):
#     if n>2:
#         I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
#     elif n<2:
#         I_theta0 = (1./np.pi) * Flux / (1 + 2*r_trunc**(2-n)/(2-n)) / theta0**2
#     else:
#         I_theta0 = (1./np.pi) * Flux / (1 + 2*math.log(r_trunc/theta0)) / theta0**2
#     return I_theta0

def power2d_Flux2Amp(n, theta0, Flux=1):
    if n>2:
        I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    else:
        raise InconvergenceError('PSF is not convergent in Infinity.')
        
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1)
            
def multi_power2d_Amp2Flux(n_s, theta_s, Amp=1, theta_trunc=1e5):
    """ convert amplitude(s) to integral flux with 2D multi-power law """
    if np.ndim(Amp)>0:
        a_s = compute_multi_pow_norm(n_s, theta_s, 1)
        a_s = np.multiply(a_s[:,np.newaxis], Amp)
    else:
        a_s = compute_multi_pow_norm(n_s, theta_s, Amp)

    I_2D = sum_I2D_multi_power2d(Amp, a_s, n_s, theta_s, theta_trunc)
        
    return I_2D

@njit
def sum_I2D_multi_power2d(Amp, a_s, n_s, theta_s, theta_trunc=1e5):
    """ Supplementary function for multi_power2d_Amp2Flux tp speed up """
    
    theta0 = theta_s[0]
    I_2D = Amp * np.pi * theta0**2

    for k in range(len(n_s)-1):

        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * math.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2)

    if n_s[-1] > 2:
        I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2) 
    elif n_s[-1] == 2:
        I_2D += 2*np.pi * a_s[-1] * math.log(theta_trunc/theta_s[-1])
    else:
        I_2D += 2*np.pi * a_s[-1] * (theta_trunc**(2-n_s[-1]) - theta_s[-1]**(2-n_s[-1])) / (2-n_s[-1])
        
    return I_2D

def multi_power2d_Flux2Amp(n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n_s, theta_s, Amp=1)


def I2I0_mof(r_core, beta, r, I=1):
    """ Convert Intensity I(r) at r to I at r_core with moffat.
        r_core and r in pixel """
    Amp = I * (1+(r/r_core)**2)**beta
    I0 = moffat2d_Amp2I0(beta, Amp)
    return I0

def I02I_mof(r_core, beta, r, I0=1):
    """ Convert I at r_core to Intensity I(r) at r with moffat.
        r_core and r in pixel """
    Amp = moffat2d_I02Amp(beta, I0)
    I = Amp * (1+(r/r_core)**2)**(-beta)
    return I

def I2Flux_mof(frac, r_core, beta, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of moffat.
        r_core and r in pixel """
    Amp = I * (1+(r/r_core)**2)**beta
    Flux_mof = moffat2d_Amp2Flux(r_core, beta, Amp=Amp)
    Flux_tot = Flux_mof / frac
    return Flux_tot

def Flux2I_mof(frac, r_core, beta, r, Flux=1):
    """ Convert total flux  at r to Intensity I(r) with fraction of moffat.
        r_core and r in pixel """
    Flux_mof = Flux * frac
    Amp = moffat2d_Flux2Amp(r_core, beta, Flux=Flux_mof)
    I = Amp * (1+(r/r_core)**2)**(-beta)
    return I


def I2I0_pow(n0, theta0, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with power law.
        theata_s and r in pixel """
    I0 = I * (r/theta0)**n0
    return I0

def I02I_pow(n0, theta0, r, I0=1):
    """ Convert Intensity I(r) at r to I at theta_0 with power law.
        theata_s and r in pixel """
    I = I0 / (r/theta0)**n0
    return I

def I2Flux_pow(frac, n0, theta0, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of power law.
        theata0 and r in pixel """
    I0 = I2I0_pow(n0, theta0, r, I=I)
    Flux_pow = power2d_Amp2Flux(n0, theta0, Amp=I0)
    Flux_tot = Flux_pow / frac
    return Flux_tot

def Flux2I_pow(frac, n0, theta0, r, Flux=1):
    """ Convert total flux to Intensity I(r) at r.
        theata0 and r in pixel """
    Flux_pow = Flux * frac
    I0 = power2d_Flux2Amp(n0, theta0, Flux=Flux_pow)
    I = I0 / (r/theta0)**n0
    return I

def I2I0_mpow(n_s, theta_s_pix, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s_pix, right=True) - 1
    I0 = I * r**(n_s[i]) * theta_s_pix[0]**(-n_s[0])
    for j in range(i):
        I0 *= theta_s_pix[j+1]**(n_s[j]-n_s[j+1])
        
    return I0

def I02I_mpow(n_s, theta_s_pix, r, I0=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s_pix, right=True) - 1
        
    I = I0 / r**(n_s[i]) / theta_s_pix[0]**(-n_s[0])
    for j in range(i):
        I *= theta_s_pix[j+1]**(n_s[j+1]-n_s[j])
        
    return I


def calculate_external_light_pow(n0, theta0, pos_source, pos_eval, I0_source):
    # Calculate light produced by source (I0, pos_source) at pos_eval. 
    r_s = distance.cdist(pos_source,  pos_eval)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    I_s = I0_s / (r_s/theta0)**n0
    I_s[(r_s==0)] = 0
    
    return I_s.sum(axis=0)

def calculate_external_light_mpow(n_s, theta_s_pix, pos_source, pos_eval, I0_source):
    # Calculate light produced by source (I0_source, pos_source) at pos_eval. 
    r_s = distance.cdist(pos_source, pos_eval)
    r_inds = np.digitize(r_s, theta_s_pix, right=True) - 1
    
    r_inds_uni, r_inds_inv = np.unique(r_inds, return_inverse=True)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    # I(r) = I0 * (theta0/theta1)^(n0) * (theta1/theta2)^(n1) *...* (theta_{k}/r)^(nk)
    I_s = I0_s * theta_s_pix[0]**n_s[0] / r_s**(n_s[r_inds])
    factors = np.array([np.prod([theta_s_pix[j+1]**(n_s[j+1]-n_s[j])
                                 for j in range(i)]) for i in r_inds_uni])
    I_s *= factors[r_inds_inv].reshape(len(I0_source),-1)
    
    I_s[(r_s==0)] = 0
    
    return I_s.sum(axis=0)
    
# #deprecated
# def I02I_mpow_2d(n_s, theta_s, r_s, I0=1):
#     """ Convert Intensity I(r) at multiple r to I at theta_0 with multi-power law.
#         theata_s and r in pixel
        
#         return I (# of I0, # of distance) """
#     r_inds = np.digitize(r_s, theta_s, right=True) - 1
#     r_inds_uni, r_inds_inv = np.unique(r_inds, return_inverse=True)
#     I0 = np.atleast_1d(I0)

#     I0_s = np.repeat(I0[:, np.newaxis], r_s.shape[-1], axis=1) 
    
#     I_s = I0_s / r_s**(n_s[r_inds]) / theta_s[0]**(-n_s[0])
    
#     factors = np.array([np.prod([theta_s[j+1]**(n_s[j+1]-n_s[j])
#                                  for j in range(i)]) for i in r_inds_uni])
#     I_s *= factors[r_inds_inv]
    
#     return I_s

# #deprecated
# def extract_external_light(I_s):
#     inds = np.arange(I_s.shape[0])
#     comb_inds = np.array(list(combinations(inds, 2)))
#     mutual = (comb_inds, inds[:,np.newaxis])
#     I_sum = np.zeros_like(I_s.shape[0])
#     for (c_ind, I) in zip(comb_inds,I_s[mutual]):
#         I_sum[c_ind[0]] += I[1]
#         I_sum[c_ind[1]] += I[0]
#     return I_sum


def I2Flux_mpow(frac, n_s, theta_s, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of multi-power law.
        theata_s and r in pixel """

    I0 = I2I0_mpow(n_s, theta_s, r, I=I)
    Flux_mpow = multi_power2d_Amp2Flux(n_s=n_s, theta_s=theta_s, Amp=I0)
    Flux_tot = Flux_mpow / frac
    
    return Flux_tot

def Flux2I_mpow(frac, n_s, theta_s, r, Flux=1):
    """ Convert total flux to Intensity I(r) at r.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s, right=True) - 1
    
    Flux_mpow = Flux * frac
    I0 = multi_power2d_Flux2Amp(n_s=n_s, theta_s=theta_s, Flux=Flux_mpow)
    
    I = I0 / r**(n_s[i]) / theta_s[0]**(-n_s[0])
    for j in range(i):
        I /= theta_s[j+1]**(n_s[j]-n_s[j+1])

    return I


### 1D/2D conversion factor ###

def C_mof2Dto1D(r_core, beta):
    """ gamma in pixel """
    return 1./(beta-1) * 2*math.sqrt(np.pi) * r_core * Gamma(beta) / Gamma(beta-1./2) 

def C_mof1Dto2D(r_core, beta):
    """ gamma in pixel """
    return 1. / C_mof2Dto1D(r_core, beta)

@njit
def C_pow2Dto1D(n, theta0):
    """ theta0 in pixel """
    return np.pi * theta0 * (n-1) / (n-2)

@njit
def C_pow1Dto2D(n, theta0):
    """ theta0 in pixel """
    return 1. / C_pow2Dto1D(n, theta0)

@njit
def C_mpow2Dto1D(n_s, theta_s):
    """ theta in pixel """
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
 
    I_2D = 1. * np.pi * theta0**2
    for k in range(len(n_s)-1):
        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2) 
    I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2)   
    
    I_1D = 1. * theta0
    for k in range(len(n_s)-1):
        if n_s[k] == 1:
            I_1D += a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_1D += a_s[k] * (theta_s[k]**(1-n_s[k]) - theta_s[k+1]**(1-n_s[k])) / (n_s[k]-1) 
    I_1D += a_s[-1] * theta_s[-1]**(1-n_s[-1]) / (n_s[-1]-1)
    
    return I_2D / I_1D 

@njit
def C_mpow1Dto2D(n_s, theta_s):
    """ theta in pixel """
    return 1. / C_mpow2Dto1D(n_s, theta_s)



############################################
# Functions for PSF rendering with Galsim
############################################

def get_center_offset(pos):
# Shift center for the purpose of accuracy (by default galsim round to integer!)
    # Originally should be x_pos, y_pos = pos + 1 (ref galsim demo)
    # But origin of star_pos in SE is (1,1) but (0,0) in python
    x_pos, y_pos = pos
    
    x_nominal = x_pos + 0.5
    y_nominal = y_pos + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    return (ix_nominal, iy_nominal), offset  

def draw_star(k, star_pos, Flux,
              psf_star, psf_size, full_image,
              pixel_scale=DF_pixel_scale):
    """ Draw star #k at position star_pos[k] with Flux[k], using a combined PSF (psf_star) on full_image"""
# Function of drawing, practically devised to facilitate parallelization.
    stamp, bounds = get_stamp_bounds(k, star_pos, Flux, psf_star, psf_size,
                                     full_image, pixel_scale=pixel_scale)
    full_image[bounds] += stamp[bounds]

def get_stamp_bounds(k, star_pos, Flux,
                     psf_star, psf_size, full_image,
                     pixel_scale=DF_pixel_scale):
    """ Get stamp and boundary of star #k at position star_pos[k] with Flux[k], using a combined PSF (psf_star) on full_image"""
    pos, flux = star_pos[k], Flux[k]       
    star = psf_star.withFlux(flux)

    # Account for the fractional part of the position
    (ix_nominal, iy_nominal), offset = get_center_offset(pos)

    stamp = star.drawImage(nx=psf_size, ny=psf_size, scale=pixel_scale,
                           offset=offset, method='no_pixel')
    stamp.setCenter(ix_nominal, iy_nominal)
    
    bounds = stamp.bounds & full_image.bounds
    
    return stamp, bounds


############################################
# Functions for making mock images
############################################

def add_image_noise(image, noise_std, random_seed=42, verbose=True):
    """ Add Gaussian noise image """
    if verbose:
        print("Generate noise background w/ stddev = %.3g"%noise_std)
    
    Image = galsim.ImageF(image)
    rng = galsim.BaseDeviate(random_seed)
    
    gauss_noise = galsim.GaussianNoise(rng, sigma=noise_std)
    Image.addNoise(gauss_noise)
    
    return Image.array


def make_base_image(image_shape, stars, psf_base, pad=50, psf_size=64, verbose=True):
    """ Background images composed of dim stars with fixed PSF psf_base"""
    if verbose:
        print("Generate base image of faint stars (flux < %.2g)."%(stars.F_bright))
    
    start = time.time()
    nX0 = image_shape[1] + 2 * pad
    nY0 = image_shape[0] + 2 * pad
    full_image0 = galsim.ImageF(nX0, nY0)
    
    star_pos = stars.star_pos_faint + pad
    Flux = stars.Flux_faint
    
    if len(star_pos) == 0:
        return np.zeros((nY0, nX0))
    
    # draw faint stars with fixed PSF using galsim in Fourier space   
    for k in range(len(star_pos)):
        try:
            draw_star(k, star_pos=star_pos, Flux=Flux,
                      psf_star=psf_base, psf_size=psf_size, full_image=full_image0)
        except GalSimBoundsError as e:
            if verbose:
                print(e.__doc__)
                print(e.message)
            continue

    image_gs0 = full_image0.array
    
    end = time.time()
    if verbose: print("Total Time: %.3f s\n"%(end-start))
    
    return image_gs0[pad:nY0-pad, pad:nX0-pad]


def make_truth_image(psf, stars, image_shape, contrast=1e6,
                     parallel=False, verbose=False, saturation=4.5e4):
    
    """
    Draw truth image according to the given position & flux. 
    In two manners: 1) convolution in FFT w/ Galsim;
                and 2) plot in real space w/ astropy model. 

    """
    if verbose:
        print("Generate the truth image.")
        start = time.time()
    
    frac = psf.frac
    gamma_pix = psf.gamma_pix
    beta = psf.beta
    
    nY, nX = image_shape
    yy, xx = np.mgrid[:nY, :nX]
    
    psf_core = psf.psf_core
        
    psf_aureole = psf.psf_aureole
    
    full_image = galsim.ImageF(nX, nY)
    
    Flux_A = stars.Flux_bright
    star_pos_A = stars.star_pos_bright
    
    image_gs = full_image.array

    # Draw bright stars in real space
    func_core_2d_s = psf.draw_core2D_in_real(star_pos_A, (1-frac) * Flux_A)
    func_aureole_2d_s = psf.draw_aureole2D_in_real(star_pos_A, frac * Flux_A)

    # option for drawing in parallel
    if (not parallel) | (parallel_enabled==False) :
        if verbose: 
            print("Rendering bright stars in serial...")
        image_real = np.sum([f2d(xx,yy) + p2d(xx,yy) 
                             for (f2d, p2d) in zip(func_core_2d_s,
                                                   func_aureole_2d_s)], axis=0)
    else:
        if verbose: 
            print("Rendering bright stars in parallel...")
        func2d_s = np.concatenate([func_core_2d_s, func_aureole_2d_s])
        p_map2d = partial(map2d, xx=xx, yy=yy)
        
        image_stars = parallel_compute(func2d_s, p_map2d,
                                       lengthy_computation=False, verbose=verbose)
        image_real = np.sum(image_stars, axis=0)
    
    # combine the two image
    image = image_gs + image_real
    
    # saturation limit
    image[image>saturation] = saturation
        
    if verbose: 
        end = time.time()
        print("Total Time: %.3f s\n"%(end-start))
    
    return image
        
def generate_image_by_flux(psf, stars, xx, yy,
                           contrast=[1e5,1e6],
                           min_psf_range=90,
                           max_psf_range=1200,
                           psf_range=[None,None],
                           psf_scale=DF_pixel_scale,
                           parallel=False,
                           draw_real=True,
                           draw_core=False,
                           brightest_only=False,
                           interpolant='cubic'):
    """
    Generate the image by total flux, given the PSF object and Star object.
    
    Parameters
    ----------
    psf : PSF model describing the PSF model shape
    stars : Star model describing positions and scaling of stars 
    contrast : Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given in psf_range.
    min_psf_range : Minimum range of PSF if contrast is used. In arcsec.
    max_psf_range : Maximum range of PSF if contrast is used. In arcsec.
    psf_range : full range of PSF size (in arcsec) for drawing [medium, very] bright stars in convolution. Use contrast if not given.  (default: None)
    psf_scale : pixel scale of PSF. iN arcsec/pixel. Default to DF pixel scale.
    parallel : whether to run drawing for medium bright stars in parallel.
    draw_real : whether to draw very bright stars in real.
    draw_core : whether to draw the core for very bright stars in real.
    brightest_only : whether to draw very bright stars only.
    interpolant : Interpolant method in Galsim.
    
    Returns
    ----------
    image : drawn image
    
    """
    
    nY, nX = xx.shape
    
    frac = psf.frac
    
    if psf_scale is None:
        psf_scale = psf.pixel_scale
    
    if not(draw_real & brightest_only):
        psf_c = psf.psf_core
    
    # Setup the canvas
    full_image = galsim.ImageF(nX, nY)
        
    if not brightest_only:
        # Draw medium bright stars with galsim in Fourier space
        psf_e, psf_size = psf.generate_aureole(contrast=contrast[0],
                                               psf_scale=psf_scale,
                                               psf_range=psf_range[0],
                                               min_psf_range=min_psf_range//3,
                                               max_psf_range=max_psf_range//3,
                                               interpolant=interpolant)
        
        psf_size = psf_size // 2 * 2
        
        psf_star = (1-frac) * psf_c + frac * psf_e               
        
        if stars.n_medbright > 0:
            if (not parallel) | (parallel_enabled==False):
                # Draw in serial
                for k in range(stars.n_medbright):
                    draw_star(k,
                              star_pos=stars.star_pos_medbright,
                              Flux=stars.Flux_medbright,
                              psf_star=psf_star,
                              psf_size=psf_size,
                              full_image=full_image)
            else:
                # Draw in parallel, automatically back to serial computing if too few jobs
                p_get_stamp_bounds = partial(get_stamp_bounds,
                                             star_pos=stars.star_pos_medbright,
                                             Flux=stars.Flux_medbright,
                                             psf_star=psf_star,
                                             psf_size=psf_size,
                                             full_image=full_image)

                results = parallel_compute(np.arange(stars.n_medbright), p_get_stamp_bounds,
                                           lengthy_computation=False, verbose=False)

                for (stamp, bounds) in results:
                    full_image[bounds] += stamp[bounds]

    if draw_real:
        # Draw aureole of very bright star (if high cost in FFT) in real space
        # Note origin of star_pos in SE is (1,1) but (0,0) in python
        image_gs = full_image.array
        
        func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright-1,
                                                       Flux=frac * stars.Flux_verybright)
        image_aureole = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)
        
        if draw_core:
            func_core_2d_s = psf.draw_core2D_in_real(stars.star_pos_verybright-1,
                                                     Flux=(1-frac) * stars.Flux_verybright)
            image_gs += np.sum([f2d(xx,yy) for f2d in func_core_2d_s], axis=0)
                
        image = image_gs + image_aureole
        
    else:
        # Draw very bright star in Fourier space 
        psf_e_2, psf_size_2 = psf.generate_aureole(contrast=contrast[1],
                                                   psf_scale=psf_scale,
                                                   psf_range=psf_range[1],
                                                   min_psf_range=min_psf_range,
                                                   max_psf_range=max_psf_range,
                                                   interpolant=interpolant)
        
        psf_size_2 = psf_size_2 // 2 * 2
        
        psf_star_2 = (1-frac) * psf_c + frac * psf_e_2
        
        for k in range(stars.n_verybright):
            draw_star(k,
                      star_pos=stars.star_pos_verybright, 
                      Flux=stars.Flux_verybright,
                      psf_star=psf_star_2, 
                      psf_size=psf_size_2,
                      full_image=full_image)
            
        image = full_image.array

    return image

def generate_image_by_znorm(psf, stars, xx, yy,
                            contrast=[1e5,1e6],
                            min_psf_range=90,
                            max_psf_range=1200,
                            psf_range=[None,None],
                            psf_scale=DF_pixel_scale,
                            parallel=False,
                            draw_real=True,
                            brightest_only=False,
                            subtract_external=True,
                            draw_core=False,
                            interpolant='cubic'):
    """
    Generate the image by flux scaling, given the PSF object and Star object.
    
    Parameters
    ----------
    psf : PSF model describing the PSF model shape
    stars : Star model describing positions and scaling of stars
    xx, yy : image grid
    contrast : Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given in psf_range.
    min_psf_range : Minimum range of PSF if contrast is used. In arcsec.
    max_psf_range : Maximum range of PSF if contrast is used. In arcsec.
    psf_range : full range of PSF size (in arcsec) for drawing [medium, very] bright stars in convolution. (default: None)
    psf_scale : pixel scale of PSF. iN arcsec/pixel. Default to DF pixel scale.
    parallel : whether to run drawing for medium bright stars in parallel.
    draw_real : whether to draw very bright stars in real.
    brightest_only : whether to draw very bright stars only.
    draw_core : whether to draw the core for very bright stars in real.
    subtract_external : whether to subtract external scattter light from very bright stars.
    interpolant : Interpolant method in Galsim.
    
    Returns
    ----------
    image : drawn image
    
    """
    nY, nX = xx.shape
    
    frac = psf.frac
    r_scale = stars.r_scale

    z_norm = stars.z_norm.copy()
    # Subtract external light from brightest stars
    if subtract_external:
        I_ext = psf.calculate_external_light(stars)
        z_norm[stars.bright] -= I_ext
    
    if draw_real & brightest_only:
        # Skip computation of Flux, and ignore core PSF
        I0_verybright = psf.I2I0(z_norm[stars.verybright], r_scale)
        
    else:
        # Core PSF
        psf_c = psf.psf_core

        # Update stellar flux:
        z_norm[z_norm<=0] = z_norm[z_norm>0].min()/10 # problematic negatives
        Flux = psf.I2Flux(z_norm, r_scale)
        stars.update_Flux(Flux) 
        
    # Setup the canvas
    full_image = galsim.ImageF(nX, nY)
        
    if not brightest_only:
        # 1. Draw medium bright stars with galsim in Fourier space
        psf_e, psf_size = psf.generate_aureole(contrast=contrast[0],
                                               psf_scale=psf_scale,
                                               psf_range=psf_range[0],
                                               min_psf_range=min_psf_range//3,
                                               max_psf_range=max_psf_range//3,
                                               interpolant=interpolant)
#         psf_size = psf_size // 2 * 2

        # Draw medium bright stars with galsim in Fourier space
        psf_star = (1-frac) * psf_c + frac * psf_e               
        
        if stars.n_medbright > 0:
            if (not parallel) | (parallel_enabled==False):
                # Draw in serial
                for k in range(stars.n_medbright):
                    draw_star(k,
                              star_pos=stars.star_pos_medbright,
                              Flux=stars.Flux_medbright,
                              psf_star=psf_star,
                              psf_size=psf_size,
                              full_image=full_image)

            else:
                # Draw in parallel, automatically back to serial computing if too few jobs
                p_get_stamp_bounds = partial(get_stamp_bounds,
                                             star_pos=stars.star_pos_medbright,
                                             Flux=stars.Flux_medbright,
                                             psf_star=psf_star,
                                             psf_size=psf_size,
                                             full_image=full_image)

                results = parallel_compute(np.arange(stars.n_medbright), p_get_stamp_bounds,
                                           lengthy_computation=False, verbose=False)

                for (stamp, bounds) in results:
                    full_image[bounds] += stamp[bounds]
                
    if draw_real:
        # Draw very bright star in real space (high cost in convolution)
        # Note origin of star_pos in SE is (1,1) but (0,0) in python
        image_gs = full_image.array
        
        if brightest_only:
            # Only plot the aureole. A Deeper mask is required.
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright-1,
                                                           I0=I0_verybright)
        else:
            # Plot core + aureole. 
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright-1,
                                                           Flux=frac * stars.Flux_verybright)
            if draw_core:
                func_core_2d_s = psf.draw_core2D_in_real(stars.star_pos_verybright,
                                                         Flux=(1-frac) * stars.Flux_verybright)
                image_gs += np.sum([f2d(xx,yy) for f2d in func_core_2d_s], axis=0)
        
        image_aureole = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)

        image = image_gs + image_aureole
        
    else:
        # Draw very bright star in real space
        psf_e_2, psf_size_2 = psf.generate_aureole(contrast=contrast[1],
                                                   psf_scale=psf_scale,
                                                   psf_range=psf_range[1],
                                                   min_psf_range=min_psf_range,
                                                   max_psf_range=max_psf_range,
                                                   interpolant=interpolant)
#         psf_size_2 = psf_size_2 // 2 * 2
        
        psf_star_2 = (1-frac) * psf_c + frac * psf_e_2
        
        for k in range(stars.n_verybright):
            draw_star(k,
                      star_pos=stars.star_pos_verybright,
                      Flux=stars.Flux_verybright,
                      psf_star=psf_star_2,
                      psf_size=psf_size_2,
                      full_image=full_image)
            
        image = full_image.array
                   
    return image


def generate_image_fit(psf_fit, stars, image_shape, norm='brightness',
                       brightest_only=False, draw_real=True, leg2d=False):
    """ Generate the fitted bright stars, the fitted background and
        a noise images (for display only). """
    
    nY, nX = image_shape
    yy, xx = np.mgrid[:nY, :nX]
    
    if norm=='brightness':
        draw_func = generate_image_by_znorm
    elif norm=='flux':
        draw_func = generate_image_by_flux
        
    if stars.n_verybright==0: subtract_external = False
    else: subtract_external = True
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image_stars = draw_func(psf_fit, stars, xx, yy,
                               psf_range=[900, max(image_shape)],
                               psf_scale=psf_fit.pixel_scale,
                               brightest_only=brightest_only,
                               subtract_external=subtract_external,
                               draw_real=draw_real)
                               
    if hasattr(psf_fit, 'bkg_std') & hasattr(psf_fit, 'bkg'):
        image_stars_noise = add_image_noise(image_stars, psf_fit.bkg_std, verbose=False)
        noise_image = image_stars_noise - image_stars
        bkg_image = psf_fit.bkg * np.ones((nY, nX))
        print("Fitted Background : %.2f +/- %.2f"%(psf_fit.bkg, psf_fit.bkg_std))
    else:
        noise_image = bkg_image = None
   
    if leg2d:
        Xgrid = np.linspace(-(1-1/nX)/2., (1-1/nX)/2., nX)
        Ygrid = np.linspace(-(1-1/nY)/2., (1-1/nY)/2., nY)
        H10 = leggrid2d(Xgrid, Ygrid, c=[[0,1],[0,0]])
        H01 = leggrid2d(Xgrid, Ygrid, c=[[0,0],[1,0]])

        bkg_image += psf_fit.A10 * H10 + psf_fit.A01 * H01

    return image_stars, noise_image, bkg_image


############################################
# Priors and Likelihood Models for Fitting
############################################

def set_prior(n_est, mu_est, std_est, n_spline=2,
              n_min=1, d_n0=0.1, d_n=0.2, std_min=3,
              theta_in=50, theta_out=300, leg2d=False,
              fix_n0=False, fix_theta=False,
              fit_sigma=True, fit_frac=False):
    
    """
    Setup prior transforms for models. 
    
    Parameters
    ----------
    n_est : estimate of the first power-law index, i.e. from profile fitting
    mu_est : estimate of sky background level, from either the global DF reduction pipeline or a local sigma-clipped mean after aggresive mask
    std_est : esimtate of sky uncertainty, from a local sigma-clipped stddev after aggresive mask
    
    n_spline : number of power-law component for modeling the aureole
    n_min : minium power index allowed in fitting
    d_n0 : stddev of noraml prior of n_0
    d_n : minimum length of prior jump in n_k for n_spline>=3, default 0.2
    theta_in : inner boundary of the first transition radius
    theta_out : outer boundary of the first transition radius
    leg2d : whether a legendre polynomial background will be fit
    std_min : estimated (expected to be poisson) noise as minimum noise
    fit_frac : whether the aureole fraction will be fit
    fit_sigma : whether the sky uncertainty will be fit
    
    Returns
    ----------
    prior_tf : prior transform function for fitting
    
    """
    
    log_t_in = np.log10(theta_in)
    log_t_out = np.log10(theta_out)
    Dlog_t = log_t_out - log_t_in
    log_t_s = np.logspace(log_t_in, log_t_out, n_spline+1)[1:-1]
    #used if fix_theta=True
    
    Prior_mu = stats.truncnorm(a=-3, b=1., loc=mu_est, scale=std_est)  # mu : N(mu_est, std_est)
    
    # counting helper for # of parameters
    K = 0
    if fit_frac: K += 1
    if fit_sigma: K += 1
    
    Prior_logsigma = stats.truncnorm(a=-3, b=0,
                                     loc=np.log10(std_est), scale=0.3)
                                     
    Prior_logfrac = stats.uniform(loc=-2.5, scale=2.2)
    
    if n_spline == 'm':
        Prior_gamma = stats.uniform(loc=0., scale=10.)       
        Prior_beta = stats.uniform(loc=1.1, scale=6.)
        
        def prior_tf_mof(u):
            v = u.copy()
            v[0] = Prior_gamma.ppf(u[0])             # gamma1
            v[1] = Prior_beta.ppf(u[1])              # beta1
            
            v[-K-1] = Prior_mu.ppf(u[-K-1])          # mu
            
            if fit_sigma:
                v[-K] = Prior_logsigma.ppf(u[-K])    # log sigma
                leg_level = v[-K]
            else:
                leg_level = 0.5
                
            if leg2d:
                v[-K-2] = stats.uniform.ppf(u[-K-2], 
                                            loc=leg_level-1.3, scale=1.3)  # log A10
                v[-K-3] = stats.uniform.ppf(u[-K-3],
                                            loc=leg_level-1.3, scale=1.3)  # log A01
            if fit_frac:
                v[-1] = Prior_logfrac.ppf(u[-1])       # log frac
            return v
        
        return prior_tf_mof
    
    else:
        Prior_n0 = stats.truncnorm(a=-3, b=3., loc=n_est, scale=d_n0)
        # n0 : N(n, d_n0)
        Prior_logtheta1 = stats.uniform(loc=log_t_in, scale=Dlog_t)
        # log theta1 : log t_in - log t_out  arcsec
        
        if n_spline==2:
            def prior_tf_2p(u):
                v = u.copy()
                
                if fix_n0:
#                    v[0] = n_est
                    v[0] = np.random.normal(n_est, d_n0)
                else:
                    v[0] = Prior_n0.ppf(u[0])
                    
                v[1] = u[1] * (v[0]- d_n0 - n_min) + n_min        # n1 : n_min - (n0-d_n0)
                v[2] = Prior_logtheta1.ppf(u[2])

                v[-K-1] = Prior_mu.ppf(u[-K-1])          # mu

                if fit_sigma:
                    v[-K] = Prior_logsigma.ppf(u[-K])       # log sigma
                    leg_amp = v[-K]
                else:
                    leg_amp = 0.5

                if leg2d:
                    v[-K-2] = stats.uniform.ppf(u[-K-2], 
                                                loc=leg_amp-1.3, scale=1.3)  # log A10
                    v[-K-3] = stats.uniform.ppf(u[-K-3],
                                                loc=leg_amp-1.3, scale=1.3)  # log A01
                if fit_frac:
                    v[-1] = Prior_logfrac.ppf(u[-1])       # log frac

                return v
            
            return prior_tf_2p

        else:
            Priors = [Prior_n0, Prior_logtheta1,
                      Prior_mu, Prior_logsigma, Prior_logfrac]
            prior_tf = partial(prior_tf_sp, Priors=Priors, n_spline=n_spline,
                               n_min=n_min, n_est=n_est, d_n0=d_n0, d_n=d_n,
                               log_t_s=log_t_s, log_t_out=log_t_out,
                               K=K, fix_n0=fix_n0, leg2d=leg2d,
                               fit_sigma=fit_sigma, fit_frac=fit_frac)

            return prior_tf


def prior_tf_sp(u, Priors, n_spline=3,
                d_n=0.2, n_min=1, n_max=4, n_est=3.3, d_n0=0.2,
                leg2d=False, fix_n0=False, flexible=False,
                fix_theta=False, log_t_s=[90, 180], log_t_out=300,
                K=1, fit_sigma=True, fit_frac=False):
                
    """ Prior Transform function for n_spline """
    
    # loglikehood vector
    v = u.copy()
    
    # read priors
    Prior_n0, Prior_logtheta1, Prior_mu, Prior_logsigma, Prior_logfrac = Priors
    
    # n prior
    if fix_n0:
        v[0] = n_est
#        v[0] = np.random.normal(n_est, d_n0)
    else:
        v[0] = Prior_n0.ppf(u[0])
    
    if flexible:
        for k in range(n_spline-2):
            v[k+1] = u[k+1] * max(-2.+d_n, n_min-v[k]+d_n) + (v[k]-d_n)
            # n_k+1 : max{n_min, n_k-2} - n_k-d_n
        
        v[k+2] = u[k+2] * min(n_max-(v[k+1]-d_n), n_max-n_min) + max(n_min, v[k+1]-d_n)
        # n_last : max(n_min, n_k-d_n) - n_max
        
    else:
        for k in range(n_spline-1):
            v[k+1] = u[k+1] * max(-2.+d_n, n_min-v[k]+d_n) + (v[k]-d_n)
            # n_k+1 : max{n_min, n_k-2} - n_k-d_n
    
    # theta prior
    if fix_theta:
        v[n_spline:2*n_spline-1] = log_t_s
    else:
        v[n_spline] = Prior_logtheta1.ppf(u[n_spline])
        # log theta1 : t_in-t_out  arcsec

        for k in range(n_spline-2):
            v[k+n_spline+1] = u[k+n_spline+1] * \
                                (log_t_out - v[k+n_spline]) + v[k+n_spline]
            # log theta_k+1: theta_k - t_out  # in arcsec

    # background prior
    v[-K-1] = Prior_mu.ppf(u[-K-1])          # mu
    if fit_sigma:
        v[-K] = Prior_logsigma.ppf(u[-K])       # log sigma
        leg_amp = v[-K]
    else:
        leg_amp = 0.5

    if leg2d:
        v[-K-2] = stats.uniform.ppf(u[-K-2],
                                    loc=leg_amp-1.3, scale=1.3)  # log A10
        v[-K-3] = stats.uniform.ppf(u[-K-3],
                                    loc=leg_amp-1.3, scale=1.3)  # log A01
    
    # frac prior
    if fit_frac:
        v[-1] = Prior_logfrac.ppf(u[-1])       # log frac

    return v

        
def draw_proposal(draw_func,
                  proposal,
                  psf, stars,
                  K=1, leg=None):
    
    # Draw image and calculate log-likelihood
    # K : position order of background in the proposal (dafault -2)
    mu = proposal[-K-1] 
    
    image_tri = draw_func(psf, stars)
    image_tri += mu
        
    if leg is not None:
        A10, A01 = 10**proposal[-K-2], 10**proposal[-K-3]
        H10, H01 = leg.coefs
        
        image_tri += A10 * H10 + A01 * H01

    return image_tri 
    
def calculate_likelihood(ypred, data, sigma):
    # Calculate log-likelihood
    residsq = (ypred - data)**2 / sigma**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))

    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike
    
    
class Legendre2D:
    """
    Legendre 2D coefficients
    
    """
    
    def __init__(self, image_shape, K=0, order=1):
        self.image_shape = image_shape
        nY, nX = image_shape
        
        #x_grid = y_grid = np.linspace(0,image_size-1, image_size)
        #self.x_grid = x_grid
        #self.y_grid = y_grid
        #self.cen = ((Ximage_size-1)/2., (Yimage_size-1)/2.)
        
        X_grid = np.linspace(-(1-1/nX)/2., (1-1/nX)/2., nX)
        Y_grid = np.linspace(-(1-1/nY)/2., (1-1/nY)/2., nY)
        self.X_grid = X_grid
        self.Y_grid = Y_grid
        
        if order == 1:
            H10 = leggrid2d(X_grid, Y_grid, c=[[0,1],[0,0]])
            H01 = leggrid2d(X_grid, Y_grid, c=[[0,0],[1,0]])
            self.coefs = [H10, H01]
            

def set_likelihood(image, mask_fit, psf, stars,
                   norm='brightness', n_spline=2,
                   n0=3.3, fix_n0=False,
                   psf_range=[None,None], leg2d=False,
                   std_est=None, G_eff=1e5,
                   fit_sigma=True, fit_frac=False,
                   brightest_only=False, parallel=False, draw_real=False):
    
    """
    Setup likelihood function.
    
    Parameters
    ----------
    image: 2d image data to be fit
    mask_fit: mask map (masked region is 1)
    psf: A PSF class to be updated
    stars: Stars class for the modeling
    
    Returns
    ----------
    loglike : log-likelihood function for fitting
    
    """
    
    data = image[~mask_fit].copy().ravel()
    
    image_shape = image.shape
    nY, nX = image_shape
    yy, xx = np.mgrid[:nY, :nX]
    
    stars_0 = stars.copy()
    z_norm = stars_0.z_norm.copy()
    
    if norm=='brightness':
        draw_func = generate_image_by_znorm
        
    elif norm=='flux':
        draw_func = generate_image_by_flux
    
    if (psf.aureole_model!='moffat') & (stars.n_verybright > 0) & (norm=='brightness'):
        subtract_external = True
    else:
        subtract_external = False
    
    p_draw_func = partial(draw_func, xx=xx, yy=yy,
                          psf_range=psf_range,
                          psf_scale=psf.pixel_scale,
                          max_psf_range=psf.theta_out,
                          brightest_only=brightest_only,
                          subtract_external=subtract_external,
                          parallel=parallel, draw_real=draw_real)
        
    # K : position order of background in the proposal (dafault -2)
    K = 0
    if fit_frac: K += 1
    if fit_sigma: K += 1
    
    # 1st-order Legendre Polynomial
    if leg2d:
        leg = Legendre2D(image_shape, order=1)
        H10, H01 = leg.coefs
    else:
        leg = None
        H10, H01 = 0, 0
        
    if n_spline == 'm':
        
        def loglike_mof(v):

            gamma1, beta1 = v[:2]
            mu = v[-K-1]
            
            if fit_sigma:
                sigma = 10**v[-K]

            param_update = {'gamma1':gamma1, 'beta1':beta1}

            if fit_frac:
                frac = 10**v[-1]
                param_update['frac'] = frac

            psf.update(param_update)

            if norm=='brightness':
                # I varies with sky background
                stars.z_norm = z_norm + (stars.BKG - mu)

            image_tri = p_draw_func(psf, stars)
            image_tri += mu
            
            ypred = image_tri[~mask_fit].ravel()

            loglike = calculate_likelihood(ypred, data, sigma)
            
            return loglike

        return loglike_mof
        
    else:
        cutoff = psf.cutoff    # whether to cutoff
        theta_0 = psf.theta_0  # inner flattening
        theta_c = psf.theta_c  # outer cutoff
        n_c = psf.n_c
        
        if n_spline==2:

            def loglike_2p(v):
            
                n_s = v[:2]
                
                ### Below is new!
                if fix_n0:
                    n_s[0] = n0
                ###
                
                theta_s = [theta_0, 10**v[2]]
                if cutoff:
                    n_s = np.append(n_s, n_c)
                    theta_s = np.append(theta_s, theta_c)
                    
                mu = v[-K-1]
                if not np.all(theta_s[1:] > theta_s[:-1]):
                    loglike = -1e100
                    return loglike

                param_update = {'n_s':n_s, 'theta_s':theta_s}

                if fit_frac:
                    frac = 10**v[-1]
                    param_update['frac'] = frac

                psf.update(param_update)
                psf.update({'n_s':n_s, 'theta_s':theta_s})

                if norm=='brightness':
                    # I varies with sky background
                    stars.z_norm = z_norm + (stars.BKG - mu)

                image_tri = p_draw_func(psf, stars)

                image_tri += mu

                if leg2d:
                    A10, A01 = 10**v[-K-2], 10**v[-K-3]
                    bkg_leg = A10 * H10 + A01 * H01
                    image_tri += bkg_leg
                    
                ypred = image_tri[~mask_fit].ravel()
                
                if fit_sigma:
                    # sigma = 10**v[-K]
                    sigma = np.sqrt((10**v[-K])**2+(ypred-mu)/G_eff)
                else:
                    #sigma = std_est
                    sigma = np.sqrt(std_est**2+(ypred-mu)/G_eff)

                loglike = calculate_likelihood(ypred, data, sigma)

                return loglike

            return loglike_2p        


        elif n_spline==3:

            def loglike_3p(v):

                n_s = v[:3]
                
#                ### Below is new!
#                if fix_n0:
#                    n_s[0] = n0
#                ###
                
                theta_s = [theta_0, 10**v[3], 10**v[4]]
                
                if cutoff:
                    n_s = np.append(n_s, n_c)
                    theta_s = np.append(theta_s, theta_c)
                    
                if not np.all(theta_s[1:] > theta_s[:-1]):
                    loglike = -1e100
                    return loglike
                    
                mu = v[-K-1]

                param_update = {'n_s':n_s, 'theta_s':theta_s}

                if fit_frac:
                    frac = 10**v[-1]  
                    param_update['frac'] = frac

                psf.update(param_update)

                if norm=='brightness':
                    # I varies with sky background
                    stars.z_norm = z_norm + (stars.BKG - mu)

                image_tri = p_draw_func(psf, stars)
                image_tri += mu

                if leg2d:
                    A10, A01 = 10**v[-K-2], 10**v[-K-3]
                    image_tri += A10 * H10 + A01 * H01
        
                ypred = image_tri[~mask_fit].ravel()
                
                if fit_sigma:
                    #sigma = 10**v[-K]
                    sigma = np.sqrt((10**v[-K])**2+(ypred-mu)/G_eff)
                else:
                    #sigma = std_est
                    sigma = np.sqrt(std_est**2+(ypred-mu)/G_eff)

                loglike = calculate_likelihood(ypred, data, sigma)
                
                return loglike

            return loglike_3p

        else:

            def loglike_sp(v):
                
                n_s = v[:n_spline]
                
                ### Below is new!
                if fix_n0:
                    n_s[0] = n0
                ###
                
                theta_s = np.append(theta_0, 10**v[n_spline:2*n_spline-1])
                
                if cutoff:
                    n_s = np.append(n_s, n_c)
                    theta_s = np.append(theta_s, theta_c)
                    
                if not np.all(theta_s[1:] > theta_s[:-1]):
                    loglike = -1e100
                    return loglike
                
                mu = v[-K-1]
                
                param_update = {'n_s':n_s, 'theta_s':theta_s}

                if fit_frac:
                    frac = 10**v[-1]
                    param_update['frac'] = frac

                psf.update(param_update)
                
                if norm=='brightness':
                    # I varies with sky background
                    stars.z_norm = z_norm + (stars.BKG - mu)

                image_tri = draw_proposal(p_draw_func, v,
                                          psf, stars,
                                          K=K, leg=leg)
                
                ypred = image_tri[~mask_fit].ravel()
                
                if fit_sigma:
                    #sigma = 10**v[-K]
                    sigma = np.sqrt((10**v[-K])**2+(ypred-mu)/G_eff)
                else:
                    #sigma = std_est
                    sigma = np.sqrt(std_est**2+(ypred-mu)/G_eff)
                
                loglike = calculate_likelihood(ypred, data, sigma)

                return loglike
            
            return loglike_sp
