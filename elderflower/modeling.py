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

import warnings

try:
    import galsim
    from galsim import GalSimBoundsError
    galsim_installed = True
except ImportError:
    warnings.warn("Galsim is not installed. Convolution-based star rendering is not enabled.")
    galsim_installed = False
    

from copy import deepcopy
from numpy.polynomial.legendre import leggrid2d
from itertools import combinations
from functools import partial, lru_cache

try: 
    from .parallel import parallel_compute
    parallel_enabled = True
except ImportError:
    warnings.warn("Joblib / multiprocessing is not installed. Parallelization is not enabled.")
    parallel_enabled = False
    
from .numeric import *
from .io import logger
from .utils import Intensity2SB, SB2Intensity
from .utils import round_good_fft, calculate_psf_size
from .utils import NormalizationError
from . import DF_pixel_scale

############################################
# Functions for making PSF models
############################################

class PSF_Model:
    """ A PSF Model object """
    
    def __init__(self, params=None,
                 core_model='moffat',
                 aureole_model='multi-power'):
        """
        Parameters
        ----------
        params : a dictionary containing keywords of PSF parameter
        core_model : model of PSF core (moffat)
        aureole_model : model of aureole ("power" or "multi-power")    
        
        """
        self.core_model = core_model
        self.aureole_model = aureole_model
        
        self.cutoff = True   # cutoff by default
        
        # Build attribute for parameters from dictionary keys 
        for key, val in params.items():
            if type(val) is list:
                params[key] = val = np.array(val)
            exec('self.' + key + ' = val')
            
        self.params = params
        
        if hasattr(self, 'fwhm'):
            self.gamma = fwhm_to_gamma(self.fwhm, self.beta)
            self.params['gamma'] = self.gamma
        elif hasattr(self, 'gamma'):
            self.fwhm  = gamma_to_fwhm(self.gamma, self.beta)
            self.params['fwhm'] = self.fwhm
        else:
            logger.error('Either fwhm or gamma needs to be given.')
        
        if galsim_installed:
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
            
        if 'fwhm' in params.keys():
            self.gamma = fwhm_to_gamma(self.fwhm, self.beta)
            self.params['gamma'] = self.gamma
        elif 'gamma' in params.keys():
            self.fwhm  = gamma_to_fwhm(self.gamma, self.beta)
            self.params['fwhm'] = self.fwhm
        else:
            pass
            
        self.pixelize(pixel_scale)
                
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
            if kwargs.get("xunit") == "arcsec":
                vline_pos = self.theta_s
            else:
                vline_pos = self.theta_s_pix
            for pos in vline_pos:
                plt.axvline(pos, ls="--", color="k", alpha=0.3, zorder=0)
                
    def generate_core(self):
        """ Generate Galsim PSF of core. """
        gamma, beta = self.gamma, self.beta
        self.fwhm = fwhm = gamma * 2. * math.sqrt(2**(1./beta)-1)
        
        if galsim_installed:
            psf_core = galsim.Moffat(beta=beta, fwhm=fwhm,
                                     flux=1., gsparams=self.gsparams) # in arcsec
            self.psf_core = psf_core
        else:
            psf_core = None
            
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
        if galsim_installed == False:
            psf_aureole = None
            psf_size = None
            
        else:
            from galsim import Moffat, ImageF, InterpolatedImage
            
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
                psf_aureole = Moffat(beta=beta1, scale_radius=gamma1,
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
                image_psf = ImageF(psf_model)
                psf_aureole = InterpolatedImage(image_psf, flux=1,
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
            verybright = stars.verybright[stars.bright]
            
            for i in range(n_iter):
                z_norm_verybright = z_norm_verybright0 - I_ext[verybright]
                z_norm_verybright[z_norm_verybright<0] = 0
                
                I0_verybright = self.I2I0(z_norm_verybright, r=r_scale)
                I_ext = cal_ext_light(I0_source=I0_verybright)
            
        return I_ext
    
    def I2Flux(self, I, r):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return I2Flux_mof(self.frac, self.gamma1_pix, self.beta1, r, I=I)
        
        elif self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n0, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2Flux_mpow(self.frac, self.n_s, self.theta_s_pix, r, I=I)
        
    def Flux2I(self, Flux, r):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return Flux2I_mof(self.frac, self.gamma1_pix, self.beta1, r, Flux=Flux)
        
        elif self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n0, self.theta_0_pix, r, Flux=Flux)
        
        elif self.aureole_model == "multi-power":
            return Flux2I_mpow(self.frac, self.n_s, self.theta_s_pix, r,  Flux=Flux)
        
    def SB2Flux(self, SB, BKG, ZP, r):
        """ Convert suface brightness SB at r to total flux, given background value and ZP. """
        # Intensity = I + BKG
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale) - BKG
        return self.I2Flux(I, r)
    
    def Flux2SB(self, Flux, BKG, ZP, r):
        """ Convert total flux to suface brightness SB at r, given background value and ZP. """
        I = self.Flux2I(Flux, r)
        return Intensity2SB(I+ BKG, BKG, ZP, self.pixel_scale)
    
    @property
    def psf_star(self):
        """ Galsim object of star psf (core+aureole) """
        if galsim_installed:
            frac = self.frac
            psf_core, psf_aureole = self.psf_core, self.psf_aureole
            return (1-frac) * psf_core + frac * psf_aureole
        else:
            return None

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
                raise NormalizationError("Both Flux and I0 are not given.")
                
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
                raise NormalizationError("Both Flux and I0 are not given.")
            
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
                raise NormalizationError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      multi_power2d(xx, yy, cen=cen,
                                                    n_s=n_s, theta_s=theta_s_pix,
                                                    I_theta0=I)
                                      for (I, pos) in zip(I_theta0, star_pos)])
            
        return f_aureole_2d_s
        
    def fit_psf_core_1D(self, image_psf, **kwargs):
        """ Fit the core parameters from 1D profiles of the input 2D PSF. """
        from .utils import fit_psf_core_1D
        params0 = {"fwhm":self.fwhm,
                   "beta":self.beta,
                   "frac":self.frac,
                   "n_s":self.n_s,
                   "theta_s":self.theta_s}
        frac, beta = fit_psf_core_1D(image_psf,
                                     params0=params0,
                                     pixel_scale=self.pixel_scale,
                                     **kwargs)
        self.frac = max(1e-7, min(frac,1.0))
        self.beta = beta
        self.update({"frac":frac, "beta":beta})

    
class Stars:
    """
    Class storing positions & flux of faint/medium-bright/bright stars
    
    """
    def __init__(self, star_pos, Flux, 
                 Flux_threshold=[2.7e5, 2.7e6],
                 z_norm=None, r_scale=12, BKG=0):
        """
        Parameters
        ----------
        star_pos: 2d array
            pixel positions of stars in the region
        Flux: 1d array
            flux of stars (in ADU)
        Flux_threshold : [float, float]
            thereshold of flux [MB, VB]
            (default: corresponding to [13.5, 11] mag for DF)
        z_norm : 1d array
            flux scaling measured at r_scale
        r_scale : int
            radius at which to measure the flux scaling
        BKG : float
            sky background value
                
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
    
    def __str__(self):
        return "A Star Class"

    def __repr__(self):
        return ' N='.join([f"{self.__class__.__name__}", str(self.n_tot)])

    @classmethod            
    def from_znorm(cls, psf, star_pos, z_norm,
                   z_threshold=[10, 300], r_scale=12):
        """ Star object built from intensity at r_scale instead of flux. """
        Flux = psf.I2Flux(z_norm, r=r_scale)
        Flux_threshold = psf.I2Flux(z_threshold, r=r_scale)
        
        return cls(star_pos, Flux, Flux_threshold,
                   z_norm=z_norm, r_scale=r_scale)
    
    def update_Flux(self, Flux):
        self.Flux = Flux
    
    def _info(self):
        Flux = self.Flux
        if len(Flux[self.medbright])>0:
            msg = "# of medium bright stars : {0} ".format(self.n_medbright)
            msg += "(flux range:{0:.2g}~{1:.2g})".format(Flux[self.medbright].min(), Flux[self.medbright].max())
            logger.info(msg)
        
        if len(Flux[self.verybright])>0:
            msg = "# of very bright stars : {0} ".format(self.n_verybright)
            msg += "(flux range:{0:.2g}~{1:.2g})".format(Flux[self.verybright].min(), Flux[self.verybright].max())
            logger.info(msg)
            
        # Rendering stars in parallel if number of bright stars exceeds 50
        if self.n_medbright < 50:
            msg = "Not many bright stars. Recommend to draw in serial."
            logger.debug(msg)
            self.parallel = False
        else:
            msg = "Crowded fields w/ bright stars > 50.  Recommend to allow parallel."
            logger.debug(msg)
            self.parallel = True
            
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
        logger.info("Only model brightest stars in the field.")
            
        stars_vb = Stars(self.star_pos_verybright,
                         self.Flux_verybright,
                         Flux_threshold=self.Flux_threshold,
                         z_norm=self.z_norm_verybright,
                         r_scale=self.r_scale, BKG=self.BKG)
        
        return stars_vb
    
    def remove_outsider(self, image_shape, gap=[36,12]):
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
                          r_scale=self.r_scale, BKG=self.BKG)
        return stars_new
     
    def save(self, name='stars', save_dir='./'):
        from .io import save_pickle
        save_pickle(self, os.path.join(save_dir, name+'.pkl'), 'Star model')
        


### 2D functions ###

@lru_cache(maxsize=16)
def generate_psf_grid(psf_size):
    """ Generate Grid of PSF and plot PSF model in real space onto it """
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


@njit
def multi_power2d(xx, yy, n_s, theta_s, I_theta0, cen, clear=False):
    """ Multi-power law for 2d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2).ravel()
    z = np.zeros(xx.size) 
    theta0 = theta_s[0]
    z[rr<=theta0] = I_theta0
    if clear:
        z[rr<=theta0] = 0
    
    for k in range(len(a_s)):
        reg = (rr>theta_s[k]) & (rr<=theta_s[k+1]) if k<len(a_s)-1 else (rr>theta_s[k])     
        z[reg] = a_s[k] * np.power(rr[reg], -n_s[k])
        
    return z.reshape(xx.shape)


############################################
# Functions for PSF rendering with Galsim
############################################

def get_center_offset(pos):
    """ Shift center for the purpose of accuracy (by default galsim round to integer!)
        Originally should be x_pos, y_pos = pos + 1 (ref galsim demo)
        But origin of star_pos in SE is (1,1) but (0,0) in python """
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
    # Function of drawing, devised to facilitate parallelization.
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

def add_image_noise(image, noise_std, random_seed=42):
    """ Add Gaussian noise image """
    if galsim_installed == False:
        logger.warning("Galsim is not installed. Function disabled.")
        return np.zeros_like(image)
    else:
        from galsim import ImageF, BaseDeviate, GaussianNoise
        
    logger.debug("Generate noise background w/ stddev = %.3g"%noise_std)
    
    Image = galsim.ImageF(image)
    rng = galsim.BaseDeviate(random_seed)
    
    gauss_noise = galsim.GaussianNoise(rng, sigma=noise_std)
    Image.addNoise(gauss_noise)
    
    return Image.array


def make_base_image(image_shape, stars, psf_base, pad=50, psf_size=64, verbose=False):
    """ Background images composed of dim stars with fixed PSF psf_base"""
    
    if galsim_installed:
        from galsim import ImageF
    else:
        return np.zeros(image_shape)
        
    if verbose:
        logger.info("Generate base image of faint stars (flux < %.2g)."%(stars.F_bright))
    
    start = time.time()
    nX0 = image_shape[1] + 2 * pad
    nY0 = image_shape[0] + 2 * pad
    full_image0 = ImageF(nX0, nY0)
    
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
            logger.debug("GalSim reported a GalSimBoundsError")
            if verbose:
                print(e.__doc__)
                print(e.message)
            continue

    image_base0 = full_image0.array
    
    end = time.time()
    if verbose: logger.info("Total Time: %.3f s\n"%(end-start))
    
    image_base = image_base0[pad:nY0-pad, pad:nX0-pad]
    
    return image_base


def make_truth_image(psf, stars, image_shape, contrast=1e6,
                     parallel=False, verbose=False, saturation=4.5e4):
    
    """
    Draw a truth image according to the given psf, position & flux.
    In two manners: 1) convolution in FFT w/ Galsim;
                and 2) plot in real space w/ astropy model. 

    """
    
    if galsim_installed == False:
        raise Exception("Galsim is not installed. Function disabled.")
    else:
        from galsim import ImageF
    
    if verbose:
        logger.info("Generate the truth image.")
        start = time.time()
    
    # attributes
    frac = psf.frac
    gamma_pix = psf.gamma_pix
    beta = psf.beta
    
    nY, nX = image_shape
    yy, xx = np.mgrid[:nY, :nX]
    
    psf_core = psf.psf_core
        
    psf_aureole = psf.psf_aureole
    
    full_image = ImageF(nX, nY)
    
    Flux_A = stars.Flux_bright
    star_pos_A = stars.star_pos_bright
    
    image_gs = full_image.array

    # Draw bright stars in real space
    func_core_2d_s = psf.draw_core2D_in_real(star_pos_A, (1-frac) * Flux_A)
    func_aureole_2d_s = psf.draw_aureole2D_in_real(star_pos_A, frac * Flux_A)

    image = np.sum([f2d(xx,yy) + p2d(xx,yy)
                     for (f2d, p2d) in zip(func_core_2d_s,
                                           func_aureole_2d_s)], axis=0)
                                               
    # combine the two image
    image += image_gs
    
    # saturation limit
    image[image>saturation] = saturation
        
    if verbose: 
        end = time.time()
        logger.info("Total Time: %.3f s\n"%(end-start))
    
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
        z_norm[z_norm<=0] = np.abs(z_norm).min() # problematic negatives
        Flux = psf.I2Flux(z_norm, r=r_scale)
        stars.update_Flux(Flux) 
        
    # Setup the canvas
    if galsim_installed == False:
        brightest_only = True
        draw_real = True
        full_image = np.empty((nY, nX), dtype=np.float32)
    else:
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
        
        if brightest_only:
            image_gs = 0.  # no galsim image
            # Only plot the aureole. A heavy mask is required.
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright-1,
                                                           I0=I0_verybright)
        else:
            image_gs = full_image.array
            # Plot core + aureole.
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright-1,
                                                           Flux=frac * stars.Flux_verybright)
            if draw_core:
                func_core_2d_s = psf.draw_core2D_in_real(stars.star_pos_verybright-1,
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
                       brightest_only=False, draw_real=True,
                       subtract_external=False, leg2d=False):
    """ Generate the fitted bright stars, the fitted background and
        a noise images (for display only). """
    
    nY, nX = image_shape
    yy, xx = np.mgrid[:nY, :nX]
    
    stars_ = stars.copy()
    
    if norm=='brightness':
        draw_func = generate_image_by_znorm
    elif norm=='flux':
        draw_func = generate_image_by_flux
        
    if stars_.n_verybright==0:
        subtract_external = False
    
    pixel_scale = psf_fit.pixel_scale
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image_stars = draw_func(psf_fit, stars_, xx, yy,
                               psf_range=[900, max(image_shape)*pixel_scale],
                               psf_scale=pixel_scale,
                               brightest_only=brightest_only,
                               subtract_external=subtract_external,
                               draw_real=draw_real, draw_core=True)
                               
    if hasattr(psf_fit, 'bkg_std') & hasattr(psf_fit, 'bkg'):
        image_stars_noise = add_image_noise(image_stars, psf_fit.bkg_std)
        noise_image = image_stars_noise - image_stars
        bkg_image = psf_fit.bkg * np.ones((nY, nX))
        logger.info("   - Background = %.3g +/- %.3g"%(psf_fit.bkg, psf_fit.bkg_std))
    else:
        noise_image = bkg_image = np.zeros_like(image_stars)
   
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
              n_min=1.2, d_n0=0.1, d_n=0.2, std_min=3,
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
    
    Prior_logsigma = stats.truncnorm(a=-3, b=1,
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
        Prior_n0 = stats.norm(loc=n_est, scale=d_n0)
        # n0 : N(n, d_n0)
        Prior_logtheta1 = stats.uniform(loc=log_t_in, scale=Dlog_t)
        # log theta1 : log t_in - log t_out  arcsec
        
        if n_spline==2:
            def prior_tf_2p(u):
                v = u.copy()
                
                if fix_n0:
                    v[0] = n_est
                    #v[0] = np.random.normal(n_est, d_n0)
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
                d_n=0.2, n_min=1.2, n_max=3.5, n_est=3.3, d_n0=0.2,
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
        # log theta1 : log t_in - t_out  # in arcsec

        for k in range(n_spline-2):
            v[k+n_spline+1] = u[k+n_spline+1] * \
                                (log_t_out - v[k+n_spline]) + v[k+n_spline]
            # log theta_k+1: log theta_k - log t_out  # in arcsec

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

def build_independent_priors(priors):
    """ Build priors for Bayesian fitting. Priors should has a (scipy-like) ppf class method."""
    def prior_transform(u):
        v = u.copy()
        for i in range(len(u)):
            v[i] = priors[i].ppf(u[i])
        return v
    return prior_transform
        
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
                   fix_n0=False, brightest_only=False,
                   psf_range=[None,None], leg2d=False,
                   std_est=None, G_eff=1e5,
                   fit_sigma=True, fit_frac=False,
                   parallel=False, draw_real=False):
    
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
        n0 = psf.n0
        theta_0 = psf.theta_0  # inner flattening
        cutoff = psf.cutoff    # whether to cutoff
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
                
                if fix_n0:
                    n_s[0] = n0
                
                theta_s = np.array([theta_0, 10**v[3], 10**v[4]])
                
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
                
                if fix_n0:
                    n_s[0] = n0
                
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
