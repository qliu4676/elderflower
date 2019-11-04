import time
import math
import galsim
from galsim import GalSimBoundsError
from utils import *

from usid_processing import parallel_compute
from functools import partial
from astropy.utils import lazyproperty
from copy import deepcopy

############################################
# Functions for making PSF models
############################################
from types import SimpleNamespace    

class PSF_Model:
    def __init__(self,
                 core_model='Moffat',
                 aureole_model='power',
                 params=None):
        
        self.core_model = core_model
        self.aureole_model = aureole_model
        
        self.params = params
        
        # Build attribute for parameters from dictionary keys 
        for key, val in params.items():
            exec('self.' + key + ' = val')
            
        if hasattr(self, 'fwhm'):
            self.gamma = fwhm_to_gamma(self.fwhm, self.beta)
            self.params['gamma'] = self.gamma
            
        if hasattr(self, 'gamma'):
            self.fwhm  = gamma_to_fwhm(self.gamma, self.beta)
            self.params['fwhm'] = self.fwhm
                
    def make_grid(self, image_size, pixel_scale=2.5):
        self.image_size = image_size
        self.yy, self.xx = np.mgrid[:image_size, :image_size]
        self.pixel_scale = pixel_scale
        
        for key, val in self.params.items():
            if (key == 'gamma') | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def update(self, params):
        for key, val in params.items():
            if np.ndim(val) > 0:
                val = np.array(val)
                
            exec('self.' + key + ' = val')
            self.params[key] = val
            
            if 'theta' in key:
                val = val / self.pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def copy(self):
        return deepcopy(self)            
                                
    def plot1D(self, **kwargs):
        """ Plot analytical 1D profile """
        from plotting import plot_PSF_model_1D
        
        # core
        frac, gamma_pix, beta = self.frac, self.gamma_pix, self.beta
        
        c_mof2Dto1D = C_mof2Dto1D(gamma_pix, beta)

        # aureole
        if self.aureole_model == "power":
            n = self.n
            theta_0_pix = self.theta_0_pix
            
            c_aureole_2Dto1D = C_pow2Dto1D(n, theta_0_pix)
            profile = lambda r: trunc_power1d_normed(r, n, theta_0_pix)

        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s_pix = self.theta_s_pix
            
            c_aureole_2Dto1D = C_mpow2Dto1D(n_s, theta_s_pix)
            profile = lambda r: multi_power1d_normed(r, n_s, theta_s_pix)
        
        f_core = lambda r: moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D
        f_aureole = lambda r: profile(r) / c_aureole_2Dto1D
        
        # plot
        plot_PSF_model_1D(frac, f_core, f_aureole, **kwargs)
        
        if self.aureole_model == "multi-power":
            for t in theta_s_pix:
                plt.axvline(t, ls="--", color="k", alpha=0.3, zorder=1)
                
    def generate_core(self, folding_threshold=1.e-10):
        """
        Generate Galsim PSF of core.
        """
        self.fwhm = self.gamma * 2. * math.sqrt(2**(1./self.beta)-1)
        gsparams = galsim.GSParams(folding_threshold=folding_threshold)
        psf_core = galsim.Moffat(beta=self.beta, fwhm=self.fwhm,
                                flux=1., gsparams=gsparams) # in arcsec
        self.psf_core = psf_core
        return psf_core
    
    def generate_aureole(self,
                         contrast=1e6,
                         psf_scale=2,
                         psf_range=None,
                         min_psf_range=30,
                         max_psf_range=600,
                         interpolant="cubic"):
        """
        Generate Galsim PSF of aureole.

        Parameters
        ----------
        contrast: Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given.
        psf_scale: Pixel scale of the PSF, in general <= pixel scale of data. In arcsec/pix.
        psf_range: Range of PSF. In arcsec.
        min_psf_range : Minimum range of PSF. In arcsec.
        max_psf_range : Maximum range of PSF. In arcsec.
        interpolant: Interpolant method in Galsim.
        
        Returns
        ----------
        psf_aureole: power law Galsim PSF, flux normalized to be 1.
        psf_size: Size of PSF used. In pixel.
        """

        
        if self.aureole_model == "power":
            n = self.n
            theta_0 = self.theta_0
            
        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s = self.theta_s
            n = n_s[0]
            theta_0 = theta_s[0]
            
        if psf_range is None:
            a = theta_0**n
            opt_psf_range = int((contrast * a) ** (1./n))
            psf_range = max(min_psf_range, min(opt_psf_range, max_psf_range))

        # full (image) PSF size in pixel
        psf_size = 2 * psf_range // psf_scale

        # Generate Grid of PSF and plot PSF model in real space onto it
        cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
        yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
        
        if self.aureole_model == "power":
            theta_0_pix = theta_0 / psf_scale
            psf_model = trunc_power2d(xx_psf, yy_psf, n, theta_0_pix, I_theta0=1, cen=cen_psf) 
            
        elif self.aureole_model == "multi-power":
            theta_s_pix = theta_s / psf_scale
            psf_model =  multi_power2d(xx_psf, yy_psf, n_s, theta_s_pix, 1, cen=cen_psf) 

        # Parse the image to Galsim PSF model by interpolation
        image_psf = galsim.ImageF(psf_model)
        psf_aureole = galsim.InterpolatedImage(image_psf, flux=1,
                                               scale=psf_scale,
                                               x_interpolant=interpolant,
                                               k_interpolant=interpolant)
        self.psf_aureole = psf_aureole
        return psf_aureole, psf_size   

        
    def Flux2Amp(self, Flux):
        """ Convert Flux to Astropy Moffat Amplitude (pixel unit) """
        Amps = [moffat2d_Flux2Amp(self.gamma_pix, self.beta, Flux=(1-self.frac)*F)
                for F in Flux]
        return np.array(Amps)
    
    def I2Flux(self, I, r=10):
        """ Convert Intensity I(r) at r to total flux with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n, self.theta_0_pix, r, I)
        
    def Flux2I(self, Flux, r=10):
        """ Convert Intensity I(r) at r to total flux with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n, self.theta_0_pix, r, Flux)
    
    def SB2Flux(self, SB, BKG, ZP, r=10):
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale)
        return self.I2Flux(I, r=r)
    
    def Flux2SB(self, Flux, BKG, ZP, r=10):
        I = self.Flux2I(Flux, r=r)
        return Intensity2SB(I, BKG, ZP, self.pixel_scale)
    
    def plot_model_galsim(self, psf_core, psf_aureole,
                          image_size=400, contrast=None):
        """ Plot Galsim 2D model averaged in 1D """
        from plotting import plot_PSF_model_galsim
        plot_PSF_model_galsim(psf_core, psf_aureole, self.params,
                              image_size, self.pixel_scale,
                              contrast=contrast,
                              aureole_model=self.aureole_model)
        
    def draw_aureole2D_in_real(self, star_pos, Flux):
        
        if self.aureole_model == "power":
            n = self.n
            theta_0_pix = self.theta_0_pix
                
            I_theta0_pow = power2d_Flux2Amp(n, theta_0_pix, Flux=1)
            func_aureole_2d_s = np.array([lambda xx, yy, cen=pos, flux=flux:\
                                          trunc_power2d(xx, yy, cen=cen,
                                                        n=n, theta0=theta_0_pix,
                                                        I_theta0=I_theta0_pow * flux)
                                          for (flux, pos) in zip(Flux, star_pos)])

        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s_pix = self.theta_s_pix
            
            I_theta0_mpow = multi_power2d_Flux2Amp(n_s=n_s, theta_s=theta_s_pix, Flux=1)
            func_aureole_2d_s = np.array([lambda xx, yy, cen=pos, flux=flux: \
                                          multi_power2d(xx, yy, cen=cen,
                                                        n_s=n_s, theta_s=theta_s_pix,
                                                        I_theta0=I_theta0_mpow * flux)
                                          for (flux, pos) in zip(Flux, star_pos)])
        return func_aureole_2d_s

    
class Stars:
    """ Class storing positions & flux of faint/medium-bright/bright stars"""
    def __init__(self, star_pos, Flux, 
                 Flux_threshold=[7e4, 2.7e6],
                 z_norm=None, r_scale=10, verbose=False):
        """
        star_pos: positions of stars
        Flux: flux of stars
        Flux_threshold : thereshold of flux
                (default: corresponding to [15, 11] mag for DF)
        """
        self.star_pos = star_pos
        self.Flux = Flux
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
            
        if verbose:
            if len(Flux[self.medbright])>0:
                print("# of medium bright (flux:%.2g~%.2g) stars: %d "\
                      %(Flux[self.medbright].min(),
                        Flux[self.medbright].max(), self.n_medbright))
            
            if len(Flux[self.verybright])>0:
                print("# of very bright (flux>%.2g) stars : %d"\
                      %(Flux[self.verybright].min(), self.n_verybright))
            
            # Rendering stars in parallel if number of bright stars exceeds 50
            if self.n_medbright < 50:
                print("Not many bright stars, will draw in serial.")
                self.parallel = False 
            else: 
                print("Crowded fields w/ bright stars > 50, will draw in parallel.")
                self.parallel = True

    @classmethod            
    def from_znorm(cls, psf, star_pos, z_norm,
                   Flux_threshold=[1e5, 5e6], r_scale=10, 
                   verbose=False):
        
        Flux = psf.I2Flux(z_norm, r_scale)
        
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
            
    def plot_flux_dist(self, ZP=None, **kwargs):
        from plotting import plot_flux_dist
        plot_flux_dist(self.Flux, [self.F_bright, self.F_verybright], **kwargs)
        if ZP is not None:
            ax1 = plt.gca()
            xticks1 = ax1.get_xticks()
            ax2 = ax1.twiny()
            ax2.set_xticks(xticks1)
            ax2.set_xticklabels(np.around(-2.5*xticks1+ZP ,1))
            ax2.set_xbound(ax1.get_xbound())

    def copy(self):
        return deepcopy(self) 
    
    def remove_outsider(self, image_size, d=[12, 24], verbose=False):
        """ Remove some stars far out of field. """
        star_pos = self.star_pos
        Flux = self.Flux

        out_B = (star_pos<-d[0]) | (star_pos>image_size+d[0])
        remove_B = np.logical_or.reduce(out_B, axis=1) & self.medbright

        out_A = (star_pos<-d[1]) | (star_pos>image_size+d[1])
        remove_A = np.logical_or.reduce(out_A, axis=1) & self.verybright

        remove = remove_A | remove_B
        return Stars(star_pos[~remove], Flux[~remove],
                     self.Flux_threshold, self.z_norm[~remove], verbose=True)

# deprecated
def remove_outsider(image_size, star_pos, Flux, Flux_threshold, d=[12, 24]):
    F_bright, F_verybright = Flux_threshold
    
    medbright = (Flux>F_bright) & (Flux<F_verybright)
    out_B = (star_pos<-d[0]) | (star_pos>image_size+d[0])
    remove_B = np.logical_or.reduce(out_B, axis=1) & medbright
    
    verybright = (Flux>=F_verybright)
    out_A = (star_pos<-d[1]) | (star_pos>image_size+d[1])
    remove_A = np.logical_or.reduce(out_A, axis=1) & verybright

    remove = remove_A | remove_B
    return remove

        
        

class Mask:
    """ Maksing of stars"""
    def __init__(self, image, stars, image_size, pad=0, mu=0):
        self.image = image
        self.stars = stars
        self.image_size = image_size
        self.image_size_pad = image_size + 2 * pad
        self.yy, self.xx = np.mgrid[:self.image_size_pad, :self.image_size_pad]
        
        self.pad = pad
        self.mu = mu
        
    def make_mask_map_dual(self, r_core, sn_thre=3,
                           draw=True, **kwargs):
        image = self.image
        stars = self.stars
        pad = self.pad 
            
        # S/N + Core mask
        mask_deep0, seg_deep0, core_region = \
            make_mask_map_dual(image, stars, self.xx, self.yy, pad=pad,
                               r_core=r_core, sn_thre=sn_thre, **kwargs)
        
        
        self.mask_deep = mask_deep0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.seg_deep = seg_deep0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.mask_deep0 = mask_deep0
        self.seg_deep0 = seg_deep0
            
        self.r_core = r_core
        
        # Display mask
        if draw:
            from plotting import draw_mask_map
            draw_mask_map(image, seg_deep0, mask_deep0, stars, pad=pad,
                          r_core=r_core, vmin=self.mu, vmax=self.mu+50)
            
    def make_mask_strip(self, width, n_strip, dist_strip=200,
                        draw=True, clean=True, **kwargs):
        if hasattr(self, 'mask_deep') is False:
            return None
        
        image = self.image
        stars = self.stars
        pad = self.pad 
        
        # Strip + Cross mask
        mask_strip_s, mask_cross_s = \
            make_mask_strip(stars, self.xx, self.yy, pad=pad,
                            width=width, n_strip=n_strip, dist_strip=dist_strip)
        
        mask_strip_all = ~np.logical_or.reduce(mask_strip_s)
        mask_cross_all = ~np.logical_or.reduce(mask_cross_s)
        
        seg_comb0 = self.seg_deep0.copy()
        ma_extra = (mask_strip_all|~mask_cross_all) & (self.seg_deep0==0)
        seg_comb0[ma_extra] = self.seg_deep0.max()+1
        mask_comb0 = (seg_comb0!=0)
        
        self.mask_comb = mask_comb0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.seg_comb = seg_comb0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.mask_comb0 = mask_comb0
        self.seg_comb0 = seg_comb0
        
        # Clean medium bright stars far from bright stars
        if clean:
            clean = clean_lonely_stars(self.xx, self.yy, mask_comb0,
                                       stars.star_pos, pad=pad)
            clean[stars.Flux >= stars.F_verybright] = False
            
            z_norm_clean = stars.z_norm[~clean] if hasattr(stars, 'z_norm') else None
            stars_new = Stars(stars.star_pos[~clean], stars.Flux[~clean],
                              stars.Flux_threshold, z_norm=z_norm_clean,
                              verbose=False)
            self.stars_new = stars_new
            self.clean = clean
        
        # Display mask
        if draw:
            from plotting import draw_mask_map_strip
            draw_mask_map_strip(image, seg_comb0,
                                mask_comb0, stars_new, pad=pad,
                                ma_example=[mask_strip_s[0],
                                            mask_cross_s[0]],
                                r_core=self.r_core, vmin=self.mu, vmax=self.mu+50)
            
        
### (Old) Galsim Modelling Funcs ###
def Generate_PSF_pow_Galsim(n, theta_t=5, psf_scale=2,
                            contrast=1e5, psf_range=None,
                            min_psf_range=30, max_psf_range=600,
                            interpolant="cubic"):
    """
    Generate power law PSF using Galsim.
        
    Parameters
    ----------
    n: Power law index
    theta_t: Inner flattening radius of power law to avoid divergence at the center. In arcsec.

    Returns
    ----------
    psf_pow: power law Galsim PSF, flux normalized to be 1.
    psf_size: Size of PSF used. In pixel.
    """
    
    # Calculate a PSF size with contrast, if not given
    if psf_range is None:
        a = theta_t**n
        opt_psf_range = int((contrast * a) ** (1./n))
        psf_range = max(min_psf_range, min(opt_psf_range, max_psf_range))
    
    # full (image) PSF size in pixel
    psf_size = 2 * psf_range // psf_scale
    
    # Generate Grid of PSF and plot PSF model in real space onto it
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
    
    theta_t_pix = theta_t / psf_scale
    psf_model = trunc_power2d(xx_psf, yy_psf, n, theta_t_pix, I_theta0=1, cen=cen_psf) 

    # Parse the image to Galsim PSF model by interpolation
    image_psf = galsim.ImageF(psf_model)
    psf_pow = galsim.InterpolatedImage(image_psf, flux=1, scale=psf_scale,
                                       x_interpolant=interpolant, k_interpolant=interpolant)
    return psf_pow, psf_size


def Generate_PSF_mpow_Galsim(contrast, n_s, theta_s, 
                             psf_scale=2, psf_range=None,
                             min_psf_range=60, max_psf_range=1200,
                             interpolant="cubic"):
    """
    Generate power law PSF using Galsim.
        
    Parameters
    ----------
    n_s: Power law indexs
    theta_s: Transition radius of power law to avoid divergence at the center. In arcsec.

    Returns
    ----------
    psf_mpow: multi-power law Galsim PSF, flux normalized to be 1.
    psf_size: Size of PSF used. In pixel.
    """
    # Calculate a PSF size with contrast, if not given
    if psf_range is None:
        a_psf = (theta_s[0])**n_s[0]
        opt_psf_range = int((contrast * a_psf) ** (1./n_s[0]))
        psf_range = max(min_psf_range, min(opt_psf_range, max_psf_range))
    
    psf_size = 2 * psf_range // psf_scale
    
    # Generate Grid of PSF and plot PSF model in real space onto it
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
    
    theta_s_psf_pix = theta_s / psf_scale
    psf_model =  multi_power2d(xx_psf, yy_psf, n_s, theta_s_psf_pix, 1, cen=cen_psf) 
    
    # Parse the image to Galsim PSF model by interpolation
    image_psf = galsim.ImageF(psf_model)
    psf_mpow = galsim.InterpolatedImage(image_psf, flux=1, scale=psf_scale,
                                        x_interpolant=interpolant, k_interpolant=interpolant)
    return psf_mpow, psf_size


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

def compute_multi_pow_norm(n_s, theta_s, I_theta0):
    """ Compute normalization factor A of each power law component A_i*(theta)^(n_i)"""
    n0, theta0 = n_s[0], theta_s[0]
    a0 = I_theta0 * theta0**(n0)
    a_s = np.zeros(len(n_s))   
    a_s[0] = a0
    
    I_theta_i = a0 * float(theta_s[1])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s[1:], theta_s[1:])):
        a_i = I_theta_i/(theta_s[i+1])**(-n_i)
        try:
            a_s[i+1] = a_i
            I_theta_i = a_i * float(theta_s[i+2])**(-n_i)
        except IndexError:
            pass    
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
    norm_mpow = quad(multi_pow, 0, np.inf, args=(n_s, theta_s, 1, a_s))[0]
    y = multi_power1d(x, n_s, theta_s, 1) / norm_mpow
    return y

### 2D functions ###

def map2d(f, xx=None, yy=None):
    return f(xx,yy)

def power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    return z 

def trunc_power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Truncated power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    z[rr<=theta0] = I_theta0
    return z

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

def power2d_Flux2Amp(n, theta0, Flux=1, r_trunc=500):
    I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1)

def multi_power2d_Amp2Flux(n_s, theta_s, Amp=1):
    a_s = compute_multi_pow_norm(n_s, theta_s, Amp)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
    
    I_2D = Amp * np.pi * theta0**2
    for k in range(len(n_s)-1):
        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2) 
    I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2)   
    return I_2D

def multi_power2d_Flux2Amp(n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n_s, theta_s, Amp=1)


def I2Flux_pow(frac, n, theta0, r, I):
    """ Convert Intensity I(r) at r to total flux with fraction of power law.
        theata0 and r in the same unit"""
    Amp_pow = I * (r/theta0)**n
    Flux_pow = power2d_Amp2Flux(n, theta0, Amp=Amp_pow)
    Flux_tot = Flux_pow / frac
    return Flux_tot

def Flux2I_pow(frac, n, theta0, r, Flux):
    """ Convert total flux to Intensity I(r) at r.
        theata0 and r in the same unit"""
    Flux_pow = Flux * frac
    Amp_pow = power2d_Flux2Amp(n, theta0, Flux=Flux_pow)
    I = Amp_pow / (r/theta0)**n
    return I

### 1D/2D conversion factor ###

def C_mof2Dto1D(r_core, beta):
    """ gamma in pixel """
    return 1./(beta-1) * 2*math.sqrt(np.pi) * r_core * Gamma(beta) / Gamma(beta-1./2) 

def C_mof1Dto2D(r_core, beta):
    """ gamma in pixel """
    return 1. / C_mof2Dto1D(r_core, beta)

def C_pow2Dto1D(n, theta0):
    """ theta0 in pixel """
    return np.pi * theta0 * (n-1) / (n-2)

def C_pow1Dto2D(n, theta0):
    """ theta0 in pixel """
    return 1. / C_pow2Dto1D(n, theta0)

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

def C_mpow1Dto2D(n_s, theta_s):
    """ theta in pixel """
    return 1. / C_mpow2Dto1D(n_s, theta_s)



############################################
# Functions for PSF rendering with Galsim
############################################

# # Shift center for the purpose of accuracy (by default galsim round to integer!)
def get_center_offset(pos):
    x_pos, y_pos = pos + 1
    x_nominal = x_pos + 0.5
    y_nominal = y_pos + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    return (ix_nominal, iy_nominal), offset  

# Auxciliary function of drawing, practically devised to facilitate parallelization.
def draw_star(k, star_pos, Flux, psf_star, psf_size, full_image, pixel_scale=2.5):
    """ Draw star #k at position star_pos[k] with Flux[k], using a combined PSF (psf_star) on full_image"""
    stamp, bounds = get_stamp_bounds(k, star_pos, Flux, psf_star, psf_size,
                                     full_image, pixel_scale=pixel_scale)
    full_image[bounds] += stamp[bounds]

def get_stamp_bounds(k, star_pos, Flux, psf_star, psf_size, full_image, pixel_scale=2.5):
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


def make_noise_image(image_size, noise_std, random_seed=42, verbose=True):
    """ Make noise image """
    if verbose:
        print("Generate noise background w/ stddev = %.3g"%noise_std)
    
    noise_image = galsim.ImageF(image_size, image_size)
    rng = galsim.BaseDeviate(random_seed)
    
    gauss_noise = galsim.GaussianNoise(rng, sigma=noise_std)
    noise_image.addNoise(gauss_noise)  
    
    return noise_image.array


def make_base_image(image_size, stars, psf_base, pad=0, verbose=True):
    """ Background images composed of dim stars with fixed PSF psf_base"""
    if verbose:
        print("Generate base image of faint stars (flux < %.2g)."%(stars.F_bright))
    
    start = time.time()
    image_size = image_size + 2 * pad
    full_image0 = galsim.ImageF(image_size, image_size)
    
    star_pos = stars.star_pos_faint + pad
    Flux = stars.Flux_faint
    
    if len(star_pos) == 0:
        return np.zeros((image_size, image_size))
    
    # draw faint stars in Moffat with galsim in Fourier space   
    for k in range(len(star_pos)):
        try:
            draw_star(k, star_pos=star_pos, Flux=Flux,
                      psf_star=psf_base, psf_size=64, full_image=full_image0)
        except GalSimBoundsError as e:
            if verbose:
                print(e.__doc__)
                print(e.message)
            continue

    image_gs0 = full_image0.array
    
    end = time.time()
    if verbose: print("Total Time: %.3fs"%(end-start))
    
    return image_gs0[pad:image_size-pad, pad:image_size-pad]


def make_truth_image(psf, stars, contrast=1e6,
                     parallel=False, verbose=True, saturation=4.5e4):
    
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

    if hasattr(psf, 'image_size'):
        image_size = psf.image_size
        yy, xx = psf.yy, psf.xx
    else:
        print('Grid has not been built.')
        return None
    
    psf_core = getattr(psf, 'psf_core', psf.generate_core())
        
    psf_aureole = getattr(psf, 'psf_aureole',
                          psf.generate_aureole(contrast=contrast,
                                               psf_range=image_size)[0])
    
    # 1. Draw normal stars in Fourier
    full_image = galsim.ImageF(image_size, image_size)
    
    # separate stars into two samples
    Flux_A = stars.Flux_verybright
    star_pos_A = stars.star_pos_verybright
    
    Flux_B = stars.Flux_medbright
    star_pos_B =stars.star_pos_medbright
    
    # stick stamps from FFT on the canvas 
    for k, (pos, flux) in enumerate(zip(star_pos_B, Flux_B)): 

        psf_star = (1-frac) * psf_core + frac * psf_aureole
        psf_star = psf_star.withFlux(flux)
        
        (ix_nominal, iy_nominal), offset = get_center_offset(pos)

        stamp = psf_star.drawImage(scale=psf.pixel_scale, offset=offset, method='no_pixel')
        stamp.setCenter(ix_nominal,iy_nominal)
        
        try:
            bounds = stamp.bounds & full_image.bounds
            full_image[bounds] += stamp[bounds]
        except GalSimBoundsError:
            continue

    image_gs = full_image.array

    # 2. Draw bright stars in real space
    Amps_A = np.array([moffat2d_Flux2Amp(gamma_pix, beta, Flux=(1-frac)*flux)
                       for flux in Flux_A])
    func_core_2d_s = np.array([models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0,
                                               gamma=gamma_pix, alpha=beta)
                               for ((x0,y0), amp) in zip(star_pos_A, Amps_A)])
    
    func_aureole_2d_s = psf.draw_aureole2D_in_real(star_pos_A, frac * Flux_A)

    # option for drawing in parallel
    if not parallel:
        if verbose: 
            print("Rendering bright stars in serial...")
        image_real = np.sum([f2d(xx,yy) + p2d(xx,yy) 
                             for (f2d, p2d) in zip(func_core_2d_s, func_aureole_2d_s)], axis=0)
    else:
        if verbose: 
            print("Rendering bright stars in parallel...")
        func2d_s = np.concatenate([func_core_2d_s, func_aureole_2d_s])
        p_map2d = partial(map2d, xx=xx, yy=yy)
        
        image_stars = parallel_compute(func2d_s, p_map2d,
                                       lengthy_computation=False, verbose=True)
        image_real = np.sum(image_stars, axis=0)
    
    # combine the two image
    image = image_gs + image_real
    
    # saturation limit
    image[image>saturation] = saturation
        
    if verbose: 
        end = time.time()
        print("Total Time: %.3fs"%(end-start))
    
    return image
        
def generate_mock_image(psf, stars,
                        contrast=[1e6,1e7],
                        min_psf_range=90, 
                        max_psf_range=1200,
                        psf_scale=None,
                        parallel=False,
                        n_parallel=4,
                        draw_real=False,
                        n_real=2.5,
                        brightest_only=False,
                        interpolant='cubic'):
    
    image_size = psf.image_size
    yy, xx = psf.yy, psf.xx
    
    frac = psf.frac
    
    psf_scale = psf.pixel_scale if psf_scale is None else 2
        
    psf_c = getattr(psf, 'psf_core', psf.generate_core())
    
    # Setup the canvas
    full_image = galsim.ImageF(image_size, image_size)
        
    if not brightest_only:
        # 1. Draw medium bright stars with galsim in Fourier space
        psf_e, psf_size = psf.generate_aureole(contrast=contrast[0],
                                           psf_scale=psf_scale,
                                           psf_range=None,
                                           min_psf_range=min_psf_range//2,
                                           max_psf_range=max_psf_range//2,
                                           interpolant=interpolant)
        
        # Rounded PSF size to 2^k or 3*2^k for faster FT
        psf_size = round_good_fft(psf_size)

        psf_star = (1-frac) * psf_c + frac * psf_e               

        if not parallel:
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
                
    # 2. Draw very bright stars
    if (psf.aureole_model == "power"):
        if psf.n < n_real:
            draw_real = True
            
    if (image_size<500) | (psf.aureole_model == "multi-power"):
        draw_real = True
        
    if draw_real:
        # Draw aureole of very bright star (if high cost in FFT) in real space
        image_gs = full_image.array
        
        func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright,
                                                       frac * stars.Flux_verybright)
        image_real = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)
        
        image = image_gs + image_real
        
    else:
        # Draw very bright star in Fourier space 
        psf_e_2, psf_size_2 = psf.generate_aureole(contrast=contrast[1],
                                                   psf_scale=psf_scale,
                                                   psf_range=None,
                                                   min_psf_range=min_psf_range,
                                                   max_psf_range=max_psf_range,
                                                   interpolant=interpolant)
        
        psf_size_2 = round_good_fft(psf_size_2)
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

def generate_image_by_znorm(psf, stars,
                            contrast=[3e4,3e5],
                            min_psf_range=192, 
                            max_psf_range=768,
                            psf_scale=None,
                            parallel=False,
                            n_parallel=4,
                            draw_real=False,
                            brightest_only=False,
                            n_real=3,
                            interpolant='cubic'):

    image_size = psf.image_size
    yy, xx = psf.yy, psf.xx
    
    frac = psf.frac
    psf_scale = psf.pixel_scale if psf_scale is None else 2
    
    psf_c = getattr(psf, 'psf_core', psf.generate_core())

    # Update stellar flux:
    
#     theta_0_pix = psf.theta_0_pix
#     Amp_pow = stars0.z_norm * (r_scale/theta_0_pix)**psf.n
#     Flux_pow = power2d_Amp2Flux(psf.n, theta_0_pix, Amp=Amp_pow)
#     Flux = Flux_pow / frac
#     K = Flux[0] / stars0.Flux[0]
    
#     stars = Stars(stars0.star_pos, Flux,
#                   K * stars0.Flux_threshold,
#                   stars0.z_norm, verbose=False)
    
    Flux = psf.I2Flux(stars.z_norm, stars.r_scale)
    stars.update_Flux(Flux)
    
#     stars = Stars.from_znorm(psf, stars0.star_pos,
#                              stars0.z_norm,
#                              r_scale=stars0.r_scale,
#                              Flux_threshold=Flux_threshold)


    # Setup the canvas
    full_image = galsim.ImageF(image_size, image_size)
        
    if not brightest_only:
        # 1. Draw medium bright stars with galsim in Fourier space
        psf_e, psf_size = psf.generate_aureole(contrast=contrast[0],
                                               psf_scale=psf_scale,
                                               psf_range=None,
                                               min_psf_range=min_psf_range//2,
                                               max_psf_range=max_psf_range//4,
                                               interpolant=interpolant)

#         # Rounded PSF size to 2^k or 3*2^k for faster FT
        psf_size = round_good_fft(psf_size)

        # Draw medium bright stars with galsim in Fourier space
        psf_star = (1-frac) * psf_c + frac * psf_e               
                
        if (psf.n >= n_parallel):
            parallel = False
            
        if not parallel:
            # Draw in serial
            for k in range(stars.n_medbright):
#                 try:
                draw_star(k,
                          star_pos=stars.star_pos_medbright,
                          Flux=stars.Flux_medbright,
                          psf_star=psf_star,
                          psf_size=psf_size,
                          full_image=full_image)
#                 except GalSimBoundsError as e:
#                     pass

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
                
           
    # 2. Draw very bright stars
    if psf.n < n_real:
        draw_real = True
        
    if draw_real:
        # Draw aureole of very bright star (if high cost in FFT) in real space
        image_gs = full_image.array
        
        func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright,
                                                       frac * stars.Flux_verybright)
        image_real = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)
        
        image = image_gs + image_real
        
    else:
        # Draw very bright star in Fourier space 
        psf_e_2, psf_size_2 = psf.generate_aureole(contrast=contrast[1],
                                                   psf_scale=psf_scale,
                                                   psf_range=None,
                                                   min_psf_range=min_psf_range,
                                                   max_psf_range=max_psf_range,
                                                   interpolant=interpolant)
        
        psf_size_2 = round_good_fft(psf_size_2)
        
        psf_star_2 = (1-frac) * psf_c + frac * psf_e_2
        
        for k in range(stars.n_verybright):
            draw_star(k, star_pos=stars.star_pos_verybright,
                      Flux=stars.Flux_verybright,
                      psf_star=psf_star_2,
                      psf_size=psf_size_2,
                      full_image=full_image)
            
        image = full_image.array
                   
    return image

def generate_image_fit(res, psf, stars, image_base):
    pmed, pmean, pcov = get_params_fit(res)
    n_fit, mu_fit, logsigma_fit = pmed
    sigma_fit = 10**logsigma_fit
    print("Bakground : %.2f +/- %.2f"%(mu_fit, sigma_fit))

    psf_fit = psf.copy()
    psf_fit.update({'n':n_fit})
    
    noise_fit = make_noise_image(psf_fit.image_size, sigma_fit, verbose=False)
    image_fit = generate_image_by_znorm(psf_fit, stars,
                                        psf_scale=psf_fit.pixel_scale, draw_real=True)
    image_fit = image_fit + image_base + mu_fit
    
    return image_fit, noise_fit, pmed

############################################
# Functions for making priors
############################################

def build_priors(priors):
    """ Build priors for Bayesian fitting. Priors should has a (scipy-like) ppf class method."""
    def prior_transform(u):
        v = u.copy()
        for i in range(len(u)):
            v[i] = priors[i].ppf(u[i])
        return v
    return prior_transform

def draw_prior(priors, xlabels=None, plabels=None, save=False, dir_name='./'):
    
    x_s = [np.linspace(d.ppf(0.01), d.ppf(0.99), 100) for d in priors]
    
    fig, axes = plt.subplots(1, len(priors), figsize=(15,4))
    for k, ax in enumerate(axes):
        ax.plot(x_s[k], priors[k].pdf(x_s[k]),'-', lw=5, alpha=0.6, label=plabels[k])
        ax.legend()
        if xlabels is not None:
            ax.set_xlabel(xlabels[k], fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig("%s/Prior.png"%dir_name,dpi=100)
        plt.close()

