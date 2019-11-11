import time
import math
import galsim
from galsim import GalSimBoundsError
from utils import *

from itertools import combinations
from functools import partial
from usid_processing import parallel_compute
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
            
            self.theta_0 = theta_0
            self.n = n
            
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
    
    
    def I2I0(self, I, r=10):
        """ Convert I(r) at r to I0 with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return I2I0_pow(self.n, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2I0_mpow(self.n_s, self.theta_s_pix, r, I=I)
        
    def I02I(self, I0, r=10):
        """ Convert I(r) at r to I0 with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return I02I_pow(self.n, self.theta_0_pix, r, I0=I0)
        
        elif self.aureole_model == "multi-power":
            return I02I_mpow(self.n_s, self.theta_s_pix, r, I0=I0)
    
    def calculate_external_light(self, stars, n_iter=1):
        """ Calculate external scatter light that affects I at norm """
        I_ext = np.zeros(stars.n_bright)
        z_norm_verybright0 = stars.z_norm_verybright
        pos_source, pos_eval = stars.star_pos_verybright, stars.star_pos_bright
        
        if self.aureole_model == "power":
            calculate_external_light = partial(calculate_external_light_pow,
                                               n0=self.n, theta0=self.theta_s_pix,
                                               pos_source=pos_source, pos_eval=pos_eval)
        elif self.aureole_model == "multi-power":
            calculate_external_light = partial(calculate_external_light_mpow,
                                               n_s=self.n_s, theta_s=self.theta_s_pix,
                                               pos_source=pos_source, pos_eval=pos_eval)
            
        for i in range(n_iter):
            z_norm_verybright = z_norm_verybright0 - I_ext[:stars.n_verybright]
            I0_verybright = self.I2I0(z_norm_verybright, r=stars.r_scale)
            I_ext = calculate_external_light(I0=I0_verybright)
            
        return I_ext
    
    def I2Flux(self, I, r=10):
        """ Convert I(r) at r to total flux with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2Flux_mpow(self.frac, self.n_s, self.theta_s_pix, r, I=I)
        
    def Flux2I(self, Flux, r=10):
        """ Convert I(r) at r to total flux with power law.
        r in pixel """
        
        if self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n, self.theta_0_pix, r, Flux=Flux)
        
        elif self.aureole_model == "multi-power":
            return Flux2I_mpow(self.frac, self.n_s, self.theta_s_pix, r,  Flux=Flux)
        
    def SB2Flux(self, SB, BKG, ZP, r=10):
        # Intensity = I + BKG
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale) - BKG
        return self.I2Flux(I, r=r)
    
    def Flux2SB(self, Flux, BKG, ZP, r=10):
        I = self.Flux2I(Flux, r=r)
        return Intensity2SB(I+ BKG, BKG, ZP, self.pixel_scale)
    
    def plot_model_galsim(self, psf_core, psf_aureole,
                          image_size=400, contrast=None,
                          save=False, dir_name='.'):
        """ Plot Galsim 2D model averaged in 1D """
        from plotting import plot_PSF_model_galsim
        plot_PSF_model_galsim(psf_core, psf_aureole, self.params,
                              image_size, self.pixel_scale,
                              contrast=contrast,
                              save=save, dir_name=dir_name,
                              aureole_model=self.aureole_model)
        
    def draw_aureole2D_in_real(self, star_pos, Flux=None, I0=None):
        
        if self.aureole_model == "power":
            n = self.n
            theta_0_pix = self.theta_0_pix
            
            if I0 is None:
                I_theta0 = power2d_Flux2Amp(n, theta_0_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            func_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                          trunc_power2d(xx, yy, cen=cen,
                                                        n=n, theta0=theta_0_pix,
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
            
            func_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                          multi_power2d(xx, yy, cen=cen,
                                                        n_s=n_s, theta_s=theta_s_pix,
                                                        I_theta0=I)
                                          for (I, pos) in zip(I_theta0, star_pos)])
            
        return func_aureole_2d_s

    
class Stars:
    """ Class storing positions & flux of faint/medium-bright/bright stars"""
    def __init__(self, star_pos, Flux, 
                 Flux_threshold=[7e4, 2.7e6],
                 z_norm=None, r_scale=10, BKG=0,
                 verbose=False):
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
            self.BKG = BKG
            
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
                print("Not many bright stars, will draw in serial.\n")
                self.parallel = False 
            else: 
                print("Crowded fields w/ bright stars > 50, will draw in parallel.\n")
                self.parallel = True

    @classmethod            
    def from_znorm(cls, psf, star_pos, z_norm,
                   z_threshold=[9, 300], r_scale=10, 
                   verbose=False):
        
        Flux = psf.I2Flux(z_norm, r_scale)
        Flux_threshold = psf.I2Flux(z_threshold, r=R_scale)
        
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
    
    def remove_outsider(self, image_size, d=[24,12], verbose=False):
        """ Remove some stars far out of field. """
        star_pos = self.star_pos
        Flux = self.Flux

        out_A = (star_pos<-d[0]) | (star_pos>image_size+d[0])
        remove_A = np.logical_or.reduce(out_A, axis=1) & self.verybright
        
        out_B = (star_pos<-d[1]) | (star_pos>image_size+d[1])
        remove_B = np.logical_or.reduce(out_B, axis=1) & self.medbright

        remove = remove_A | remove_B
        return Stars(star_pos[~remove], Flux[~remove], self.Flux_threshold,
                     self.z_norm[~remove], BKG=self.BKG, verbose=True)
        

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
        
    def make_mask_map_dual(self, r_core, r_out=None,
                           mask_base=None, sn_thre=2.5,
                           draw=True, save=False, dir_name='.',
                           **kwargs):
        image = self.image
        stars = self.stars
        pad = self.pad 
            
        # S/N + Core mask
        mask_deep0, seg_deep0 = \
            make_mask_map_dual(image, stars, self.xx, self.yy, 
                               pad=pad, r_core=r_core, r_out=r_out,
                               mask_base=mask_base, sn_thre=sn_thre, **kwargs)
        
        
        self.mask_deep = mask_deep0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.seg_deep = seg_deep0[pad:self.image_size+pad, pad:self.image_size+pad]
        self.mask_deep0 = mask_deep0
        self.seg_deep0 = seg_deep0
            
        self.r_core = r_core
        self.r_out = r_out
        
        # Display mask
        if draw:
            from plotting import draw_mask_map
            draw_mask_map(image, seg_deep0, mask_deep0, stars,
                          pad=pad, r_core=r_core, r_out=r_out,
                          vmin=self.mu, vmax=self.mu+50, save=save, dir_name=dir_name)
            
    def make_mask_strip(self, width, n_strip, dist_strip=200,
                        draw=True, clean=True,
                        save=False, dir_name='.',
                        **kwargs):
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
                              BKG=stars.BKG, verbose=False)
            print(stars_new.z_norm.shape)
            self.stars_new = stars_new
            self.clean = clean
        
        # Display mask
        if draw:
            from plotting import draw_mask_map_strip
            draw_mask_map_strip(image, seg_comb0,
                                mask_comb0, stars_new, pad=pad,
                                ma_example=[mask_strip_s[0],
                                            mask_cross_s[0]],
                                r_core=self.r_core, vmin=self.mu, vmax=self.mu+50, save=save, dir_name=dir_name)
            
        
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
    norm_mpow = quad(multi_pow, 0, np.inf,
                     args=(n_s, theta_s, 1, a_s), limit=100)[0]
    y = multi_power1d(x, n_s, theta_s, 1) / norm_mpow
    return y

### 2D functions ###

def map2d(f, xx=None, yy=None):
    return f(xx,yy)

def map2d_k(k, func_list, xx=None, yy=None):
    return func_list[k](xx, yy)


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

def power2d_Flux2Amp(n, theta0, Flux=1):
    if n>2:
        I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    else:
        raise InconvergenceError('PSF is not convergent in Infinity.')
        
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1)

def multi_power2d_Amp2Flux(n_s, theta_s, Amp=1, theta_trunc=1e5):
    if np.ndim(Amp)>0:
        a_s = compute_multi_pow_norm(n_s, theta_s, 1)
        a_s = np.multiply(a_s[:,np.newaxis], Amp)
    else:
        a_s = compute_multi_pow_norm(n_s, theta_s, Amp)
        
    n0, theta0 = n_s[0], theta_s[0]
    
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

#     else:
#         raise InconvergenceError('PSF is not convergent in Infinity.')
        
    return I_2D

def multi_power2d_Flux2Amp(n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n_s, theta_s, Amp=1)

def I2I0_pow(n0, theta0, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with power law.
        theata_s and r in pixel """
    I_ = I * (r/theta0)**n0
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

def I2I0_mpow(n_s, theta_s, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s, right=True) - 1
        
    I0 = I * r**(n_s[i]) * theta_s[0]**(-n_s[0])
    for j in range(i):
        I0 *= theta_s[j+1]**(n_s[j]-n_s[j+1])
        
    return I0

def I02I_mpow(n_s, theta_s, r, I0=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s, right=True) - 1
        
    I = I0 / r**(n_s[i]) / theta_s[0]**(-n_s[0])
    for j in range(i):
        I *= theta_s[j+1]**(n_s[j+1]-n_s[j])
        
    return I


def calculate_external_light_pow(n0, theta0, pos_source, pos_eval, I0):
    # Calculate light produced by source (I0, pos_source) at pos_eval. 
    r_s = distance.cdist(pos_source,  pos_eval)
    
    I0_s = np.repeat(I0[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    I_s = I0_s / (r_s/theta0)**n0
    I_s[(r_s==0)] = 0
    
    return I_s.sum(axis=0)

def calculate_external_light_mpow(n_s, theta_s, pos_source, pos_eval, I0):
    # Calculate light produced by source (I0, pos_source) at pos_eval. 
    r_s = distance.cdist(pos_source,  pos_eval)
    r_inds = np.digitize(r_s, theta_s, right=True) - 1
    r_inds[r_inds<0] = 0
    
    r_inds_uni, r_inds_inv = np.unique(r_inds, return_inverse=True)
    
    I0_s = np.repeat(I0[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    I_s = I0_s / r_s**(n_s[r_inds]) / theta_s[0]**(-n_s[0])
    I_s[(r_s==0)] = 0
    
    factors = np.array([np.prod([theta_s[j+1]**(n_s[j+1]-n_s[j])
                                 for j in range(i)]) for i in r_inds_uni])
    I_s *= factors[r_inds_inv].reshape(len(I0),-1)
    
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
    if verbose: print("Total Time: %.3f s\n"%(end-start))
    
    return image_gs0[pad:image_size-pad, pad:image_size-pad]


def make_truth_image(psf, stars, contrast=1e6,
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
    
    full_image = galsim.ImageF(image_size, image_size)
    
    Flux_A = stars.Flux_bright
    star_pos_A = stars.star_pos_bright
    
#     Flux_B = stars.Flux_medbright
#     star_pos_B =stars.star_pos_medbright
    
    # 1. Draw normal stars in Fourier
#     for k, (pos, flux) in enumerate(zip(star_pos_B, Flux_B)): 

#         psf_star = (1-frac) * psf_core + frac * psf_aureole
#         psf_star = psf_star.withFlux(flux)
        
#         (ix_nominal, iy_nominal), offset = get_center_offset(pos)

#         # stick stamps from FFT on the canvas 
#         stamp = psf_star.drawImage(scale=psf.pixel_scale, offset=offset, method='no_pixel')
#         stamp.setCenter(ix_nominal,iy_nominal)
        
#         try:
#             bounds = stamp.bounds & full_image.bounds
#             full_image[bounds] += stamp[bounds]
#         except GalSimBoundsError:
#             continue

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
        
def generate_mock_image(psf, stars,
                        contrast=[1e6,1e7],
                        psf_range=[None, None],
                        min_psf_range=90, 
                        max_psf_range=1200,
                        psf_scale=None,
                        parallel=False,
                        n_parallel=4,
                        draw_real=False,
                        parallel_real=False,
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
                                               psf_range=psf_range[0],
                                               min_psf_range=min_psf_range//2,
                                               max_psf_range=max_psf_range//2,
                                               interpolant=interpolant)
        
        # Rounded PSF size to 2^k or 3*2^k for faster FT
#         psf_size = round_good_fft(psf_size)
        psf_size = psf_size // 2 * 2
        
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
            
#     if (image_size<500) | (psf.aureole_model == "multi-power"):
#         draw_real = True
        
    if draw_real:
        # Draw aureole of very bright star (if high cost in FFT) in real space
        image_gs = full_image.array
        
        func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright,
                                                       Flux=frac * stars.Flux_verybright)
        image_real = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)
        
        image = image_gs + image_real
        
    else:
        # Draw very bright star in Fourier space 
        psf_e_2, psf_size_2 = psf.generate_aureole(contrast=contrast[1],
                                                   psf_scale=psf_scale,
                                                   psf_range=psf_range[1],
                                                   min_psf_range=min_psf_range,
                                                   max_psf_range=max_psf_range,
                                                   interpolant=interpolant)
        
#         psf_size_2 = round_good_fft(psf_size_2)
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

def generate_image_by_znorm(psf, stars,
                            contrast=[3e4,3e5],
                            psf_range=[None,None],
                            min_psf_range=120, 
                            max_psf_range=1200,
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
    
    if psf_scale is None:
        psf_scale = psf.pixel_scale    

    # Subtract external light from brightest stars
    I_ext = psf.calculate_external_light(stars)
    z_norm = stars.z_norm.copy()
    z_norm[stars.bright] -= I_ext
    
    if draw_real & brightest_only:
        # Skip computation of Flux, and ignore core PSF
        I0_verybright = psf.I2I0(z_norm[stars.verybright], stars.r_scale)
        
    else:
        # Core PSF
        psf_c = getattr(psf, 'psf_core', psf.generate_core())

        # Update stellar flux:
        Flux = psf.I2Flux(z_norm, stars.r_scale)
        stars.update_Flux(Flux) 
        
    # Setup the canvas
    full_image = galsim.ImageF(image_size, image_size)
        
    if not brightest_only:
        # 1. Draw medium bright stars with galsim in Fourier space
        psf_e, psf_size = psf.generate_aureole(contrast=contrast[0],
                                               psf_scale=psf_scale,
                                               psf_range=psf_range[0],
                                               min_psf_range=min_psf_range//2,
                                               max_psf_range=max_psf_range//4,
                                               interpolant=interpolant)
        psf_size = psf_size // 2 * 2

        # Draw medium bright stars with galsim in Fourier space
        psf_star = (1-frac) * psf_c + frac * psf_e               
                
        if (psf.n >= n_parallel):
            parallel = False
            
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
#     if psf.n < n_real:
#         draw_real = True
        
    if draw_real:
        # Draw aureole of very bright star (if high cost in FFT) in real space
        image_gs = full_image.array
        
        if brightest_only:
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright,
                                                           I0=I0_verybright)
        else:
            func_aureole_2d_s = psf.draw_aureole2D_in_real(stars.star_pos_verybright,
                                                           Flux=frac * stars.Flux_verybright)
        
        image_real = np.sum([f2d(xx,yy) for f2d in func_aureole_2d_s], axis=0)
        
        image = image_gs + image_real
        
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
            draw_star(k, star_pos=stars.star_pos_verybright,
                      Flux=stars.Flux_verybright,
                      psf_star=psf_star_2,
                      psf_size=psf_size_2,
                      full_image=full_image)
            
        image = full_image.array
                   
    return image


def generate_image_fit(res, psf, stars, image_base):
    pmed, pmean, pcov = get_params_fit(res)
    N_n = (len(pmed)-2+1)//2
    pixel_scale = psf.pixel_scale
    
    if psf.aureole_model == "power":
        n_fit, mu_fit, logsigma_fit = pmed
        
    elif psf.aureole_model == "multi-power":
        n_s_fit = np.concatenate([pmed[:N_n], [4]])
        theta_0 = psf.theta_s[0]
        theta_s_fit = np.concatenate([[theta_0],
                                      np.atleast_1d(10**pmed[N_n:-2]),[900]])
        mu_fit, logsigma_fit = pmed[-2:]
        
    sigma_fit = 10**logsigma_fit
    print("Bakground : %.2f +/- %.2f"%(mu_fit, sigma_fit))

    psf_fit = psf.copy()
    
    if psf.aureole_model == "power":
        psf_fit.update({'n':n_fit})
        
    elif psf.aureole_model == "multi-power":
        psf_fit.update({'n_s':n_s_fit, 'theta_s':theta_s_fit})
    
    noise_fit = make_noise_image(psf_fit.image_size, sigma_fit, verbose=False)
    image_fit = generate_image_by_znorm(psf_fit, stars, psf_range=[640,1200],
                                        psf_scale=pixel_scale, draw_real=False)
    image_fit = image_fit + image_base + mu_fit
    
    return image_fit, noise_fit, pmed


############################################
# Priors and Likelihood Models for Fitting
############################################

def set_prior(n_est, mu_est, std_est,
              n_min=1, theta_in=90, theta_out=600,
              nspline=5, method='2p', **kwargs):
    
    log_t_in = np.log10(theta_in)
    log_t_out = np.log10(theta_out)
    dlog_t = log_t_out - log_t_in
    
    Prior_mu = stats.truncnorm(a=-2, b=0.1, loc=mu_est, scale=std_est)  # mu
    Prior_sigma = stats.truncnorm(a=-2, b=0.1,
                                  loc=np.log10(std_est), scale=0.5)   # sigma 
    
    if method == 'p':
        from plotting import draw_independent_priors
        Prior_n = stats.uniform(loc=n_est-0.5, scale=1.)   # n : n+/-0.5
        Priors = [Prior_n, Prior_mu, Prior_sigma]
        draw_independent_priors(Priors, **kwargs)

        prior_tf_p = build_independent_priors(Priors)
        return prior_tf_p

    elif method == '2p':
        def prior_tf_2p(u):
            v = u.copy()
            v[0] = u[0] * 0.6 + n_est-0.3        # n0 : n +/- 0.3
            v[1] = u[1] * (v[0]- 0.5 - n_min) + n_min        # n1 : n_min - n0-0.5
            v[2] = u[2] * dlog_t + log_t_in      # log theta1 : t_in-t_out  arcsec
            v[-2] = Prior_mu.ppf(u[-2])          # mu
            v[-1] = Prior_sigma.ppf(u[-1])       # log sigma 
            return v
        
        return prior_tf_2p
    
    elif method == '3p':
        def prior_tf_3p(u):
            v = u.copy()
            v[0] = u[0] * 0.6 + n_est-0.3                   # n0 : n +/- 0.3
            v[1] = u[1] * 0.5 + (v[0]-1)                    # n1 : n0-1 - n0-0.5
            v[2] = u[2] * max(-0.5, n_min+0.5-v[1]) + (v[1]-0.5)
                # n2 : [n_min, n1-1] - n1-0.5
            v[3] = u[3] * dlog_t + log_t_in                 
                # log theta1 : t_in-t_out  arcsec
            v[4] = u[4] * (log_t_out - v[3]) + v[3]
                # log theta2 : theta1 - t_out  arcsec
            v[-2] = Prior_mu.ppf(u[-2])          # mu
            v[-1] = Prior_sigma.ppf(u[-1])       # log sigma
            return v
        
        return prior_tf_3p
    
    elif method=='sp':
        def prior_tf_sp(u):
            v = u.copy()

            v[0] = u[0] * 0.6 + n_est-0.3                   
            # n0 : n +/- 0.3                                    
            
            for k in range(nspline-1):
                v[k+1] = u[k+1] * max(-0.3, 1.3-v[1]) + (v[1]-0.3)         
                # n_k+1 : [1, n_k-0.6] - n_k-0.3

            v[nspline] = u[nspline] * dlog_t + log_t_in
            # log theta1 : t_in-t_out  arcsec

            for k in range(nspline-2):
                
                v[k+nspline+1] = u[k+nspline+1] * min(0.7, log_t_out - v[k+nspline]) + v[k+nspline]
                # log theta_k+1: theta_k - [5*theta_k, t_out]  # in arcsec

            v[-2] = Prior_mu.ppf(u[-2])          # mu
            v[-1] = Prior_sigma.ppf(u[-1])       # log sigma
            
            return v
        
        return prior_tf_sp
        

def set_likelihood(data, mask_fit, psf, stars,
                   image_base=None, psf_range=[None,None],
                   brightest_only=True, parallel=False, draw_real=False,
                   draw_func=generate_mock_image, nspline=5, method='2p'):
    
    theta_0 = psf.theta_0
    
    if image_base is None:
        image_base = np.zeros((psf.image_size, psf.image_size))
        
    if method =='p':
        
        def loglike(v):
            n, mu = v[:-1]
            sigma = 10**v[-1]

            psf.update({'n':n})

            image_tri = draw_func(psf, stars, psf_range=psf_range,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            image_tri = image_tri + image_base + mu 

            ypred = image_tri[~mask_fit].ravel()
            residsq = (ypred - Y)**2 / sigma**2
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike
    
    elif method == '2p':
        
        def loglike_2p(v):
            n_s = v[:2]
            theta_s = [theta_0, 10**v[2]]
            mu, sigma = v[-2], 10**v[-1]

            psf.update({'n_s':n_s, 'theta_s':theta_s})

            image_tri = draw_func(psf, stars, psf_range=psf_range,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            image_tri = image_tri + image_base + mu 

            ypred = image_tri[~mask_fit].ravel()
            residsq = (ypred - data)**2 / sigma**2
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike
        
        return loglike_2p        
        
    elif method == '3p':
        
        def loglike_3p(v):
            n_s = v[:3]
            theta_s = [theta_0, 10**v[3], 10**v[4]]
            mu, sigma = v[-2], 10**v[-1]

            psf.update({'n_s':n_s, 'theta_s':theta_s})

            image_tri = draw_func(psf, stars, psf_range=psf_range,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            image_tri = image_tri + image_base + mu 

            ypred = image_tri[~mask_fit].ravel()
            residsq = (ypred - data)**2 / sigma**2
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike
        
        return loglike_3p
    
    elif method=='sp':
        def loglike_sp(v):
            n_s = v[:nspline]
            theta_s = [theta_0] + (10**v[nspline:-2]).tolist()
            mu, sigma = v[-2], 10**v[-1]

            psf.update({'n_s':n_s, 'theta_s':theta_s})

            image_tri = draw_func(psf, stars, psf_range=psf_range,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            image_tri = image_tri + image_base + mu 

            ypred = image_tri[~mask_fit].ravel()
            residsq = (ypred - data)**2 / sigma**2
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma**2))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike
        
        return loglike_sp
        