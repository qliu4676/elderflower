import time
import math
import galsim
from galsim import GalSimBoundsError
from utils import *

from types import SimpleNamespace    


############################################
# Functions for making PSF models
############################################
from types import SimpleNamespace    

class PSF_Model:
    def __init__(self,
                 core_model='Moffat',
                 aureole_model='power',
                 pixel_scale=2.5,
                 params=None):
        
        self.core_model = core_model
        self.aureole_model = aureole_model
        
        self.pixel_scale = pixel_scale
        
        self.params = params
        for key, val in params.items():
            exec('self.' + key + ' = val')
            
            if (key == 'gamma') | ('theta' in key):
                exec('self.' + key + '_pix' + ' = val / pixel_scale')
            
    def plot(self, size=400):
        
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
        
        # plot
        plt.figure(figsize=(6,5))

        r = np.logspace(0, np.log10(size), 100)

        comp1 = moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D
        comp2 = profile(r) / c_aureole_2Dto1D

        if self.aureole_model == "multi-power":
            for t in theta_s_pix:
                plt.axvline(t, ls="--", color="k",alpha=0.3, zorder=1)
                
        plt.plot(r, np.log10((1-frac) * comp1 + comp2 * frac),
                 ls="-", lw=3,alpha=0.9, zorder=5, label='combined')
        plt.plot(r, np.log10((1-frac) * comp1),
                 ls="--", lw=3, alpha=0.9, zorder=1, label='core')
        plt.plot(r, np.log10(comp2 * frac),
                 ls="--", lw=3, alpha=0.9, label='aureole')
        
        plt.legend(loc=1, fontsize=12)
        plt.xscale('log')
        plt.xlabel('r [pix]', fontsize=14)
        plt.ylabel('log Intensity', fontsize=14)
        plt.ylim(-9, -0.5)
    
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
    
    def plot_model_galsim(self, psf_core, psf_aureole, image_size=400, contrast=None):
        from plotting import plot_PSF_model_galsim
        plot_PSF_model_galsim(psf_core, psf_aureole, self.params,
                              image_size, self.pixel_scale,
                              contrast=contrast, aureole_model=self.aureole_model)

    
### Galsim Modelling Funcs ###
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

def power2d_Flux2Amp(n, theta0, Flux=1, trunc=True):
    if trunc:
        I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    else:
        I_theta0 = (1./np.pi) * Flux * (n-2)/2 / theta0**2
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1, trunc=True):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1, trunc=trunc)

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


def Ir2Flux_tot(frac, n, theta0, r, Ir):
    """ Convert Intensity I(r) at r to total flux with frac = fraction of power law """
    Flux_pow = power2d_Amp2Flux(n, theta0, Amp=Ir * (r/theta0)**n)
    Flux_tot = Flux_pow / frac
    return Flux_tot


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

def make_base_image(image_size, psf_base, star_pos, Flux, verbose=False):
    """ Background images composed of dim stars with fixed PSF psf_base"""
    if len(star_pos) == 0:
        return np.zeros((image_size, image_size))
    
    start = time.time()
    full_image0 = galsim.ImageF(image_size, image_size)
    
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
    
    return image_gs0


def make_noise_image(image_size, noise_std, random_seed=42):
    """ Make noise image """
    noise_image = galsim.ImageF(image_size, image_size)
    rng = galsim.BaseDeviate(random_seed)
    gauss_noise = galsim.GaussianNoise(rng, sigma=noise_std)
    noise_image.addNoise(gauss_noise)  
    return noise_image.array


def make_truth_image(psf, star_pos, Flux, image_size,
                     Flux_threshold = [4e3, 2.7e6], contrast=1e6,
                     parallel=False, verbose=True, saturation=4.5e4):
    
    """
    Build Truth image with the position & flux. 
    Two methods provided: 1) Galsim convolution in FFT and 2) Astropy model in real space. 
    Flux_threshold : thereshold of flux [18 mag, 11 mag]
    """
    start = time.time()
    
    frac = psf.frac
    gamma_pix = psf.gamma_pix
    beta = psf.beta
    
    if psf.aureole_model == "power":
        n = psf.n
        theta_0_pix = psf.theta_0_pix
    elif psf.aureole_model == "multi-power":
        n_s = psf.n_s
        theta_s_pix = psf.theta_s_pix
    
    if hasattr(psf, 'psf_core'):
        psf_core = psf.psf_core
    else:
        psf_core = psf.generate_core()
        
    if hasattr(psf, 'psf_aureole'):
        psf_aureole = psf.psf_aureole
    else:
        psf_aureole, _ = psf.generate_aureole(contrast=contrast, psf_range=image_size)
    
    
    # grid and center of image
    yy, xx = np.mgrid[:image_size, :image_size]
    cen = ((image_size-1)/2., (image_size-1)/2.)
    
    # Draw normal stars in Fourier
    full_image = galsim.ImageF(image_size, image_size)
    
    # separate stars into two samples
    Flux_A = Flux[Flux >= Flux_threshold[1]]
    star_pos_A = star_pos[Flux >= Flux_threshold[1]]
    
    Flux_B = Flux[Flux < Flux_threshold[1]]
    star_pos_B = star_pos[Flux < Flux_threshold[1]]
    
    for k, (pos, flux) in enumerate(zip(star_pos_B, Flux_B)): 

        if flux >= Flux_threshold[0]:  
            psf_star = (1-frac) * psf_core + frac * psf_aureole
        else:
            psf_star = psf_core      # for very faint stars, just assume truth is core

        star = psf_star.withFlux(flux)
        
        (ix_nominal, iy_nominal), offset = get_center_offset(pos)

        stamp = star.drawImage(scale=psf.pixel_scale, offset=offset, method='no_pixel')
        stamp.setCenter(ix_nominal,iy_nominal)
        
        try:
            bounds = stamp.bounds & full_image.bounds
            full_image[bounds] += stamp[bounds]
        except GalSimBoundsError:
            continue


    image_gs = full_image.array

    # Draw bright stars in real
    Amps_A = np.array([moffat2d_Flux2Amp(gamma_pix, beta, Flux=(1-frac)*flux) for flux in Flux_A])
    func_core_2d_s = np.array([models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0, gamma=gamma_pix, alpha=beta)
                               for (amp, (x0,y0)) in zip(Amps_A, star_pos_A)])

    if psf.aureole_model == "power":
        I_theta0_pow = power2d_Flux2Amp(n, theta_0_pix, Flux=1)
        func_aureole_2d_s = np.array([lambda xx, yy, cen=cen, flux=flux:\
                                      trunc_power2d(xx, yy, cen=cen,
                                                    n=n, theta0=theta_0_pix,
                                                    I_theta0=I_theta0_pow * frac * flux)
                                      for (flux, cen) in zip(Flux_A, star_pos_A)])
        
    elif psf.aureole_model == "multi-power":
        I_theta0_mpow = multi_power2d_Flux2Amp(n_s=n_s, theta_s=theta_s_pix, Flux=1)
        func_aureole_2d_s = np.array([lambda xx, yy, cen=cen, flux=flux: \
                                          multi_power2d(xx, yy, cen=cen,
                                                        n_s=n_s, theta_s=theta_s_pix,
                                                        I_theta0=I_theta0_mpow * frac * flux)
                                      for (flux, cen) in zip(Flux_A, star_pos_A)])

    # Draw stars in real space
    if not parallel:
        print("Rendering bright stars in serial...")
        image_real = np.sum([f2d(xx,yy) + p2d(xx,yy) 
                             for (f2d, p2d) in zip(func_core_2d_s, func_aureole_2d_s)], axis=0)
    else:
        print("Rendering bright stars in parallel...")
        func2d_s = np.concatenate([func_core_2d_s[Flux>F_faint], func_aureole_2d_s])
        p_map2d = partial(map2d, xx=xx, yy=yy)
        
        image_stars = parallel_compute(func2d_s, p_map2d,
                                       lengthy_computation=False, verbose=True)
        image_real = np.sum(image_stars, axis=0)

    image = image_gs + image_real
        
    image[image>saturation] = saturation
        
    end = time.time()
    if verbose: print("Total Time: %.3fs"%(end-start))
    
    return image


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

