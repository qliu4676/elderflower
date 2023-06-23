import math
import numpy as np
from scipy.integrate import quad
from scipy.spatial import distance
from scipy.special import gamma as Gamma

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def dummy_decorator(func, *args, **kwargs):
            return func
        return dummy_decorator

############################################
# Analytic Functions for models
############################################

### numeric conversion ###
def fwhm_to_gamma(fwhm, beta):
    """ in arcsec """
    return fwhm / 2. / math.sqrt(2**(1./beta)-1)

def gamma_to_fwhm(gamma, beta):
    """ in arcsec """
    return gamma / fwhm_to_gamma(1, beta)

### interpolate value ###
def interp_I0(r, I, r0, r1, r2):
    """ Interpolate I0 at r0 with I(r) between r1 and r2 """
    range_intp = (r>r1) & (r<r2)
    logI0 = np.interp(r0, r[(r>r1)&(r<r2)], np.log10(I[(r>r1)&(r<r2)]))
    return 10**logI0
    
def compute_mean_I(r, I, r1, r2):
    """ Compute mean I under I(r) between r1 and r2 """
    range_intg = (r>r1) & (r<r2)
    r_range = r[range_intg]
    return np.trapz(I[range_intg], r_range)/(r_range.max()-r_range.min())
    
### funcs on single element ###

@njit
def compute_multi_pow_norm(n_s, theta_s, I_theta0):
    """ Compute normalization factor A of each power law component A_i*(theta)^(n_i)"""
    n0, theta0 = n_s[0], theta_s[0]
    a0 = I_theta0 * theta0**(n0)
    a_s = np.zeros(len(n_s))
    a_s[0] = a0
    
    I_theta_i = a0 * float(theta_s[1])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s[1:], theta_s[1:])):
        a_i = I_theta_i/(theta_s[i+1])**(-n_i)
        a_s[i+1] = a_i
        I_theta_i = a_i * float(theta_s[i+2])**(-n_i)
    return a_s


def trunc_pow(x, n, theta0, I_theta0=1):
    """ Truncated power law for single element, I = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * x**(-n) if x > theta0 else I_theta0
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
    
def flattened_linear(x, k, x0, y0):
    """ A linear function flattened at (x0,y0) of 1d array """
    return np.array(list(map(lambda x:k*x + (y0-k*x0) if x>=x0 else y0, x)))
    
def piecewise_linear(x, k1, k2, x0, y0):
    """ A piecewise linear function transitioned at (x0,y0) of 1d array """
    return np.array(list(map(lambda x:k1*x + (y0-k1*x0) if x>=x0 else k2*x + (y0-k2*x0), x)))
    
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

def multi_power1d(x, n_s, theta_s, I_theta0, clear=False):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    theta0 = theta_s[0]
    
    y = np.zeros_like(x)
    y[x<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (x>theta_s[k]) & (x<=theta_s[k+1]) if k<len(a_s)-1 else (x>theta_s[k])
        y[reg] = a_s[k] * np.power(x[reg], -n_s[k])
        
    if clear:
        y[x<=theta0] = 0
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

def power2d(xx, yy, n, theta0, I_theta0, cen):
    """ Power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n)
    return z

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
    """ Calculate light produced by source (I0, pos_source) at pos_eval. """
    r_s = distance.cdist(pos_source,  pos_eval)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1)
    
    r_s += 1e-3 # shift to avoid zero division
    I_s = I0_s / (r_s/theta0)**n0
    I_s[(r_s==1e-3)] = 0
    
    return I_s.sum(axis=0)

def calculate_external_light_mpow(n_s, theta_s_pix, pos_source, pos_eval, I0_source):
    """ Calculate light produced by source (I0_source, pos_source) at pos_eval. """
    r_s = distance.cdist(pos_source, pos_eval)
    r_inds = np.digitize(r_s, theta_s_pix, right=True) - 1
    
    r_inds_uni, r_inds_inv = np.unique(r_inds, return_inverse=True)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1)
    
    # Eq: I(r) = I0 * (theta0/theta1)^(n0) * (theta1/theta2)^(n1) *...* (theta_{k}/r)^(nk)
    r_s += 1e-3 # shift to avoid zero division
    I_s = I0_s * theta_s_pix[0]**n_s[0] / r_s**(n_s[r_inds])
    factors = np.array([np.prod([theta_s_pix[j+1]**(n_s[j+1]-n_s[j])
                                 for j in range(i)]) for i in r_inds_uni])
    I_s *= factors[r_inds_inv].reshape(len(I0_source),-1)
    
    I_s[(r_s==1e-3)] = 0
    
    return I_s.sum(axis=0)

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
