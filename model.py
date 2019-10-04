import time
import galsim
from galsim import GalSimBoundsError
from utils import *

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
    psf_scale: Pixel scale of the PSF, in general <= pixel scale of data. In arcsec/pix.
    theta_t: Inner flattening radius of power law to avoid divergence at the center. In arcsec.
    contrast: Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given.
    psf_range: Range of PSF. In arcsec.
    min_psf_range : Minimum range of PSF. In arcsec.
    max_psf_range : Maximum range of PSF. In arcsec.
    x_interpolant/k_interpolant: Interpolant method in Galsim.

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
    
    # PSF size in pixel
    psf_size = 2 * psf_range // psf_scale
    
    # Generate Grid of PSF and plot PSF model in real space onto it
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
    psf_model = trunc_power2d(xx_psf, yy_psf, n, cen=cen_psf, theta0=theta_t/psf_scale, I_theta0=1) 

    # Parse the image to Galsim PSF model by interpolation
    image_psf = galsim.ImageF(psf_model)
    psf_pow = galsim.InterpolatedImage(image_psf, flux=1, scale=psf_scale,
                                       x_interpolant=interpolant, k_interpolant=interpolant)
    return psf_pow, psf_size


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
    stamp, bounds = get_stamp_bounds(k, star_pos, Flux, psf_star, psf_size, full_image, pixel_scale=pixel_scale)
    full_image[bounds] += stamp[bounds]

def get_stamp_bounds(k, star_pos, Flux, psf_star, psf_size, full_image, pixel_scale=2.5):
    """ Get stamp and boundary of star #k at position star_pos[k] with Flux[k], using a combined PSF (psf_star) on full_image"""
    pos, flux = star_pos[k], Flux[k]       
    star = psf_star.withFlux(flux)

    # Account for the fractional part of the position
    (ix_nominal, iy_nominal), offset = get_center_offset(pos)

    stamp = star.drawImage(nx=psf_size, ny=psf_size, scale=pixel_scale, offset=offset, method='no_pixel')
    stamp.setCenter(ix_nominal, iy_nominal)
    
    bounds = stamp.bounds & full_image.bounds
    
    return stamp, bounds


    

def make_base_image(image_size, star_pos, Flux, psf_base):
    
    start = time.time()
    full_image0 = galsim.ImageF(image_size, image_size)
    
    # draw faint stars in Moffat with galsim in Fourier space   
    for k in range(len(star_pos)):
        try:
            draw_star(k, star_pos=star_pos, Flux=Flux,
                      psf_star=psf_base, psf_size=64, full_image=full_image0)
        except GalSimBoundsError:
            continue

    image_gs0 = full_image0.array
    end = time.time()
    print("Total Time: %.3fs"%(end-start))
    return image_gs0

def make_noise_image(image_size, noise_var, random_seed=42):
    noise_image = galsim.ImageF(image_size, image_size)
    rng = galsim.BaseDeviate(random_seed)
    gauss_noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_var))
    noise_image.addNoise(gauss_noise)  
    return noise_image.array


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
        ax.set_xlabel(xlabels[k], fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig("%s/Prior.png"%dir_name,dpi=100)
        plt.close()

