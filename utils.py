import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
import numpy as np
from astropy.table import Table
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch
norm1 = ImageNormalize(stretch=LogStretch())
norm2 = ImageNormalize(stretch=LogStretch())
from skimage import morphology, transform


from photutils import Background2D, SExtractorBackground
from photutils import detect_threshold, detect_sources, deblend_sources
from astropy.stats import SigmaClip, mad_std, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def power1d(x, n, theta, I_theta): 
    x[x<=0] = x[x>0].min()
    a = I_theta/(theta)**(-n)
    y = a * np.power(x, -n)
    return y

def power2d(x, y, n, theta, I_theta, cen=cen): 
    d = np.sqrt((x-cen[0])**2+(y-cen[1])**2)
    a = I_theta/(theta)**(-n)
    d[d<=0] = d[d>0].min()
    z = a * np.power(d, -n) 
    return z 

def colorbar(mappable, pad=0.2):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    return fig.colorbar(mappable, cax=cax)

def background_sub_SE(field, mask=None, b_size=64, f_size=3, n_iter=5):
    # Subtract background using SE estimator with mask
    Bkg = Background2D(field, mask=mask, bkg_estimator=SExtractorBackground(),
                       box_size=(b_size, b_size), filter_size=(f_size, f_size),
                       sigma_clip=SigmaClip(sigma=3., maxiters=n_iter))
    back = Bkg.background
    back_rms = Bkg.background_rms
    if mask is not None:
        back *= ~mask
    field_sub = field - back
    return back, back_rms

def display_background_sub(field, back):
    norm1 = ImageNormalize(stretch=LogStretch())
    norm2 = ImageNormalize(stretch=LogStretch())
    # Display and save background subtraction result with comparison 
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(17,6))
    ax1.imshow(field, origin="lower", aspect="auto", cmap="gray", vmin=vmin_3mad(field), vmax=vmax_2sig(field),norm=norm1)
    im2 = ax2.imshow(back, origin='lower', aspect="auto", cmap='gray')
    colorbar(im2)
    ax3.imshow(field - back, origin='lower', aspect="auto", cmap='gray', vmin=0., vmax=vmax_2sig(field - back),norm=norm2)
    plt.tight_layout()

def vmin_3mad(img):
    # lower limit of visual imshow defined by 3 mad above median
    return np.median(img)-3*mad_std(img)

def vmax_2sig(img):
    # upper limit of visual imshow defined by 2 sigma above median
    return np.median(img)+2*np.std(img)

def source_detection(data, sn=2, b_size=120, k_size=3, fwhm=3, 
                     morph_oper=morphology.dilation,
                     sub_background=True, mask=None):
    if sub_background:
        back, back_rms = background_sub_SE(data, b_size=b_size)
        threshold = back + (sn * back_rms)
    else:
        back = np.zeros_like(data)
        threshold = np.nanstd(data)
    sigma = fwhm * gaussian_fwhm_to_sigma    # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=k_size, y_size=k_size)
    kernel.normalize()
    segm_sm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel, mask=mask)
    
    segm_sm = morph_oper(segm_sm.data)
    data_ma = data.copy() - back
    data_ma[segm_sm!=0] = np.nan

    return data_ma, segm_sm