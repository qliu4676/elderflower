import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.utils import lazyproperty

from .utils import crop_image
from .plotting import AsinhNorm


class ImageButler:
    """ A Image Butler """
    
    def __init__(self, hdu_path, pixel_scale=2.5, pad=100, verbose=True):
        """ 
        
        Parameters
        ----------
        
        hdu_path: path of hdu data
        pixel_scale: pixel scale in arcsec/pixel
        pad : padding size of the image for fitting (default: 100)
        
        """
        
        self.verbose = verbose
        self.pixel_scale = pixel_scale
        self.pad = pad
        
        # Read hdu
        if os.path.isfile(hdu_path) is False:
            sys.exit("Image does not exist. Check path.")
            
        with fits.open(hdu_path) as hdul:
            self.hdu_path = hdu_path
            if verbose: print("Read Image :", hdu_path)
            self.full_image = hdul[0].data
            self.header = header = hdul[0].header
            self.wcs = wcs.WCS(header)

        # Background level and Zero Point
        try:
            bkg, ZP = np.array([header["BACKVAL"], header["ZP"]]).astype(float)
            if verbose:
                print("BACKVAL: %.2f , ZP: %.2f , PIXSCALE: %.2f\n" %(bkg, ZP, pixel_scale))

        except KeyError:
            print("BKG / ZP / PIXSCALE missing in header --->")
            ZP = np.float(input("Input the value of ZP :"))
            bkg = np.float(input("Input the value of background :"))
            
        self.ZP = ZP
        self.bkg = bkg
        
    def __str__(self):
        return "An ImageButler Class"

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.hdu_path}"

        
class Image(ImageButler):
    """ A Image Class """
        
    def __init__(self, hdu_path, image_bounds0, pixel_scale=2.5, pad=100, verbose=True):
        """ 
        
        Parameters
        ----------
        
        hdu_path: path of hdu data
        image_bounds0: boundary of region to be fit [X min, Y min, X max, Y max]
        pixel_scale: pixel scale in arcsec/pixel
        pad : padding size of the image for fitting (default: 100)
        
        """
        
        super().__init__(hdu_path,pixel_scale, pad, verbose)
        
        self.image_bounds0 = image_bounds0
        
        patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = image_bounds0
        
        self.image_size = (patch_Xmax0 - patch_Xmin0) - 2 * pad
        
        # Crop image
        self.image_bounds = (patch_Xmin0+pad, patch_Ymin0+pad,
                             patch_Xmax0-pad, patch_Ymax0-pad)

        self.image0 = crop_image(self.full_image, image_bounds0, draw=False)
        
        # Cutout of data
        self.image = self.image0[pad:-pad,pad:-pad]
        
    def __str__(self):
        return "An Image Class"

    def __repr__(self):
        return ''.join([f"{self.__class__.__name__}", str(self.image_bounds0)])
        
        
class ImageList(ImageButler):
    """ A ImageList Class """
    
    def __init__(self, hdu_path, image_bounds0_list,
                 pixel_scale=2.5, pad=100, verbose=False):
        
        """ 
        
        Parameters
        ----------
        
        hdu_path: path of hdu data
        image_bounds0_list: list of boundaries of regions to be fit (Nx4)
                            [X min, Y min, X max, Y max]
        pixel_scale: pixel scale in arcsec/pixel
        pad : padding size of the image for fitting (default: 100)
        
        """
        
        super().__init__(hdu_path, pixel_scale, pad, verbose)
        
        self.Images = [Image(hdu_path, image_bounds0, pixel_scale, pad, verbose)
                       for image_bounds0 in np.atleast_2d(image_bounds0_list)]
        self.N_Image = len(self.Images)
    
    
    @lazyproperty
    def images(self):
        return np.array([Image.image for Image in self.Images])

    
    def assign_star_props(self,
                          tables_faint,
                          tables_res_Rnorm,
                          *args, **kwargs):
        
        """ Assign position and flux for faint and bright stars from tables. """
    
        from .utils import assign_star_props
        
        stars0, stars_all = [], []
        
        for Image, tab_f, tab_res in zip(self.Images,
                                         tables_faint,
                                         tables_res_Rnorm):
            s0, sa = assign_star_props(tab_f, tab_res, Image,
                                       *args, **kwargs)
            stars0 += [s0]
            stars_all += [sa]

        return stars0, stars_all
    
    
    def make_base_image(self, psf_star, stars_all, psf_size=64, vmax=30, draw=True):
        
        """ Make basement image with fixed PSF and stars """
        
        from .modeling import make_base_image
        
        image_base = np.zeros_like(self.images)
        
        for i, (Image, stars) in enumerate(zip(self.Images, stars_all)):
        # Make sky background and draw dim stars
            image_base[i] = make_base_image(Image.image_size, stars,
                                            psf_star, self.pad, psf_size, verbose=self.verbose)
            
            if draw:
                #display
                plt.imshow(image_base[i], vmin=0, vmax=vmax, norm=AsinhNorm(a=0.1))
                plt.colorbar()
                plt.show()
            
        self.image_base= image_base
            
    
    def make_mask(self, stars_list, dir_measure='../output/Measure',
                  by='radius',  r_core=None, r_out=None,
                  sn_thre=2.5, n_dilation=5, count=None,
                  n_strip=48, wid_strip=16, dist_strip=None,
                  wid_cross=10, dist_cross=72, clean=True,
                  draw=True, save=False, save_dir='../output/pic'):
        
        """Make Strip + Cross Mask"""
        
        from .mask import Mask
        
        masks = []
        
        for Image, stars in zip(self.Images,
                                stars_list):
            mask = Mask(Image, stars)

            # Primary SN threshold mask
            mask.make_mask_map_deep(dir_measure, by, r_core, r_out, count,
                                    draw=draw, save=save, save_dir=save_dir)
            
            if stars.n_verybright > 0:
                # Supplementary Strip + Cross mask
                
                if dist_strip is None:
                    dist_strip = Image.image_size    
                    
                mask.make_mask_strip(n_strip, wid_strip, dist_strip,
                                     wid_cross, dist_cross, clean=clean,
                                     draw=draw, save=save, save_dir=save_dir)

            masks += [mask]
                
        self.Masks = masks
        
        self.stars = [mask.stars_new for mask in masks]
    
    
    @property
    def mask_fit(self):
        """ Masking for fit """
        return [getattr(mask, 'mask_comb', mask.mask_deep) for mask in self.Masks]
        
    @property
    def data(self):
        """ 1D array to be fit """
        data = [image[~mask].copy().ravel()
                    for (image, mask) in zip(self.images, self.mask_fit)]

        return data
        
        
    def estimate_bkg(self):
        
        """ Estimate background level and std """
        
        from astropy.stats import sigma_clip
        
        self.mu_est = np.zeros(len(self.Images))
        self.std_est = np.zeros(len(self.Images))
        
        for i, (Image, mask) in enumerate(zip(self.Images, self.mask_fit)):
        
            data_sky = sigma_clip(Image.image[~mask], sigma=3)
            
            mu_patch, std_patch = np.mean(data_sky), np.std(data_sky)
            
            self.mu_est[i] = mu_patch
            self.std_est[i] = std_patch
            
            print(repr(Image))
            print("Estimate of Background: (%.3f +/- %.3f)"%(mu_patch, std_patch))
            
    def set_container(self,
                      psf, stars,
                      n_est=3.2,
                      n_spline=2,
                      leg2d=False,
                      fit_sigma=True,
                      fit_frac=False,
                      brightest_only=False,
                      parallel=False,
                      draw_real=True,
                      n_min=1,
                      theta_in=50,
                      theta_out=240):
        """ Container for fit storing prior and likelihood function """
        
        from .container import Container
        
        self.containers = []
        
        for i in range(self.N_Image):
            
            container = Container(n_spline, leg2d, 
                                  fit_sigma, fit_frac,
                                  brightest_only,
                                  parallel, draw_real)
            # Set Priors
            container.set_prior(n_est, self.bkg, self.std_est[i],
                                n_min=n_min, theta_in=theta_in, theta_out=theta_out)

            # Set Likelihood
            container.set_likelihood(self.data[i], self.mask_fit[i], psf, stars[i], 
                                     psf_range=[None, None], norm='brightness',
                                     image_base=self.image_base[i])
            
            # Set a few attributes to container for convenience
            container.image = self.images[i]
            container.data = self.data[i]
            container.mask = self.Masks[i]
            container.image_size = self.Images[i].image_size
            
            self.containers += [container]

            


