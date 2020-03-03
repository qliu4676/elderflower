import os
import sys
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.utils import lazyproperty

from .utils import crop_image

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
            self.data = hdul[0].data
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

        self.image0 = crop_image(self.data, image_bounds0, draw=False)
        
        # Cutout of data
        self.image = self.image0[pad:-pad,pad:-pad]
        
    def __str__(self):
        return "An Image Class"

    def __repr__(self):
        return ''.join([f"{self.__class__.__name__}", str(self.image_bounds0)])
        
        
class ImageList(ImageButler):
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
        
        self.images = [Image(hdu_path, image_bounds0, pixel_scale, pad, verbose)
                       for image_bounds0 in np.atleast_2d(image_bounds0_list)]

    def assign_star_props(self,
                          tables_faint,
                          tables_res_Rnorm,
                          *args, **kwargs):
        
        """ Assign position and flux for faint and bright stars from tables. """
    
        from .utils import assign_star_props
        
        stars0, stars_all = [], []
        
        for image, tab_f, tab_res in zip(self.images,
                                         tables_faint,
                                         tables_res_Rnorm):
            s0, sa = assign_star_props(tab_f, tab_res, image,
                                       *args, **kwargs)
            stars0 += [s0]
            stars_all += [sa]

        return stars0, stars_all
    
    def make_mask(self, stars_list, dir_measure='../output/Measure',
                  by='radius',  r_core=None, r_out=None,
                  sn_thre=2.5, n_dilation=5, count=None,
                  n_strip=48, wid_strip=16, dist_strip=800,
                  wid_cross=10, dist_cross=72, clean=True,
                  draw=True, save=False, save_dir='../output/pic'):
        
        """Make Strip + Cross Mask"""
        
        from .mask import Mask
        
        masks = []
        
        for image, stars in zip(self.images,
                                stars_list):
            mask = Mask(image, stars)

            # Deep Mask
            # seg_base = "./Measure-PS/Seg_PS_X%dY%d.fits" %(patch_Xmin0, patch_Ymin0)
            mask.make_mask_map_deep(dir_measure, by, 
                                    r_core, r_out, count,
                                    draw=draw, save=save, save_dir=save_dir)

            # Strip + Cross mask
            mask.make_mask_strip(n_strip, wid_strip, dist_strip,
                                 wid_cross, dist_cross, clean=clean,
                                 draw=draw, save=save, save_dir=save_dir)

            masks += [mask]
                
        self.masks = masks
        
        self.stars = [mask.stars_new for mask in masks]
    
    
    @property
    def mask_fit(self):
        return [getattr(mask, 'mask_comb', mask.mask_deep) for mask in self.masks]
        
    @property
    def data_fit(self):
        
        data_fit = []
        
        for image, mask in zip(self.images, self.mask_fit):
            Y = image.image[~mask].copy().ravel()
            data_fit += [Y]
            
        return data_fit

