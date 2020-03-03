import os
import sys
import numpy as np

from astropy import wcs
from astropy.io import fits
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
        image_bounds0: boundary of measurement
        pixel_scale: pixel scale in arcsec/pixel
        pad : padding size of the image for fitting (default: 100)
        
        """
        
        super().__init__(hdu_path, pixel_scale, pad, verbose)
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
    def __init__(self, hdu_path, image_bounds0_list, pixel_scale=2.5, pad=100, verbose=False):
        super().__init__(hdu_path, pixel_scale, pad, verbose)
        
        self.images = [Image(hdu_path, image_bounds0, pixel_scale, pad, verbose)
                       for image_bounds0 in image_bounds0_list]
    