import os
import sys
import numpy as np

from astropy import wcs
from astropy.io import fits
from utils import crop_image

class Image:
    """ A Image Class """
    
    def __init__(self, hdu_path, image_bounds0, pixel_scale=2.5, pad=100):
        """ 
        
        Parameters
        ----------
        
        hdu_path: path of hdu data
        image_bounds0: boundary of measurement
        pixel_scale: pixel scale in arcsec/pixel
        pad : padding size of the image for fitting (default: 100)
        
        """
        
        self.image_bounds0 = image_bounds0
        self.pixel_scale = pixel_scale
        self.pad = pad
        
        patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = image_bounds0
        
        image_size = (patch_Xmax0 - patch_Xmin0) - 2 * pad
        
        self.image_size = image_size

        # Read hdu
        if os.path.isfile(hdu_path) is False:
            sys.exit("Image does not exist. Check path.")
            
        with fits.open(hdu_path) as hdul:
            print("Read Image :", hdu_path)
            self.data = data = hdul[0].data
            self.header = header = hdul[0].header
            self.wcs = wcs.WCS(header)

        # Background level and Zero Point
        try:
            bkg, ZP = np.array([header["BACKVAL"], header["ZP"]]).astype(float)
            print("BACKVAL: %.2f , ZP: %.2f , PIXSCALE: %.2f\n" %(bkg, ZP, pixel_scale))

        except KeyError:
            print("BKG / ZP / PIXSCALE missing in header --->")
            ZP = np.float(input("Input the value of ZP :"))
            bkg = np.float(input("Input the value of background :"))
            
        self.ZP = ZP
        self.bkg = bkg

        # Crop image
        image_bounds = (patch_Xmin0+pad, patch_Ymin0+pad,
                        patch_Xmax0-pad, patch_Ymax0-pad)

        self.image0, self.seg_patch0 = crop_image(data, image_bounds0, draw=False)
        
        self.image_bounds = image_bounds
        
        # Cutout of data
        self.image = self.image0[pad:-pad,pad:-pad]