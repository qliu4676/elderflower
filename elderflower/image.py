import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
import astropy.units as u
from astropy.utils import lazyproperty

from .plotting import display, AsinhNorm

# Pixel scale (arcsec/pixel) for reduced and raw Dragonfly data
DF_pixel_scale = 2.5
DF_raw_pixel_scale = 2.85

# Gain (e-/ADU) of Dragonfly
DF_Gain = 0.37

class ImageButler:
    """
    
    A class storing Image info.
    
    Parameters
    ----------
    
    hdu_path : str
        path of hdu data
    obj_name : str
        object name
    band : str
        filter name
    pixel_scale : float
        pixel scale in arcsec/pixel
    pad : int
        padding size of the image for fitting (default: 50)
    ZP : float or None (default)
        zero point (if None, read from header)
    bkg : float or None (default)
        background (if None, read from header)
    G_eff : float or None (default)
        effective gain (e-/ADU)
    
    """
    
    def __init__(self, hdu_path, obj_name='', band='G',
                 pixel_scale=DF_pixel_scale, pad=50,
                 ZP=None, bkg=None, G_eff=None, verbose=True):
        from .utils import crop_image
    
        self.verbose = verbose
        self.obj_name = obj_name
        self.band = band
        self.pixel_scale = pixel_scale
        self.pad = pad
        
        # Read hdu
        assert os.path.isfile(hdu_path), "Image does not exist. Check path."
            
        with fits.open(hdu_path) as hdul:
            self.hdu_path = hdu_path
            if verbose: print("Read Image :", hdu_path)
            self.full_image = hdul[0].data
            self.header = header = hdul[0].header
            self.wcs = wcs.WCS(header)
            
        self.bkg = bkg
        self.ZP = ZP
        self.G_eff = G_eff
            
    def __str__(self):
        return "An ImageButler Class"

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.hdu_path}"
 
 
class Image(ImageButler):
    """
    
    An class storing images.
    
    Parameters
    ----------
    
    hdu_path : str
        path of hdu data
    bounds0 : list [X min, Y min, X max, Y max]
        boundary of region to be fit
    obj_name : str
        object name
    band : str
        filter name
    pixel_scale : float
        pixel scale in arcsec/pixel
    pad : int
        padding size of the image for fitting (default: 50)
    ZP : float or None (default)
        zero point (if None, read from header)
    bkg : float or None (default)
        background (if None, read from header)
    G_eff : float or None (default)
        effective gain (e-/ADU)
    
    """
        
    def __init__(self, hdu_path, bounds0,
                 obj_name='', band='G', pixel_scale=DF_pixel_scale,
                 pad=50, ZP=None, bkg=None, G_eff=None, verbose=True):
        from .utils import crop_image
        
        super().__init__(hdu_path, obj_name, band,
                         pixel_scale, pad, ZP, bkg, G_eff, verbose)
        
        self.bounds0 = bounds0
        
        patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = bounds0
        
        Ximage_size0 = (patch_Xmax0 - patch_Xmin0)
        Yimage_size0 = (patch_Ymax0 - patch_Ymin0)
        self.image_shape0 = (Yimage_size0, Ximage_size0)
        self.image_shape = (Yimage_size0 - 2 * pad, Ximage_size0 - 2 * pad)
        
        # Crop image
        self.bounds = (patch_Xmin0+pad, patch_Ymin0+pad,
                       patch_Xmax0-pad, patch_Ymax0-pad)

        self.image0 = crop_image(self.full_image, bounds0, origin=0, draw=False)
        
        # Cutout of data
        self.image = self.image0[pad:Yimage_size0-pad,
                                 pad:Ximage_size0-pad]
        
    def __str__(self):
        return "An Image Class"

    def __repr__(self):
        return ''.join([f"{self.__class__.__name__} cutout", str(self.bounds0)])
        
    def display(self, **kwargs):
        """ Display the image """
        display(self.image, **kwargs)
        
    def read_measurement_table(self, dir_measure, **kwargs):
        """ Read faint stars info and brightness measurement """
        from .utils import read_measurement_table
        self.table_faint, self.table_norm = \
                    read_measurement_table(dir_measure,
                                           self.bounds0,
                                           obj_name=self.obj_name,
                                           band=self.band,
                                           pad=self.pad, **kwargs)
                                           
    def assign_star_props(self, **kwargs):
        """ Assign position and flux for faint and bright stars from tables. """
    
        from .utils import assign_star_props
        
        if hasattr(self, 'table_faint') & hasattr(self, 'table_norm'):
            pos_ref = self.bounds[0], self.bounds[1]
            self.stars_bright, self.stars_all = \
            assign_star_props(self.ZP, self.bkg, self.image_shape, pos_ref,
                              self.table_norm, self.table_faint, **kwargs)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no stars info. \
                                    Read measurement tables first!")
                                    
                                    
    def generate_image_psf(self, psf, SE_catalog, seg_map, draw=False, dir_name='./tmp'):
        """ Generate image of stars from a PSF Model"""
        from elderflower.utils import (crop_catalog, identify_extended_source,
                                       calculate_color_term, cross_match,
                                       add_supplementary_SE_star, measure_Rnorm_all)
        from .modeling import generate_image_fit
        from .utils import check_save_path
        import shutil
        
        check_save_path(dir_name, make_new=False, verbose=False)
        
        band = self.band
        mag_name = '%smag'%band.lower()
        obj_name = self.obj_name
        bounds = self.bounds
        
        SE_cat = crop_catalog(SE_catalog, bounds)
        SE_cat_target, ext_cat = identify_extended_source(SE_cat, draw=draw)
        tab_target, tab_target_full, catalog_star = cross_match(self.wcs,
                                                                SE_cat_target,
                                                                bounds,
                                                                sep=3*u.arcsec,
                                                                mag_limit=15,
                                                                mag_name=mag_name,
                                                                verbose=False)
        
        CT = calculate_color_term(tab_target_full, mag_range=[13,18],
                                  mag_name=mag_name+'_PS', draw=draw)
        
        catalog_star["MAG_AUTO_corr"] = catalog_star[mag_name] + CT #corrected mag
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name+'_PS'] + CT
        
        catalog_star_name = os.path.join(dir_name, f'{obj_name}-catalog_PS_{band}_all.txt')
        catalog_star.write(catalog_star_name, overwrite=True, format='ascii')
        
        tab_target = add_supplementary_SE_star(tab_target, SE_cat_target,
                                               mag_saturate=13, draw=draw)
        
        tab_norm, res_thumb = measure_Rnorm_all(tab_target, bounds,
                                                self.wcs, self.full_image, seg_map,
                                                mag_limit=15, r_scale=12, width=1,
                                                obj_name=obj_name, mag_name=mag_name+'_PS',
                                                save=True, verbose=False, dir_name=dir_name)
        
        self.read_measurement_table(dir_name, r_scale=12, mag_limit=15, use_PS1_DR2=False)
        
        self.assign_star_props(r_scale=12, mag_threshold=[13.5,10.5],
                                verbose=False, draw=False, save=False, save_dir='test')
        
        stars = self.stars_bright
        
        image_stars, _, _ = generate_image_fit(psf, stars, self.image_shape)
        print("Image of stars is generated based on the PSF Model!")
        shutil.rmtree(dir_name)
        
        return image_stars

        
class ImageList(ImageButler):
    """
    
    A class storing a list of images.
    
    Parameters
    ----------

    hdu_path : str
        path of hdu data
    bounds0_list : list / turple
        list of boundaries of regions to be fit (Nx4)
        [[X min, Y min, X max, Y max],[...],...]
    obj_name : str
        object name
    band : str
        filter name
    pixel_scale : float
        pixel scale in arcsec/pixel
    pad : int
        padding size of the image for fitting (default: 50)
    ZP : float or None (default)
        zero point (if None, read from header)
    bkg : float or None (default)
        background (if None, read from header)
    G_eff : float or None (default)
        effective gain (e-/ADU)
    
    """
    
    def __init__(self, hdu_path, bounds0_list,
                 obj_name='', band='G', pixel_scale=DF_pixel_scale,
                 pad=50, ZP=None, bkg=None, G_eff=None, verbose=False):
        
        super().__init__(hdu_path, obj_name, band,
                         pixel_scale, pad, ZP, bkg, G_eff, verbose)
                         
        self.bounds0_list = np.atleast_2d(bounds0_list)
        
        self.Images = [Image(hdu_path, bounds0,
                             obj_name, band, pixel_scale,
                             pad, ZP, bkg, G_eff, verbose)
                       for bounds0 in self.bounds0_list]
        self.N_Image = len(self.Images)
        
        
    def __iter__(self):
        for Img in self.Images:
            yield Img
            
    def __getitem__(self, index):
        return self.Images[index]
    
    @lazyproperty
    def images(self):
        return np.array([Img.image for Img in self.Images])

    def display(self, fig=None, ax=None):
        """ Display the image list """
        
        if fig is None:
            n_row = int((self.N_Image-1)//4+1)
            fig, axes = plt.subplots(n_row, 4, figsize=(14,4*n_row))
            
        # Draw
        for i, ax in zip(range(self.N_Image), axes.ravel()):
            display(self.images[i], fig=fig, ax=ax)
            
        # Delete extra ax
        for ax in axes.ravel()[self.N_Image:]: fig.delaxes(ax)
            
        plt.tight_layout()
    
    def read_measurement_tables(self, dir_measure, **kwargs):
        """ Read faint stars info and brightness measurement """
        
        self.tables_norm = []
        self.tables_faint = []
        
        for Img in self.Images:
            Img.read_measurement_table(dir_measure, **kwargs)
            self.tables_faint += [Img.table_faint]
            self.tables_norm += [Img.table_norm]
                                    
    
    def assign_star_props(self, *args, **kwargs):
        """ Assign position and flux for faint and bright stars from tables. """
        
        stars_bright, stars_all = [], []
    
        for Img in self.Images:
            Img.assign_star_props(*args, **kwargs)
            stars_bright += [Img.stars_bright]
            stars_all += [Img.stars_all]
            
        self.stars_bright = stars_bright
        self.stars_all = stars_all

        return stars_bright, stars_all
    
    
    def make_base_image(self, psf_star, stars_all, psf_size=64, vmax=30, draw=True):
        
        """ Make basement image with fixed PSF and stars """
        
        from .modeling import make_base_image
        
        image_base = np.zeros_like(self.images)
        
        for i, (Image, stars) in enumerate(zip(self.Images, stars_all)):
        # Make sky background and draw dim stars
            image_base[i] = make_base_image(Image.image_shape, stars,
                                            psf_star, self.pad, psf_size,
                                            verbose=self.verbose)
            
            if draw:
                #display
                plt.imshow(image_base[i], vmin=0, vmax=vmax, norm=AsinhNorm(a=0.1))
                plt.colorbar()
                plt.show()
            
        self.image_base= image_base
            
    
    def make_mask(self, stars_list, dir_measure='../output/Measure',
                  by='aper',  r_core=None, r_out=None,
                  sn_thre=2.5, n_dilation=5, count=None,
                  n_strip=48, wid_strip=16, dist_strip=None,
                  wid_cross=10, dist_cross=72, clean=True,
                  draw=True, save=False, save_dir='../output/pic'):
        
        """Make Strip + Cross Mask"""
        
        from .mask import Mask
        from .utils import crop_image
        
        masks = []
        
        for Image, stars in zip(self.Images,
                                stars_list):
            mask = Mask(Image, stars)
            
            # Mask the main object by given shape parameters or read a map
            mask.make_mask_object(self.obj_name)
            
            # crop the full mask map into smaller one
            if hasattr(mask, 'mask_obj_field'):
                mask.mask_obj0 = crop_image(mask.mask_obj_field,
                                            Image.bounds0,
                                            origin=0, draw=False)
            else:
                mask.mask_obj0 = np.zeros(mask.shape, dtype=bool)

            # Primary SN threshold mask
            mask.make_mask_map_deep(dir_measure, by,
                                    r_core, r_out, count,
                                    obj_name=self.obj_name,
                                    band=self.band, 
                                    draw=draw, save=save, save_dir=save_dir)
            
            if stars.n_verybright > 0:
                # Supplementary Strip + Cross mask
                
                if dist_strip is None:
                    dist_strip = max(Image.image_shape)
                    
                mask.make_mask_advanced(n_strip, wid_strip, dist_strip,
                                        wid_cross, dist_cross, 
                                        clean=clean, draw=draw,
                                        save=save, save_dir=save_dir)

            masks += [mask]
                
        self.Masks = masks
        
        self.stars = [mask.stars_new for mask in masks]
    
    
    @property
    def mask_fits(self):
        """ Masks for fitting """
        return [mask.mask_fit for mask in self.Masks]
    
    @property
    def data(self):
        """ 1D array to be fit """
        data = [image[~mask_fit].copy().ravel()
                    for (image, mask_fit) in zip(self.images, self.mask_fits)]

        return data
        self
        
        
    def estimate_bkg(self):
        """ Estimate background level and std """
        
        from astropy.stats import sigma_clip
        
        self.mu_est = np.zeros(len(self.Images))
        self.std_est = np.zeros(len(self.Images))
        
        for i, (Image, mask) in enumerate(zip(self.Images, self.mask_fits)):
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
                      theta_out=300):
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
            container.set_likelihood(self.data[i],
                                     self.mask_fits[i],
                                     psf, stars[i],
                                     psf_range=[None, None],
                                     norm='brightness',
                                     G_eff=self.G_eff,
                                     image_base=self.image_base[i])
            
            # Set a few attributes to container for convenience
            container.image = self.images[i]
            container.data = self.data[i]
            container.mask = self.Masks[i]
            container.image_shape = self.Images[i].image_shape
            
            self.containers += [container]


