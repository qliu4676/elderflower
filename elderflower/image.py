import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
import astropy.units as u
from astropy.utils import lazyproperty

from .plotting import display, AsinhNorm, colorbar

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
                 pixel_scale=DF_pixel_scale, pad=0,
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
            self.full_wcs = wcs.WCS(header)
            
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
                 pad=0, ZP=None, bkg=None, G_eff=None, verbose=True):
        from .utils import crop_image
        
        super().__init__(hdu_path, obj_name, band,
                         pixel_scale, pad, ZP, bkg, G_eff, verbose)
        
        self.bounds0 = np.array(bounds0)
        
        patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = self.bounds0
        
        nX0 = (patch_Xmax0 - patch_Xmin0)
        nY0 = (patch_Ymax0 - patch_Ymin0)
        self.image_shape0 = (nY0, nX0)
        self.image_shape = (nY0 - 2 * pad, nX0 - 2 * pad)
        
        self.cen0 = ((nX0-1)/2., (nY0-1)/2.)
        self.cen = ((nX0 - 2 * pad-1)/2.,
                    (nY0 - 2 * pad-1)/2.)
        
        full_wcs = self.full_wcs
        # Image cutout
        self.bounds = np.array([patch_Xmin0+pad, patch_Ymin0+pad,
                                patch_Xmax0-pad, patch_Ymax0-pad])

        self.image0, self.wcs0 = crop_image(self.full_image,
                                            self.bounds0, wcs=full_wcs)
        
        # Cutout with pad
        self.image, self.wcs = crop_image(self.full_image,
                                          self.bounds, wcs=full_wcs)
        
    def __str__(self):
        return "An Image Class"

    def __repr__(self):
        return ''.join([f"{self.__class__.__name__} cutout", str(self.bounds0)])
        
    def display(self, **kwargs):
        """ Display the image """
        display(self.image, **kwargs)
        
    def make_base_image(self, psf_star, stars, vmax=30, draw=False):
        """ Make basement image with fixed PSF and stars """

        from .modeling import make_base_image

        psf_size = int(120/self.pixel_scale) # 2 arcmin
        # Draw dim stars
        self.image_base = make_base_image(self.image_shape, stars,
                                          psf_star, self.pad, psf_size,
                                          verbose=self.verbose)
        if draw:
            #display
            m = plt.imshow(self.image_base, norm=AsinhNorm(a=0.1, vmin=0, vmax=vmax))
            colorbar(m)
            plt.show()
        
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
                                    
    def fit_n0(self, dir_measure, **kwargs):
        from .utils import fit_n0
        n0, d_n0 = fit_n0(dir_measure, self.bounds0,
                         self.obj_name, self.band, self.bkg, self.ZP, **kwargs)
        self.n0 = n0
        self.d_n0 = d_n0
        return n0, d_n0
                                    
    def generate_image_psf(self, psf,
                           SE_catalog, seg_map,
                           r_scale=12,
                           mag_threshold=[13.5,12],
                           mag_saturate=13,
                           mag_limit=15,
                           make_segm=False, K=2.5,
                           catalog_sup='SE',
                           use_PS1_DR2=False, draw=False,
                           keep_tmp=False, dir_tmp='./tmp'):
        """ Generate image of stars from a PSF Model"""
        
        from .utils import (crop_catalog,
                            identify_extended_source,
                            calculate_color_term,
                            fit_empirical_aperture,
                            make_segm_from_catalog,
                            add_supplementary_SE_star,
                            add_supplementary_atlas,
                            measure_Rnorm_all)
        from .crossmatch import cross_match_PS1
        from .modeling import generate_image_fit
        from .utils import check_save_path
        import shutil
        
        if use_PS1_DR2: dir_tmp+='_PS2'
        check_save_path(dir_tmp, make_new=False, verbose=False)
        
        band = self.band
        obj_name = self.obj_name
        bounds = self.bounds
        
        SE_cat = crop_catalog(SE_catalog, bounds)
        SE_cat_target, ext_cat = identify_extended_source(SE_cat, draw=draw)
        
        # Use PANSTARRS DR1 or DR2?
        if use_PS1_DR2:
            mag_name = mag_name_cat = band.lower()+'MeanPSFMag'
        else:
            mag_name = band.lower()+'mag'
            mag_name_cat = band.lower()+'mag_PS'

        # Crossmatch with PANSTRRS mag < mag_limit
        tab_target, tab_target_full, catalog_star = \
                                    cross_match_PS1(band, self.full_wcs,
                                                    SE_cat_target, bounds,
                                                    pixel_scale=self.pixel_scale,
                                                    mag_limit=mag_limit,
                                                    use_PS1_DR2=use_PS1_DR2,
                                                    verbose=False)
        
        CT = calculate_color_term(tab_target_full, mag_range=[13,18],
                                  mag_name=mag_name_cat, draw=draw)
        
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
        catalog_star["MAG_AUTO_corr"] = catalog_star[mag_name] + CT #corrected mag
        
        catalog_star_name = os.path.join(dir_tmp, f'{obj_name}-catalog_PS_{band}_all.txt')
        catalog_star.write(catalog_star_name, overwrite=True, format='ascii')
        
        if catalog_sup == "ATLAS":
            tab_target = add_supplementary_atlas(tab_target, catalog_sup, SE_catalog,
                                                 mag_saturate=mag_saturate)
            
        elif catalog_sup == "SE":
            print('Add supplementary star based on SE measurements.')
            tab_target = add_supplementary_SE_star(tab_target, SE_cat_target,
                                                   mag_saturate=mag_saturate, draw=draw)
        self.tab_target = tab_target
        
        # Measure I at r0
        tab_norm, res_thumb = measure_Rnorm_all(tab_target, bounds,
                                                self.full_wcs, self.full_image, seg_map,
                                                mag_limit=mag_limit, r_scale=r_scale, width_ring_pix=0.5,
                                                enlarge_window=2,
                                                width_cross_pix=int(10/self.pixel_scale),
                                                obj_name=obj_name, mag_name=mag_name_cat,
                                                save=True, verbose=False, dir_name=dir_tmp)
        
        self.read_measurement_table(dir_tmp,  r_scale=r_scale, mag_limit=mag_limit)
        
        # Make Star Models
        self.assign_star_props(r_scale=r_scale, mag_threshold=mag_threshold,
                               verbose=True, draw=False, save=False)
        
        self.stars_gen = stars = self.stars_bright
        
        if make_segm:
            # Make mask map
            estimate_radius = fit_empirical_aperture(tab_target_full, seg_map,
                                                    mag_name=mag_name_cat, K=K,
                                                    degree=2, draw=draw)

            seg_map_cat = make_segm_from_catalog(catalog_star, bounds,
                                                estimate_radius,
                                                mag_name=mag_name,
                                                obj_name=obj_name,
                                                band=band,
                                                ext_cat=ext_cat,
                                                draw=draw,
                                                save=False,
                                                dir_name=dir_tmp)
            self.seg_map = seg_map_cat
        
        # Generate model star
        image_stars, _, _ = generate_image_fit(psf.copy(), stars.copy(), self.image_shape)
        print("Image of stars has been generated based on the PSF Model!")
        
        # Delete tmp dir
        if not keep_tmp:
            shutil.rmtree(dir_tmp)
        
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
                 pad=0, ZP=None, bkg=None, G_eff=None, verbose=False):
        
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
        
    
    def make_base_image(self, psf_star, stars_all, vmax=30, draw=True):
        
        """ Make basement image with fixed PSF and stars """
        
        for i, (Image, stars) in enumerate(zip(self.Images, stars_all)):
            Image.make_base_image(psf_star, stars)
    
    def make_mask(self, stars_list, dir_measure='../output/Measure',
                  by='aper',  r_core=24, r_out=None,
                  sn_thre=2.5, count=None, mask_obj=None,
                  n_strip=48, wid_strip=30, dist_strip=None,
                  wid_cross=30, dist_cross=180, clean=True,
                  draw=True, save=False, save_dir='../output/pic'):
        
        """Make Strip + Cross Mask"""
        
        from .mask import Mask
        from .utils import crop_image
        
        masks = []
        
        for Image, stars in zip(self.Images,
                                stars_list):
            mask = Mask(Image, stars)
            
            # Mask the main object by given shape parameters or read a map
            mask.make_mask_object(mask_obj)
            
            # crop the full mask map into smaller one
            if hasattr(mask, 'mask_obj_field'):
                mask.mask_obj0 = crop_image(mask.mask_obj_field, Image.bounds0)
            else:
                mask.mask_obj0 = np.zeros(mask.shape, dtype=bool)

            # Primary SN threshold mask
            mask.make_mask_map_deep(dir_measure, by,
                                    r_core, r_out, count,
                                    obj_name=self.obj_name,
                                    band=self.band, 
                                    draw=draw, save=save, save_dir=save_dir)
            
            # Supplementary Strip + Cross mask
            if dist_strip is None:
                dist_strip = max(Image.image_shape) * self.pixel_scale
                
            mask.make_mask_advanced(n_strip, wid_strip, dist_strip,
                                    wid_cross, dist_cross, 
                                    clean=clean, draw=draw,
                                    save=save, save_dir=save_dir)

            masks += [mask]
                
        self.Masks = masks
        
        self.stars = [mask.stars_new for mask in masks]
    
    
    @property
    def mask_fit(self):
        """ Masks for fitting """
        return [mask.mask_fit for mask in self.Masks]
    
    @property
    def data(self):
        """ 1D array to be fit """
        data = [image[~mask_fit].copy().ravel()
                for (image, mask_fit) in zip(self.images, self.mask_fit)]

        return data
        self
        
        
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
    
    def fit_n0(self, dir_measure, **kwargs):
        """ Fit power index of first component with bright star profiles """
        self.n0, self.d_n0 = [], []
        for i in range(self.N_Image):
            if hasattr(self, 'std_est'):
                kwargs['sky_std'] = self.std_est[i]
            else:
                print('Sky stddev has not been estimated yet.')
            n0, d_n0 = self.Images[i].fit_n0(dir_measure, **kwargs)
            self.n0 += [n0]
            self.d_n0 += [d_n0]
            
    def set_container(self,
                      psf, stars,
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
            image_shape = self.Images[i].image_shape
            
            container = Container(n_spline, leg2d, 
                                  fit_sigma, fit_frac,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            
            if hasattr(self, '_n0'):
                # use fixed n0 is given
                n0, d_n0 = self._n0, 1e-2
            else:
                # get first component power index if fitted
                # if not fitted, n0 will be added to the prior
                n0 = getattr(self.Images[i],'n0', None)
                d_n0 = getattr(self.Images[i],'d_n0', 1e-2)
                
            if n0 is None:
                container.fix_n0 = False
                n0, d_n0 = psf.n0, 0.2
            else:
                container.fix_n0 = True
                
            if theta_in is None:
                theta_in = self.Masks[i].r_core_m * self.pixel_scale
                
            if theta_out is None:
                if psf.cutoff:
                    theta_out = psf.theta_c
                else:
                    theta_out = max(image_shape) * self.pixel_scale
            psf.theta_out = theta_out
                
            # Set Priors
            container.set_prior(n0, self.bkg, self.std_est[i],
                                n_min=n_min, d_n0=d_n0,
                                theta_in=theta_in, theta_out=theta_out)

            # Set Likelihood
            container.set_likelihood(self.images[i],
                                     self.mask_fit[i],
                                     psf, stars[i],
                                     n0=n0,
                                     psf_range=[None, None],
                                     norm='brightness',
                                     G_eff=self.G_eff,
                                     image_base=self.Images[i].image_base)
            
            # Set a few attributes to container for convenience
            container.image = self.images[i]
            container.data = self.data[i]
            container.mask = self.Masks[i]
            container.image_shape = image_shape
            
            self.containers += [container]


