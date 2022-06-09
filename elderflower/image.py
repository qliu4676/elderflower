import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
import astropy.units as u
from astropy.utils import lazyproperty
from photutils.segmentation import SegmentationImage

from .io import logger
from .mask import mask_param_default
from .plotting import display, AsinhNorm, colorbar
from . import DF_pixel_scale, DF_raw_pixel_scale

class ImageButler:
    """
    
    A class storing Image info.
    
    Parameters
    ----------
    
    hdu_path : str
        path of hdu data
    obj_name : str
        object name
    band : str, 'g' 'G' 'r' or 'R'
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
    
    def __init__(self, hdu_path,
                 obj_name='', band='G',
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
            if verbose: logger.info(f"Read Image: {hdu_path}")
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
    band : str, 'g' 'G' 'r' or 'R'
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
    @property
    def fwhm(self):
        """ FWHM in arcsec """
        return self.get_median("FWHM_IMAGE") * self.pixel_scale

    def get_median(self,
                   colname,
                   mag_range=[14, 16],
                   mag_name="MAG_AUTO_corr"):
        """ Return median value of SE measurements in the image """
        tab = self.table_norm
        cond = (tab[mag_name]>mag_range[0]) & (tab[mag_name]<mag_range[1])
        return np.median(tab[cond][colname])
        
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
                           mag_limit_segm=22,
                           make_segm=False, K=2,
                           catalog_sup='SE',
                           catalog_sup_atlas=None,
                           use_PS1_DR2=False,
                           subtract_external=True,
                           draw=False,
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
        check_save_path(dir_tmp, overwrite=True, verbose=False)
        
        band = self.band
        obj_name = self.obj_name
        bounds = self.bounds
        
        SE_cat = crop_catalog(SE_catalog, bounds)
        SE_cat_target, ext_cat, mag_saturate = identify_extended_source(SE_cat, draw=draw)
        
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
        
        CT = calculate_color_term(tab_target_full, mag_range=[mag_saturate,mag_limit+2],
                                  mag_name=mag_name_cat, draw=draw)
        
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
        catalog_star["MAG_AUTO_corr"] = catalog_star[mag_name] + CT #corrected mag
        
        catalog_star_name = os.path.join(dir_tmp, f'{obj_name}-catalog_PS_{band}_all.txt')
        catalog_star.write(catalog_star_name, overwrite=True, format='ascii')
        
        if catalog_sup == "ATLAS":
            tab_target = add_supplementary_atlas(tab_target, catalog_sup_atlas, SE_catalog,
                                                 mag_saturate=mag_saturate)
            
        elif catalog_sup == "SE":
            tab_target = add_supplementary_SE_star(tab_target, SE_cat_target,
                                                   mag_saturate=mag_saturate, draw=draw)
        self.tab_target = tab_target
        
        # Measure I at r0
        wcs, image = self.full_wcs, self.full_image
        width_cross = int(10/self.pixel_scale)
        tab_norm, res_thumb = measure_Rnorm_all(tab_target, bounds, wcs,
                                                image, seg_map,
                                                mag_limit=mag_limit,
                                                mag_saturate=mag_saturate,
                                                r_scale=r_scale,
                                                k_enlarge=2,
                                                width_cross=width_cross,
                                                obj_name=obj_name,
                                                mag_name=mag_name_cat,
                                                save=True, dir_name=dir_tmp,
                                                verbose=False)
        
        self.read_measurement_table(dir_tmp, r_scale=r_scale, mag_limit=mag_limit)
        
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
                                                mag_limit=mag_limit_segm,
                                                ext_cat=ext_cat,
                                                draw=draw,
                                                save=False,
                                                dir_name=dir_tmp)
            self.seg_map = seg_map_cat
        
        # Generate model star
        image_stars, _, _ = generate_image_fit(psf.copy(), stars.copy(),
        self.image_shape, subtract_external=subtract_external)
        logger.info("Image of stars has been generated based on the PSF Model!")
        
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
    band : str, 'g' 'G' 'r' or 'R'
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
            
        self.fwhm = np.mean([Img.fwhm for Img in self.Images])
    
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
    
    def make_mask(self, stars_list, dir_measure,
                  mask_param=mask_param_default,
                  save=False, save_dir='../output/pic',
                  draw=True, verbose=True):
        
        """
        Make deep mask, object mask, strip mask, and cross mask.
        
        The 'deep' mask is based on S/N.
        The object mask is for masking target sources such as large galaxies.
        The strip mask is spider-like, used to reduce sample size of pixels
         at large radii, equivalent to assign lower weights to outskirts.
        The cross mask is for masking stellar spikes of bright stars.
        
        Parameters
        ----------
        stars_list: list of modeling.Stars object
            List of Stars object.
        dir_measure : str
            Directory storing the measurement.
        mask_param: dict, optional
            Parameters setting up the mask map.
            mask_type : 'aper' or 'brightness', default 'aper'
                "aper": aperture-like masking
                "brightness": brightness-limit masking
            r_core : int or [int, int], optional, default 24
                Radius (in pix) for the inner mask of [very, medium]
                bright stars. Default is 1' for Dragonfly.
            r_out : int or [int, int] or None, optional, default None
                Radius (in pix) for the outer mask of [very, medium]
                bright stars. If None, turn off outer mask.
            sn_thre : float, optional, default 2.5
                SNR threshold used for deep mask.
            SB_threshold : float, optional, default 24.5
                Surface brightness upper limit for masking.
                Only used if mask_type = 'brightness'.
            mask_obj : str, optional
                Object mask file name. See mask.make_mask_object
            wid_strip : int, optional, default 24
                Width of strip in pixel for masks of very bright stars.
            n_strip : int, optional, default 48
                Number of strip for masks of very bright stars.
            dist_strip : float, optional, default None
                Range of each strip mask (in arcsec)
                If not given, use image size.
            wid_cross : float, optional, default 20
                Half-width of the spike mask (in arcsec).
            dist_cross: float, optional, default 180
                Range of each spike mask (in arcsec) (default: 3 arcmin)
            clean : bool, optional, default True
                Whether to remove medium bright stars far from any available
                pixels for fitting. A new Stars object will be stored in
                stars_new, otherwise it is simply a copy.
        draw : bool, optional, default True
            Whether to draw mask map
        save : bool, optional, default True
            Whether to save the image
        save_dir : str, optional
            Path of saving plot, default current.
        
        """
        
        from .mask import Mask
        from .utils import crop_image
        
        # S/N threshold of deep mask
        sn_thre = mask_param['sn_thre']
        
        # aper mask params
        mask_type = mask_param['mask_type']
        r_core = mask_param['r_core']
        r_out = mask_param['r_out']
        
        # strip mask params
        wid_strip = mask_param['wid_strip']
        n_strip = mask_param['n_strip']
        dist_strip = mask_param['dist_strip']
        
        # cross mask params
        wid_cross = mask_param['wid_cross']
        dist_cross = mask_param['dist_cross']
        
        if mask_type=='brightness':
            from .utils import SB2Intensity
            count = SB2Intensity(mask_param['SB_threshold'], self.bkg,
                                 self.ZP, self.pixel_scale)[0]
        else:
            count = None
            
        masks = []
        
        for i, (Image, stars) in enumerate(zip(self.Images, stars_list)):
            if verbose:
                logger.info("Prepare mask for region {}.".format(i+1))
                
            mask = Mask(Image, stars)
            
            # Read a map of object masks (e.g. large galaxies) or
            # create a new one with given shape parameters.
            # Note the object mask has full shape as SExtractor outputs
            mask.make_mask_object(mask_param['mask_obj'], wcs=self.full_wcs)
            
            if hasattr(mask, 'mask_obj_field'):
                # Crop the full object mask map into a smaller one
                mask.mask_obj0 = crop_image(mask.mask_obj_field, Image.bounds0)
            else:
                mask.mask_obj0 = np.zeros(mask.image_shape0, dtype=bool)

            # Primary SN threshold mask
            mask.make_mask_map_deep(dir_measure,
                                    mask_type,
                                    r_core, r_out,
                                    count=count,
                                    sn_thre=sn_thre,
                                    obj_name=self.obj_name,
                                    band=self.band, 
                                    draw=draw, save=save,
                                    save_dir=save_dir)
            
            # Supplementary Strip + Cross mask
            if dist_strip is None:
                dist_strip = max(Image.image_shape) * self.pixel_scale
                
            mask.make_mask_advanced(n_strip, wid_strip, dist_strip,
                                    wid_cross, dist_cross, 
                                    clean=mask_param['clean'],
                                    draw=draw, save=save, save_dir=save_dir)

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
        """ Estimate background level and std. """
        
        from astropy.stats import sigma_clip
        
        self.mu_est = np.zeros(len(self.Images))
        self.std_est = np.zeros(len(self.Images))
        
        for i, (Image, mask) in enumerate(zip(self.Images, self.mask_fit)):
            data_sky = sigma_clip(Image.image[~mask], sigma=3)
            
            mu_patch, std_patch = np.mean(data_sky), np.std(data_sky)
            
            self.mu_est[i] = mu_patch
            self.std_est[i] = std_patch
            
            msg = "Estimate of Background: ({0:.3g} +/- {1:.3g}) for "
            msg = msg.format(mu_patch, std_patch) + repr(Image)
            logger.info(msg)
    
    def fit_n0(self, dir_measure, N_min_fit=10, **kwargs):
        """ Fit power index of 1st component with bright stars. """
        self.n0, self.d_n0 = [], []
        for i in range(self.N_Image):
            if hasattr(self, 'std_est'):
                kwargs['sky_std'] = self.std_est[i]
            else:
                logger.warning('Sky stddev is not estimated.')
                
            N_fit = max(N_min_fit, self.stars[i].n_verybright)
            n0, d_n0 = self.Images[i].fit_n0(dir_measure, N_fit=N_fit, **kwargs)
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
                      n_min=1.1,
                      d_n0_min=0.1,
                      theta0_range=[50, 300],
                      method='nested',
                      verbose=True):
        """ Container for fit storing prior and likelihood function """
        
        from .container import Container
        
        self.containers = []
        
        for i in range(self.N_Image):
            if verbose:
                logger.info("Region {}:".format(i+1))
            image_shape = self.Images[i].image_shape
            
            container = Container(n_spline, leg2d, 
                                  fit_sigma, fit_frac,
                                  brightest_only=brightest_only,
                                  parallel=parallel, draw_real=draw_real)
            
            if hasattr(self, 'n0_'):
                # Use a given fixed n0
                n0, d_n0 = self.n0_, 0.1
                if verbose:
                    msg = "   - n0 is fixed to be a static value = {}.".format(n0)
                    logger.warning(msg)
            else:
                # Get first component power index if already fitted
                # Otherwise n0 will be added as a parameter in the prior
                n0 = getattr(self.Images[i],'n0', None)
                d_n0 = getattr(self.Images[i],'d_n0', 0.1)
            
            if (self.fix_n0 is False) | (n0 is None):
                container.fix_n0 = False
                d_n0 = max(d_n0, d_n0_min) # set a min dev for n0
                
                if n0 is None:  n0, d_n0 = psf.n0, 0.3  # rare case
                
                if verbose:
                    logger.info("   - n0 will be included in the full fitting.")
                
            else:
                container.fix_n0 = self.fix_n0
                if verbose:
                    msg = "   - n0 will not be included in the full fitting."
                    msg += " Adopt fitted value n0 = {:.3f}.".format(n0)
                    logger.info(msg)
            
            theta_in, theta_out = theta0_range
            
            if theta_in is None:
                theta_in = self.Masks[i].r_core_m * self.pixel_scale
                
            if theta_out is None:
                if psf.cutoff:
                    theta_out = psf.theta_c
                else:
                    theta_out = int(0.8 * max(image_shape) * self.pixel_scale)
            psf.theta_out = theta_out
            
            logger.info("theta_in = {:.2f}, theta_out = {:.2f}".format(theta_in, theta_out))
            
            # Set priors (Bayesian) or bounds (MLE)
            prior_kws = dict(n_min=n_min, d_n0=d_n0,
                             theta_in=theta_in, theta_out=theta_out)
                             
            if method == 'mle':
                # Set bounds on variables
                container.set_MLE_bounds(n0, self.bkg, self.std_est[i], **prior_kws)
            else:
                # Set Priors
                container.set_prior(n0, self.bkg, self.std_est[i], **prior_kws)

            # Set Likelihood
            container.set_likelihood(self.images[i],
                                     self.mask_fit[i],
                                     psf, stars[i],
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


class Thumb_Image:
    """
    A class for operation and info storing of a thumbnail image.
    Used for measuring scaling and stacking.
    
    row: astropy.table.row.Row
        Astropy table row.
    wcs: astropy.wcs.wcs
        WCS of image.
    """

    def __init__(self, row, wcs):
        self.wcs = wcs
        self.row = row
        
    def make_star_thumb(self,
                        image, seg_map=None,
                        n_win=20, seeing=2.5, max_size=200,
                        origin=1, verbose=False):
        """
        Crop the image and segmentation map into thumbnails.

        Parameters
        ----------
        image : 2d array
            Full image
        seg_map : 2d array
            Full segmentation map
        n_win : int, optional, default 20
            Enlarge factor (of fwhm) for the thumb size
        seeing : float, optional, default 2.5
            Estimate of seeing FWHM in pixel
        max_size : int, optional, default 200
            Max thumb size in pixel
        origin : 1 or 0, optional, default 1
            Position of the first pixel. origin=1 for SE convention.
            
        """
        
        from .utils import coord_Im2Array

        # Centroid in the image from the SE measurement
        # Note SE convention is 1-based (differ from photutils)
        X_c, Y_c = self.row["X_IMAGE"], self.row["Y_IMAGE"]

        # Define thumbnail size
        fwhm =  max(self.row["FWHM_IMAGE"], seeing)
        win_size = min(int(n_win * max(fwhm, 2)), max_size)

        # Calculate boundary
        X_min, X_max = max(origin, X_c - win_size), min(image.shape[1], X_c + win_size)
        Y_min, Y_max = max(origin, Y_c - win_size), min(image.shape[0], Y_c + win_size)
        x_min, y_min = coord_Im2Array(X_min, Y_min, origin) # py convention
        x_max, y_max = coord_Im2Array(X_max, Y_max, origin)

        X_WORLD, Y_WORLD = self.row["X_WORLD"], self.row["Y_WORLD"]

        if verbose:
            print("NUMBER: ", self.row["NUMBER"])
            print("X_c, Y_c: ", (X_c, Y_c))
            print("RA, DEC: ", (X_WORLD, Y_WORLD))
            print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
            print("X_min, X_max, Y_min, Y_max: ", X_min, X_max, Y_min, Y_max)

        # Crop
        self.img_thumb = image[x_min:x_max, y_min:y_max].copy()
        if seg_map is None:
            self.seg_thumb = None
            self.mask_thumb = np.zeros_like(self.img_thumb, dtype=bool)
        else:
            self.seg_thumb = seg_map[x_min:x_max, y_min:y_max]
            self.mask_thumb = (self.seg_thumb!=0) # mask sources

        # Centroid position in the cutout (0-based py convention)
        #self.cen_star = np.array([X_c - X_min, Y_c - Y_min])
        self.cen_star = np.array([X_c - y_min - origin, Y_c - x_min - origin])

    def extract_star(self, image,
                     seg_map=None,
                     sn_thre=2.5,
                     display_bkg=False,
                     display=False, **kwargs):
        
        """
        Local background and segmentation.
        If no segmentation map provided, do a local detection & deblend
        to remove faint undetected source.
        
        Parameters
        ----------
        image : 2d array
            Full image
        seg_map : 2d array
            Full segmentation map
        sn_thre : float, optional, default 2.5
            SNR threshold used for detection if seg_map is None
        display_bkg : bool, optional, default False
            Whether to display background measurment
        display : bool, optional, default False
            Whether to display detection & deblend around the star
        
        """
        from .utils import (background_extraction,
                            detect_sources, deblend_sources)
        # Make thumbnail image
        self.make_star_thumb(image, seg_map, **kwargs)
        
        img_thumb = self.img_thumb
        seg_thumb = self.seg_thumb
        mask_thumb = self.mask_thumb
        
        # Measure local background, use constant if the thumbnail is small
        shape = img_thumb.shape
        b_size = round(min(shape)//5/25)*25
        
        if shape[0] >= b_size:
            back, back_rms = background_extraction(img_thumb, b_size=b_size)
        else:
            im_ = np.ones_like(img_thumb)
            img_thumb_ma = img_thumb[~mask_thumb]
            back, back_rms = (np.median(img_thumb_ma)*im_,
                              mad_std(img_thumb_ma)*im_)
        self.bkg = back
        self.bkg_rms = back_rms
        
        if display_bkg:
            # show background subtraction
            from .plotting import display_background
            display_background(img_thumb, back)
                
        if seg_thumb is None:
            # do local source detection to remove faint stars using photutils
            threshold = back + (sn_thre * back_rms)
            segm = detect_sources(img_thumb, threshold, npixels=5)

            # deblending using photutils
            segm_deb = deblend_sources(img_thumb, segm, npixels=5,
                                           nlevels=64, contrast=0.005)
        else:
            segm_deb = SegmentationImage(seg_thumb)
            
        # mask other sources in the thumbnail
        star_label = segm_deb.data[round(self.cen_star[1]), round(self.cen_star[0])]
        star_ma = ~((segm_deb.data==star_label) | (segm_deb.data==0))
        self.star_ma = star_ma
        
        if display:
            from .plotting import display_source
            display_source(img_thumb, segm_deb, star_ma, back)
            
            
    def compute_Rnorm(self, R=12, **kwargs):
        """
        Compute the scaling factor at R using an annulus.
        Note the output values include the background level.
        
        Paramters
        ---------
        R : int, optional, default 12
            radius in pix at which the scaling factor is meausured
        kwargs : dict
            kwargs passed to compute_Rnorm
        
        """
        from .utils import compute_Rnorm
        I_mean, I_med, I_std, I_flag = compute_Rnorm(self.img_thumb,
                                                     self.star_ma,
                                                     self.cen_star,
                                                     R=R, **kwargs)
        self.I_mean = I_mean
        self.I_med = I_med
        self.I_std = I_std
        self.I_flag = I_flag
        
        # Use the median of background as the local background
        self.I_sky = np.median(self.bkg)
