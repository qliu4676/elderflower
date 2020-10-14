import os
import math
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from .modeling import Stars
from .utils import background_sub_SE, crop_pad
from .image import DF_pixel_scale

class Mask:
    """ Class for masking sources """
    
    def __init__(self, Image, stars):
        """
        
        Parameters
        ----------
        Image : an Image class
        stars : a Star object  
        
        """
        self.Image = Image
        self.stars = stars
        self.image0 = Image.image0
        self.shape = Image.image0.shape
        self.pixel_scale = Image.pixel_scale
        
        self.bounds0 = Image.bounds0
        self.image_shape = Image.image_shape
        self.Ximage_size = Image.image_shape[1]
        self.Yimage_size = Image.image_shape[0]
        self.pad = Image.pad
        
        self.yy, self.xx = np.mgrid[:self.Yimage_size + 2 * self.pad,
                                    :self.Ximage_size + 2 * self.pad]
        
        self.pad = Image.pad
        self.bkg = Image.bkg
    
    def __str__(self):
        return "A Mask Class"

    def __repr__(self):
        return f"{self.__class__.__name__} for {repr(self.Image)}"

    
    @property
    def mask_base(self):
        mask_base0 = getattr(self, 'mask_base0', self.mask_deep0)
        return crop_pad(mask_base0, self.pad)
    
    @property
    def seg_base(self):
        seg_base0 = getattr(self, 'seg_base0', self.seg_deep0)
        return crop_pad(seg_base0, self.pad)
    
    @property
    def mask_deep(self):
        return crop_pad(self.mask_deep0, self.pad)
    
    @property
    def seg_deep(self):
        return crop_pad(self.seg_deep0, self.pad)
    
    @property
    def mask_comb(self):
        return crop_pad(self.mask_comb0, self.pad)
    
    @property
    def seg_comb(self):
        return crop_pad(self.seg_comb0, self.pad)
        
    @property
    def mask_fit(self):
        """ Mask for fit """
        return getattr(self, 'mask_comb', self.mask_deep)
    
    
    def make_mask_object(self, mask_obj=None):
        """ Read mask of objects w/ ellip apertures given shape parameters
#        Parameters
#        ----------
#        obj_name : object name
#        k : float, enlargement factor
#
#        Notes
#        -----
#        if {obj_name}_maskobj.fits exists, use as mask
#        otherwise find {obj_name}_shape.txt and make new masl
#        the txt should record following parameters in rows:
#            pos : turple or array or turples
#                position(s) (x,y) of apertures
#            a_ang : float or 1d array
#                semi-major axis length(s) in arcsec
#            b_ang : float or 1d array
#                semi-minor axis length(s) in arcsec
#            PA_ang : float or 1d array
#                patch angle (ccw from north) in degree

        """
        
        if mask_obj is not None:
            
            if os.path.isfile(mask_obj):
                print("Read mask map of objects: ", os.path.abspath(mask_obj))
                # read existed mask map
                self.mask_obj_field = fits.getdata(mask_obj).astype(bool)
            else:
                "Object mask not found. Skip."
            
#        else:
#            file_obj = '%s_shape.txt'%obj_name
#
#            if os.path.isfile(file_obj):
#                print("Read shape parameters of objects: ",
#                os.path.abspath(file_obj))
#                # read shape parameters from file
#                par = np.atleast_2d(np.loadtxt(file_obj_pars))
#
#                pos = par[:,:1]
#                a_ang, b_ang, PA_ang = par[:,2], par[:,3], par[:,4]
#
#                # make mask map with parameters
#                self.mask_obj_field = make_mask_aperture(pos, a_ang, b_ang,
#                                                        PA_ang, self.shape,
#                                                        k, self.pixel_scale)
                
        
    def make_mask_map_deep(self, dir_measure=None, by='aper',
                           r_core=None, r_out=None, count=None,
                           draw=True, save=False, save_dir='.', 
                           obj_name='', band='G', *args, **kwargs):
        """
        Make deep mask map of bright stars based on either of:
        (1) aperture (2) brightness
        The mask map is then combined with a base segm map (if given) (for masking sources below S/N threshold) and a S_N seg map (for masking bright sources/features not contained in the catalog)
        
        Parameters
        ----------
        by : mask type
            "aper": aperture-like masking
            "brightness": brightness-limit masking
        r_core : core radius of [medium, very bright] stars to be masked
        count : absolute count (in ADU) above which is masked        
        obj_name : name of object
        band : filter name. r/R/G/g for DF.
        draw : whether to draw mask map
        save : whether to save the image
        save_dir : path of saving
        
        """
        
        image0 = self.image0
        stars = self.stars
        pad = self.pad
        
        mag_name = band.lower() + 'mag'
        
        if dir_measure is not None:
            patch_Xmin0, patch_Ymin0, patch_Xmax0, patch_Ymax0 = self.bounds0
            fname = "%s-segm_%s_catalog_X[%d-%d]Y[%d-%d].fits"\
                     %(obj_name, mag_name, patch_Xmin0, patch_Xmax0, patch_Ymin0, patch_Ymax0)
            fname_seg_base = os.path.join(dir_measure, fname)
            print("Read mask map built from catalog: ", fname_seg_base)
            # Try finding basement segment map generated by catalog
            
            if os.path.isfile(fname_seg_base) is False:
                print("%s doe not exist. Use source detection only."%fname_seg_base)
                seg_base0 = None
            else:
                seg_base0 = fits.getdata(fname_seg_base)
                self.seg_base0 = seg_base0
                self.mask_base0 = seg_base0 > 0
        else:
            seg_base0 = None
        
        # S/N + Core mask
        mask_deep0, seg_deep0 = make_mask_map_dual(image0, stars, self.xx, self.yy,
                                                   by=by, pad=pad, seg_base=seg_base0,
                                                   r_core=r_core, r_out=r_out, count=count, 
                                                   n_bright=stars.n_bright, *args, **kwargs)
        
        # combine with object mask
        mask_obj0 = self.mask_obj0
        mask_deep0 = mask_deep0 & mask_obj0
        seg_deep0[mask_obj0] = seg_deep0.max() + 1
        
        self.mask_deep0 = mask_deep0
        self.seg_deep0 = seg_deep0
        
        self.r_core = r_core
        self.r_core_m = max(np.unique(r_core))
        
        self.count = count
        
        # Display mask
        if draw:
            from .plotting import draw_mask_map
            draw_mask_map(image0, seg_deep0, mask_deep0, stars,
                          pad=pad, r_core=r_core, r_out=r_out,
                          save=save, save_dir=save_dir)
            
            
    def make_mask_advanced(self, n_strip=48,
                           wid_strip=16, dist_strip=800,
                           wid_cross=10, dist_cross=72,
                           clean=True, draw=True, 
                           save=False, save_dir='.'):
        
        """
        Make spider-like mask map and mask stellar spikes for bright stars.
        The spider-like mask map is to reduce sample size of pixels at large
        radii, equivalent to assign lower weights to outskirts.
        Note: make_mask_map_deep() need to be run first.
        
        Parameters
        ----------
        n_strip : number of each strip mask
        wid_strip : width of each strip mask
        dist_strip : furthest range of each strip mask
        wid_cross : half-width of spike mask
        dist_cross: furthest range of each spike mask
        clean : whether to remove medium bright stars far from any available
                pixels for fitting. A new Stars object will be stored in
                stars_new, otherwise it is simply a copy.
        draw : whether to draw mask map
        save : whether to save the image
        save_dir : path of saving
        
        """
        
        if hasattr(self, 'mask_deep0') is False:
            return None
        
        image0 = self.image0
        stars = self.stars
        
        pad = self.pad 
        
        # Strip + Cross mask
        mask_strip_s, mask_cross_s =  make_mask_strip(stars, self.xx, self.yy,
                                                      pad=pad, n_strip=n_strip,
                                                      wid_strip=wid_strip, dist_strip=dist_strip,
                                                      wid_cross=wid_cross, dist_cross=dist_cross)

        # combine strips
        mask_strip_all = ~np.logical_or.reduce(mask_strip_s)
        mask_cross_all = ~np.logical_or.reduce(mask_cross_s)
        
        seg_deep0 = self.seg_deep0
        
        # combine deep, crosses and strips
        seg_comb0 = seg_deep0.copy()
        ma_extra = (mask_strip_all|~mask_cross_all) & (seg_deep0==0)
        seg_comb0[ma_extra] = seg_deep0.max()-2
        mask_comb0 = (seg_comb0!=0)
        
        # assign attribute
        self.mask_comb0 = mask_comb0
        self.seg_comb0 = seg_comb0
        
        # Clean medium bright stars far from bright stars
        if clean:
            from .utils import clean_isolated_stars
            clean = clean_isolated_stars(self.xx, self.yy, mask_comb0,
                                       stars.star_pos, pad=pad)
            if stars.n_verybright > 0:
                clean[stars.Flux >= stars.F_verybright] = False
            
            z_norm_clean = stars.z_norm[~clean] if hasattr(stars, 'z_norm') else None
            stars_new = Stars(stars.star_pos[~clean], stars.Flux[~clean],
                              stars.Flux_threshold, z_norm=z_norm_clean,
                              r_scale=stars.r_scale, BKG=stars.BKG, verbose=False)
            self.stars_new = stars_new
            
        else:
            self.stars_new = stars.copy()
            
        # Display mask
        if draw:
            from .plotting import draw_mask_map_strip
            draw_mask_map_strip(image0, seg_comb0, mask_comb0, stars_new,
                                pad=pad, r_core=self.r_core, 
                                ma_example=[mask_strip_s[0], mask_cross_s[0]],
                                save=save, save_dir=save_dir)
            
            
# Make mask maps

def make_mask_detection(image, sn_thre=3, b_size=25, npix=5):
    """ Make mask map with S/N > sn_thre """
    from photutils import detect_sources, deblend_sources
    
    # detect source
    back, back_rms = background_sub_SE(image, b_size=b_size)
    threshold = back + (sn_thre * back_rms)
    segm0 = detect_sources(image, threshold, npixels=npix)
    
    # dilation
    segmap = segm0.data.copy()
        
    segmap[(segmap!=0)&(segm0.data==0)] = segmap.max()+1
    mask_deep = (segmap!=0)
    
    return mask_deep, segmap


def make_mask_aperture(fname, RA, Dec, A_ang, B_ang, PA_ang, wcs, shape,
                       enlarge=3, pixel_scale=DF_pixel_scale):
    
    """
    
    Make mask map with elliptical apertures.
    
    Parameters
    ----------
    fname : name of saved mask
    RA, Dec : float or 1d array
        coordinates(s) of aperture centers
    A_ang, B_ang : float or 1d array
        semi-major/minor axis length(s) in arcsec
    PA_ang : float or 1d array
        patch angle (counter-clockwise from north) in degree
    wcs : astropy wcs
    shape : image shape
    enlarge : float, enlargement factor
    pixel_scale : pixel scale in arcsec/pixel
    
    Returns
    ----------
    mask : 2d array mask map (masked area = 1)
    
    """
    
    from photutils import EllipticalAperture
    
    mask = np.zeros(shape)
    
    # shape properties of apertures
    aper_props = np.atleast_2d(np.array([RA, Dec, A_ang, B_ang, PA_ang]).T)
    
    for ra, dec, a_ang, b_ang, pa_ang in aper_props:
        
        # convert coordinates to positions
        coords = SkyCoord(f'{ra} {dec}', unit=u.deg)
        pos = wcs.all_world2pix(ra, dec, 0) # 0-original in photutils
        
        # convert angular to pixel unit
        a_pix = a_ang / pixel_scale
        b_pix = b_ang / pixel_scale

        # correct PA to theta in photutils (from +x axis)
        theta =  np.mod(pa_ang+90, 360) * np.pi/180

#        pos = (shape[1]-1)/2., (shape[0]-1)/2.

        # make elliptical aperture
        aper = EllipticalAperture(pos, enlarge*a_pix, enlarge*b_pix, theta)

        # convert aperture to mask
        ma_aper = aper.to_mask(method='center')
        ma = ma_aper.to_image(shape).astype(bool)
        
        mask[ma] = 1.0
        
    fits.writeto(fname, mask, overwrite=True)
    
    return mask


def make_mask_map_core(image, star_pos, r_core=12):
    """ Make stars out to r_core """

    # mask core
    yy, xx = np.indices(image.shape)
    mask_core = np.zeros_like(image, dtype=bool)
    
    if np.ndim(r_core) == 0:
        r_core = np.ones(len(star_pos)) * r_core
    
    core_region= np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r for (pos,r) in zip(star_pos,r_core)])
    
    mask_core[core_region] = 1
    segmap = mask_core.astype(int).copy()
    
    return mask_core, segmap


def make_mask_map_dual(image, stars,
                       xx=None, yy=None, by='aper',
                       pad=0, r_core=24, r_out=None, count=None,
                       seg_base=None, n_bright=25, sn_thre=3, 
                       nlevels=64, contrast=0.001, npix=4, b_size=64):
    """ Make mask map in dual mode: for faint stars, mask with S/N > sn_thre;
    for bright stars, mask core (r < r_core pix) """
    from photutils import detect_sources, deblend_sources
    from photutils.segmentation import SegmentationImage
    
    if (xx is None) | (yy is None):
        yy, xx = np.mgrid[:image.shape[0]+2*pad, :image.shape[1]+2*pad]
        
    star_pos = stars.star_pos_bright + pad
    
    if by == 'aper':
        if len(np.unique(r_core)) == 1:
            r_core_A, r_core_B = r_core, r_core
            r_core_s = np.ones(len(star_pos)) * r_core
        else:
            r_core_A, r_core_B = r_core[:2]
            r_core_s = np.array([r_core_A if F >= stars.F_verybright else r_core_B
                                 for F in stars.Flux_bright])

        if r_out is not None:
            if len(np.unique(r_out)) == 1:
                r_out_A, r_out_B = r_out, r_out
                r_out_s = np.ones(len(star_pos)) * r_out_s
            else:
                r_out_A, r_out_B = r_out[:2]
                r_out_s = np.array([r_out_A if F >= stars.F_verybright else r_out_B
                                     for F in stars.Flux_bright])
            print("Mask outer regions: r > %d (%d) pix "%(r_out_A, r_out_B))
            
    if sn_thre is not None:
        print("Detect and deblend source... Mask S/N > %.1f"%(sn_thre))
        # detect all source first 
        back, back_rms = background_sub_SE(image, b_size=b_size)
        threshold = back + (sn_thre * back_rms)
        segm0 = detect_sources(image, threshold, npixels=npix)

        # deblend source
        segm_deb = deblend_sources(image, segm0, npixels=npix,
                                   nlevels=nlevels, contrast=contrast)

    #     for pos in star_pos:
    #         if (min(pos[0],pos[1]) > 0) & (pos[0] < image.shape[0]) & (pos[1] < image.shape[1]):
    #             star_lab = segmap[coord_Im2Array(pos[0], pos[1])]
    #             segm_deb.remove_label(star_lab)

        segmap = segm_deb.data.copy()
        max_lab = segm_deb.max_label

        # remove S/N mask map for input (bright) stars
        for pos in star_pos:
            rr2 = (xx-pos[0])**2+(yy-pos[1])**2
            lab = segmap[np.where(rr2==np.min(rr2))][0]
            segmap[segmap==lab] = 0
            
    if seg_base is not None:
        segmap2 = seg_base
        
        if sn_thre is not None:
            # Combine Two mask
            segmap[segmap2>n_bright] = max_lab + segmap2[segmap2>n_bright]
            segm_deb = SegmentationImage(segmap)
        else:
            # Only use seg_base, bright stars are aggressively masked
            segm_deb = SegmentationImage(segmap2)
        
        max_lab = segm_deb.max_label
    
    if by == 'aper':
        # mask core for bright stars out to given radii
        print("Mask core regions: r < %d (%d) pix "%(r_core_A, r_core_B))
        core_region = np.logical_or.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) < r
                                            for (pos,r) in zip(star_pos,r_core_s)])
        mask_star = core_region.copy()

        if r_out is not None:
            # mask outer region for bright stars out to given radii
            outskirt = np.logical_and.reduce([np.sqrt((xx-pos[0])**2+(yy-pos[1])**2) > r
                                             for (pos,r) in zip(star_pos,r_out_s)])
            mask_star = (mask_star) | (outskirt)
    
    elif by == 'brightness':
        # If count is not given, use 5 sigma above background.
        if count is None:
            count = np.mean(back + (5 * back_rms))
        # mask core for bright stars below given ADU count
        print("Mask core regions: Count > %.2f ADU "%count)
        mask_star = image >= count
        
    segmap[mask_star] = max_lab+1
    
    # set dilation border a different label (for visual)
    segmap[(segmap!=0)&(segm_deb.data==0)] = max_lab+2
    
    # set mask map
    mask_deep = (segmap!=0)
    
    return mask_deep, segmap


def make_mask_strip(stars, xx, yy, pad=0, n_strip=24,
                    wid_strip=16, dist_strip=800,
                    wid_cross=10, dist_cross=72):    
    """ Make mask map in strips with width=width """
    
    print("Use sky strips crossing very bright stars")
    
    if stars.n_verybright>0:
        mask_strip_s = np.empty((stars.n_verybright, xx.shape[0], xx.shape[1]))
        mask_cross_s = np.empty_like(mask_strip_s)
    else:
        return None, None
    
    star_pos = stars.star_pos_verybright + pad
    
    phi_s = np.linspace(-90, 90, n_strip+1)
    a_s = np.tan(phi_s*np.pi/180)
    
    for k, (x_b, y_b) in enumerate(star_pos):
        m_s = (y_b-a_s*x_b)
        mask_strip = np.logical_or.reduce([abs((yy-a*xx-m)/math.sqrt(1+a**2)) < wid_strip 
                                           for (a, m) in zip(a_s, m_s)])
        mask_cross = np.logical_or.reduce([abs(yy-y_b)<wid_cross, abs(xx-x_b)<wid_cross])
        dist_map1 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_strip
        dist_map2 = np.sqrt((xx-x_b)**2+(yy-y_b)**2) < dist_cross
        mask_strip_s[k] = mask_strip & dist_map1
        mask_cross_s[k] = mask_cross & dist_map2

    return mask_strip_s, mask_cross_s
