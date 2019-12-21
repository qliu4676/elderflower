#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preparatory script doing the following tasks:
1) Crossmatch a source table (e.g. from SExtractor, SEP, etc.) with star catalog(s), by default using PANSTARRS DR2 catalog. A new table will be saved which contains corrected source magnitude (e.g. MAG_AUTO).
2) Make enlarged segmentation map(s) for sub-region(s) from the star catalog (by default < 23 mag).
3) Measure normalization (i.e. flux scaling) at a certain radius for bright stars (by default < 15 mag).

> Parameter
[-f][--FILTER]: filter of image to be crossmatched. g/G/r/R for Dragonfly.
[-b][--IMAGE_BOUNDS]: bounds of region(s) to be processed in pixel coordinate. [Xmin, Ymin, Xmax, Ymax],[...],...
[-r][--R_SCALE]: radius at which normalziation is measured, in pixel.
[-I][--IMAGE]: path of image.
[-C][--SE_CATALOG]: path of source table containing following columns: "NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD", "MAG_AUTO".
[-S][--SEGMAP]: path of segmentation map corresponding to SE_CATALOG.
[--PS]: if set, use PANSTARRS DR2 API to do crossmatch.
[-m][--MAG_THRESHOLD]: magnitude threshold below which normalization of stars are measured (default: 15).
[-W][--WEIGHT]: path of weight map used in source extraction. Optional.
[--DIR_NAME]: directory name for saving outputs.

> Example Usage
1. In jupyter notebook / lab
%matplotlib inline
%run Measure_Rnorm.py -f "G" -r 12 -b "[3000, 1300, 4000, 2300]"
2. In bash
python Measure_Rnorm.py -f "G" -r 12 -b "[3000, 1300, 4000, 2300]"

"""

import sys
import getopt
from utils import *
from modeling import *
from plotting import *

def main(argv):  
    # Default Parameters
    band = "G"
    save, draw = True, True
    use_PS1_DR2 = False
    mag_thre, r_scale = 15, 12
    
    image_bounds = [[800, 1600, 1800, 2600],
                    [3100, 1400, 4100, 2400]]  # in image coords
    
    try:
        optlists, args = getopt.getopt(argv, "f:b:m:r:I:C:S:W:",
                                       ["FILTER=", "IMAGE_BOUNDS=", "IMAGE=", 
                                        "MAG_THRESHOLD=", "R_SCALE=", "PS",
                                        "SE_CATALOG=", "SEGMAP=", "WEIGHT_MAP=", "DIR_NAME="])
        opts = [opt for opt, arg in optlists]        
        
    except getopt.GetoptError:
        sys.exit('Wrong Option.')
    
    for opt, arg in optlists:
        if opt in ("-f", "--FILTER"):
            if arg in ["G", "R", "r", "g"]:
                band = arg.upper()
            else:
                sys.exit("Filter Not Available.")
                
    # Default Path
    hdu_path = "./data/coadd_Sloan%s_NGC_5907.fits"%band
    seg_map = "./SE_APASS/coadd_Sloan%s_NGC_5907_seg.fits"%band
    SE_catalog = "./SE_APASS/coadd_Sloan%s_NGC_5907.cat"%band
    weight_map = "./SE_APASS/weight_NGC5907.fits"
    dir_name = './Measure'
        
    for opt, arg in optlists:
        if opt in ("-I", "--IMAGE"):
            hdu_path = arg
        elif opt in ("-b", "--IMAGE_BOUNDS"):    
            image_bounds = np.array(re.findall(r'\d+', arg), dtype=int).reshape(-1,4)
        elif opt in ("-r", "--R_SCALE"):
            r_scale = np.float(arg)
        elif opt in ("-m", "--MAG_THRESHOLD"):
            mag_thre = np.float(arg)
        elif opt in ("-C", "--SE_CATALOG"):
            SE_catalog = arg
        elif opt in ("-S", "--SEGMAP"):
            seg_map = arg
        elif opt in ("-W", "--WEIGHT_MAP"):
            weight_map = arg
        elif opt in ("--DIR_NAME"):
            dir_name = arg
            
    if ("--PS" in opts): use_PS1_DR2 = True 
        
    check_save_path(dir_name, make_new=False)

    # Run Measurement
    Match_Mask_Measure(hdu_path, seg_map, SE_catalog, image_bounds,
                       weight_map=weight_map, band=band,
                       r_scale=r_scale, mag_thre=mag_thre,
                       draw=draw, use_PS1_DR2=use_PS1_DR2,
                       save=save, dir_name=dir_name)
    return opts
    
    
def Match_Mask_Measure(hdu_path,
                       seg_map,
                       SE_catalog,
                       image_bounds,
                       weight_map=None,
                       band="G",
                       r_scale=12,
                       mag_thre=15,
                       draw=True,
                       save=True,
                       use_PS1_DR2 = True,
                       dir_name='./Measure'):
    
    print("Measure the intensity at R = %d for stars < %.1f as normalization of fitting\n"%(r_scale, mag_thre))
    b_name = band.lower()
    mag_name = b_name + 'mag'
    obj_name = 'NGC5907-' + band
    image_bounds = np.atleast_2d(image_bounds)
    
    ############################################
    # Read and Display
    ############################################

    # Read hdu
    if os.path.isfile(hdu_path) is False:
        sys.exit("Image does not exist. Check path.")
    with fits.open(hdu_path) as hdul:
        print("Read Image :", hdu_path)
        data = hdul[0].data
        header = hdul[0].header
        wcs_data = wcs.WCS(header)

    # Read output from create_photometric_light_APASS 
    seg_map = fits.getdata(seg_map)
    SE_cat_full = Table.read(SE_catalog, format="ascii.sextractor")
    
    if weight_map is not None:
        weight_edge = fits.getdata(weight_map)

    # Read global background model zero point and pixel scale from header
    try:
        BKG, ZP, pix_scale = np.array([header["BACKVAL"], header["REFZP"],
                                       header["PIXSCALE"]]).astype(float)
        
        std = mad_std(data[seg_map==0&(weight_edge>0.5)]) 
        print("BACKVAL: %.2f , stddev: %.2f , ZP: %.2f , PIXSCALE: %.2f\n" %(BKG, std, ZP, pix_scale))

    except KeyError:
        print("BKG / ZP / PIXSCALE missing in header --->")
        ZP = np.float(input("Input a value of ZP :"))
        BKG = np.float(input("Manually set a value of background :"))
        std = np.float(input("Manually set a value of background RMS :"))
        data += BKG
        
    # Convert SE measured flux into mag
    flux = SE_cat_full["FLUX_AUTO"]
    mag = -2.5 * np.ma.log10(flux).filled(flux[flux>0].min()) + ZP
#     mag = -2.5 * np.log10(flux) + ZP
    SE_cat_full["MAG_AUTO"] = np.around(mag, 5)
    
    field_bounds = [700, 700, data.shape[1]-700, data.shape[0]-700]
    if not use_PS1_DR2:
        print("Match field %r with catalog\n"%field_bounds)
    
    print("Measure Sky Patch (X min, Y min, X max, Y max) :")
    [print("%r"%b) for b in image_bounds.tolist()]
    
    # Display field_bounds and sub-regions to be matched
    _, _ = crop_image(data, field_bounds, seg_map,
                      weight_map=weight_edge, sub_bounds=image_bounds, draw=draw)

    ############################################
    # Crossmatch with Star Catalog (across the field)
    ############################################

    # Crossmatch with PANSTRRS at threshold of mag_thre mag
    if use_PS1_DR2:
        tab_target, tab_target_full, catalog_star = \
                                cross_match_PS1_DR2(wcs_data, SE_cat_full, image_bounds,
                                                    mag_thre=mag_thre, band='g')        
    else:
        tab_target, tab_target_full, catalog_star = \
                                cross_match(wcs_data, SE_cat_full, image_bounds,
                                            mag_thre=mag_thre, mag_name=mag_name)

    # Calculate color correction between PANSTARRS and DF filter
    if use_PS1_DR2:
        mag_name = mag_name_cat = b_name+'MeanPSFMag'
    else:
        mag_name_cat = mag_name+'_PS'
        
    CT = calculate_color_term(tab_target_full, mag_name=mag_name_cat, draw=draw)
    catalog_star["MAG_AUTO"] = catalog_star[mag_name] + CT
    
    # Save matched table and catalog
    if save:
        tab_target_name = os.path.join(dir_name,
                                       '%s-catalog_match_%s%dmag.txt'%(obj_name, b_name, mag_thre))
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name_cat] + CT
        tab_target.write(tab_target_name, overwrite=True, format='ascii')

        catalog_star_name = os.path.join(dir_name,
                                         '%s-catalog_PS_%s_all.txt'%(obj_name, b_name))
        catalog_star["FLUX_AUTO"] = 10**((catalog_star["MAG_AUTO"]-ZP)/(-2.5))
        catalog_star.write(catalog_star_name, overwrite=True, format='ascii')
        
        print('Save the PANSTARRS catalog and matched sources in %s'%dir_name)
        
    ############################################
    # Build Mask & Measure Scaling (in selected patch)
    ############################################
    
    # Empirical enlarged aperture size from magnitude based on matched SE detection
    estimate_radius = fit_empirical_aperture(tab_target_full, seg_map, mag_name=mag_name_cat,
                                             mag_range=[13,20], K=2.5, degree=3, draw=draw)
    
    for image_bound in image_bounds:
        
        # Crop the star catalog and matched SE catalog
        patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = image_bound
                                         
        # Catalog bound slightly wider than the region
        cat_bound = (patch_Xmin-50, patch_Ymin-50, patch_Xmax+50, patch_Ymax+50)

        catalog_star_patch = crop_catalog(catalog_star, cat_bound, sortby=mag_name,
                                          keys=("X_IMAGE"+'_PS', "Y_IMAGE"+'_PS'))
        tab_target_patch = crop_catalog(tab_target, cat_bound, sortby=mag_name_cat,
                                        keys=("X_IMAGE", "Y_IMAGE"))

        # Make segmentation map from catalog based on SE seg map of one band
        seg_map_cat = make_segm_from_catalog(catalog_star_patch, image_bound, estimate_radius,
                                             mag_name=mag_name, cat_name='PS',
                                             draw=draw, save=save, dir_name=dir_name)

        # Measure average intensity (source+background) at e_scale
        print("Measure intensity at R = %d for catalog stars %s < %.1f in %r:"\
              %(r_scale, mag_name, mag_thre, image_bound))
        tab_res_Rnorm, res_thumb = measure_Rnorm_all(tab_target_patch, image_bound,
                                                     wcs_data, data, seg_map, 
                                                     r_scale=r_scale, mag_thre=mag_thre, 
                                                     obj_name=obj_name, mag_name=mag_name_cat, 
                                                     read=False, save=save, dir_name=dir_name)
        plot_bright_star_profile(tab_target_patch, tab_res_Rnorm, res_thumb, bkg_sky=BKG, std_sky=std)
        

if __name__ == "__main__":
    main(sys.argv[1:])