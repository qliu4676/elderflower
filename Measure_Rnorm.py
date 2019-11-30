import sys
import getopt
from utils import *
from modeling import *
from plotting import *

def main(argv):  
    # Default Parameters
    band = "G"
    save, draw = True, True
    mag_thre, r_scale = 15, 12

    image_bounds = [[800, 1600, 1800, 2600],
                    [3100, 1400, 4100, 2400]]  # in image coords
    
    hdu_path = "./data/coadd_Sloan%s_NGC_5907.fits"%band
    seg_map = "./SE_APASS/coadd_Sloan%s_NGC_5907_seg.fits"%band
    SE_catalog = "./SE_APASS/coadd_Sloan%s_NGC_5907.cat"%band
    weight_map = "./SE_APASS/weight_NGC5907.fits"
    dir_name = './Measure'
    
    try:
        opts, args = getopt.getopt(argv, "f:b:m:r:I:C:S:W:",
                                   ["FILTER=", "IMAGE_BOUNDS=", "MAG_THRESHOLD=", "R_SCALE=", 
                                    "IMAGE=", "SE_CATALOG=", "SEGMENT=", "WEIGHT=", "DIR_NAME="])
    except getopt.GetoptError:
        print('Wrong Option.')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-f", "--FILTER"):
            if arg in ["G", "R", "r", "g"]:
                band = arg.upper()
            else:
                print("Filter Not Available.")
                sys.exit(1)
        elif opt in ("-b", "--IMAGE_BOUNDS"):    
            image_bounds = np.array(re.findall(r'\d+', arg), dtype=int).reshape(-1,4)
        elif opt in ("-r", "--R_SCALE"):
            r_scale = np.float(arg)
        elif opt in ("-m", "--MAG_THRESHOLD"):
            mag_thre = np.float(arg)
        elif opt in ("-I", "--IMAGE"):
            hdu_path = arg
        elif opt in ("-C", "--SE_CATALOG"):
            SE_catalog = arg
        elif opt in ("-S", "--SEGMENT"):
            seg_map = arg
        elif opt in ("-W", "--WEIGHT"):
            weight_map = arg
        elif opt in ("--DIR_NAME"):
            dir_name = arg
            check_save_path(dir_name, make_new=False)


    Match_Mask_Measure(hdu_path, seg_map, SE_catalog, image_bounds,
                       weight_map=weight_map, band=band,
                       r_scale=r_scale, mag_thre=mag_thre,
                       draw=draw, save=save, dir_name=dir_name)
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
                       dir_name='./Measure'):
    
    print("Measure the intensity at R = %d for stars < %.1f as normalization of fitting\n"%(r_scale, mag_thre))
    
    mag_name = band.lower() + 'mag'
    obj_name = 'NGC5907-' + band
    image_bounds = np.atleast_2d(image_bounds)
    
    ############################################
    # Read and Display
    ############################################

    # Read hdu
    with fits.open(hdu_path) as hdul:
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
        data += BKG
        
    # Convert SE measured flux into mag
    flux = SE_cat_full["FLUX_AUTO"]
    mag = -2.5 * np.ma.log10(flux).filled(flux[flux>0].min()) + ZP
#     mag = -2.5 * np.log10(flux) + ZP
    SE_cat_full["MAG_AUTO"] = np.around(mag, 5)
    
    field_bounds = [700, 700, data.shape[1]-700, data.shape[0]-700]
    print("Match field %r with catalog\n"%field_bounds)
    
    print("Measure Sky Patch (X min, Y min, X max, Y max) :")
    [print("%r"%b) for b in image_bounds.tolist()]
    
    _, _ = crop_image(data, field_bounds, seg_map,
                      weight_map=weight_edge, sub_bounds=image_bounds, draw=draw)

    ############################################
    # Crossmatch with Star Catalog (across the field)
    ############################################

    # Crossmatch with PANSTRRS at a threshold of mag_thre mag
    tab_target, tab_target_full, catalog_star = cross_match(header, SE_cat_full, field_bounds,
                                                            mag_thre=mag_thre, mag_name=mag_name)

    # Calculate color correction of PANSTAR and DF
    CT = calculate_color_term(tab_target_full, mag_name=mag_name+'_PS', draw=draw)
    catalog_star["MAG_AUTO"] = catalog_star[mag_name] + CT
    
    # Save matched table and catalog
    if save:
        tab_target_name = './%s/%s-catalog_match_%s%dmag.txt'%(dir_name, obj_name, mag_name[0], mag_thre)
        tab_target["MAG_AUTO_corr"] = tab_target[mag_name+'_PS'] + CT
        tab_target.write(tab_target_name, overwrite=True, format='ascii')

        catalog_star_name = './%s/%s-catalog_PS_%s_all.txt'%(dir_name, obj_name, mag_name[0])
        catalog_star.write(catalog_star_name, overwrite=True, format='ascii')
        print('Save the PANSTARRS catalog and matched sources in %s'%dir_name)
        
    ############################################
    # Build Mask & Measure Scaling (in selected patch)
    ############################################
    
    # Empirical enlarged aperture size from magnitude based on matched SE detection
    estimate_radius = fit_empirical_aperture(tab_target_full, seg_map, mag_name=mag_name+'_PS',
                                             mag_range=[10,20], K=2, degree=3, draw=draw)
    
    for image_bound in image_bounds:
        
        # Crop the star catalog and matched SE catalog
        patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = image_bound

        catalog_bound = (patch_Xmin-100, patch_Ymin-100, patch_Xmax+100, patch_Ymax+100)

        catalog_star_patch = crop_catalog(catalog_star, catalog_bound, sortby=mag_name,
                                          keys=("X_IMAGE"+'_PS', "Y_IMAGE"+'_PS'))
        tab_target_patch = crop_catalog(tab_target, catalog_bound, sortby='MAG_AUTO',
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
                                                     obj_name=obj_name, mag_name=mag_name+'_PS', 
                                                     read=False, save=save, dir_name=dir_name)

if __name__ == "__main__":
    main(sys.argv[1:])