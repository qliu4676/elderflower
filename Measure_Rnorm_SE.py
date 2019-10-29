import pandas as pd
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table, Column, setdiff, join, unique

from utils import *
from model import *

save = True
draw = False
image_bounds = 800, 1800, 1800, 2800  # in image coords
mag_limit = 15
hdu_path = "./coadd_SloanR_NGC_5907.fits"
seg_map = "./SE_APASS/coadd_SloanR_NGC_5907_seg.fits"
weight_map = "./weight_NGC5907.fits"
SE_catalog = "./SE_APASS/coadd_SloanR_NGC_5907.cat"

def main(hdu_path, seg_map, SE_catalog, image_bounds, 
         weight_map=None, R_norm=10, mag_limit=mag_limit,
         save=save, draw=draw):
    
    print("Measure the intensity at R=%.1f for stars < %.1f as the normalization of fitting\n"%(R_norm, mag_limit))
    
    ############################################
    # Read and Crop
    ############################################

    # Read hdu
    hdu = fits.open(hdu_path)[0]
    data = hdu.data
    header = hdu.header
    wcs_data = wcs.WCS(header)
    seeing = 2.5

    # Read output from create_photometric_light_APASS 
    seg_map = fits.open(seg_map)[0].data
    if weight_map is not None:
        weight_edge = fits.open(weight_map)[0].data
    SE_cat_full = Table.read(SE_catalog, format="ascii.sextractor")

    # Read homogeneous background model and stddev, zero point and pixel scale from header
    mu, std = np.float(hdu.header["BACKVAL"]), mad_std(hdu.data[seg_map==0&(weight_edge>0.5)]) 
    ZP, pix_scale = np.float(hdu.header["REFZP"]), np.float(hdu.header["PIXSCALE"])
    print("mu: %.2f , std: %.2f , ZP: %.2f , pix_scale: %.2f\n" %(mu, std, ZP, pix_scale))

    # Convert SE measured flux into mag
    SE_cat_full["RMAG_AUTO"] = -2.5*np.log10(SE_cat_full["FLUX_AUTO"]) + ZP

    # Crop the whole field into smaller patch
    patch_Xmin, patch_Ymin, patch_Xmax, patch_Ymax = image_bounds
    patch, seg_patch = crop_image(data, seg_map, image_bounds, weight_map=weight_edge, 
                                  sky_mean=mu, sky_std=std, color="w", draw=draw)
    print("Using Sky Patch: [%d:%d, %d:%d]\n"%image_bounds)

    ############################################
    # Crossmatch
    ############################################

    # Query from Vizier online star catalog

    # Crossmatch with URAT
    result = query_vizier(catalog_name="URAT", radius=2*u.deg,
                          columns=['RAJ2000', 'DEJ2000', 'mfa', 'gmag', 'e_gmag', 'rmag', 'e_rmag'],
                          column_filters={'mfa':'=1', 'rmag':'{0} .. {1}'.format(8, 18)}, header=header)

    Cat_URAT = result['I/329/urat1']
    Cat_URAT = transform_coords2pixel(Cat_URAT, wcs=wcs_data, cat_name="URAT")     
    print("URAT Brightest Star: Rmag = %.3f"%Cat_URAT["rmag"].min())

    # Bright stars R < 10 mag are not included in URAT, match with USNO
    result = query_vizier(catalog_name="USNO", radius=2*u.deg,
                          columns=['RAJ2000', 'DEJ2000', "Bmag","Rmag"],
                          column_filters={"Rmag":'{0} .. {1}'.format(5, 15)}, header=header)

    Cat_USNO = result['I/252/out']
    Cat_USNO = transform_coords2pixel(Cat_USNO, wcs=wcs_data, cat_name="USNO")
    print("USNO Brightest Star: Rmag = %.3f"%Cat_USNO["Rmag"].min())


    ############################################
    # Merge Catalog
    ############################################

    # Merge SE catalog with cross-matched catalogs
    df_match_URAT = merge_catalog(SE_cat_full, Cat_URAT, sep=2.5*u.arcsec, 
                                  keep_columns=["NUMBER","X_IMAGE","Y_IMAGE","ELLIPTICITY","RMAG_AUTO", 'FWHM_IMAGE',
                                                "ID_URAT","X_IMAGE_URAT","Y_IMAGE_URAT","rmag","gmag","FLAGS"])
    df_match_USNO = merge_catalog(SE_cat_full, Cat_USNO, sep=5*u.arcsec,
                                   keep_columns=["NUMBER","X_IMAGE","Y_IMAGE","ELLIPTICITY","RMAG_AUTO",
                                                 "ID_USNO","X_IMAGE_USNO","Y_IMAGE_USNO","Rmag","FLAGS"])

    df_SE_match_all = pd.merge(df_match_URAT, df_match_USNO, how='outer')

    # Crop the catalog into smaller spatial range, with a padding on the used field 
    bounds = (patch_Xmin-200, patch_Ymin-200, patch_Xmax+200, patch_Ymax+200)
    df_SE_match = crop_catalog(df_SE_match_all, bounds, keys=("X_IMAGE", "Y_IMAGE"))
    df_SE_match = df_SE_match.reset_index(drop=True)

    # Only measure stars brighter than mag_limit
    target = df_SE_match["RMAG_AUTO"] < mag_limit
    df_SE_target = df_SE_match[target].sort_values("RMAG_AUTO")

    ############################################
    # Measure Scaling
    ############################################

    # Measure 3sigma-clipped intensity at R as the input normalization for fitting
    res_Rnorm, res_thumb = compute_Rnorm_batch(df_SE_target, SE_cat_full, wcs_data, data, seg_map,
                                               R=R_norm, wid=0.5, return_full=True)

    # Sort the result array into a table
    table_res_Rnorm = Table(np.hstack([df_SE_target['NUMBER'].values[:, None],
                                       df_SE_target['X_IMAGE'].values[:, None],
                                       df_SE_target['Y_IMAGE'].values[:, None],
                                       res_Rnorm]), 
                            names=['NUMBER','X_IMAGE','Y_IMAGE','mean','med','std','sky'],
                            dtype=['int']+['float']*6)

    if save:
        table_res_Rnorm.write('Rnorm_10pix_15mag_X%sY%s.txt'%(patch_Xmin, patch_Ymin),
                              overwrite=True, format='ascii')
        save_thumbs(res_thumb, 'Thumb_15mag_X%sY%s'%(patch_Xmin, patch_Ymin))
        print("\nFinising Saving!")

        
main(hdu_path, seg_map, SE_catalog, image_bounds,
     weight_map=weight_map, R_norm=10, mag_limit=mag_limit,
     save=save, draw=draw)

