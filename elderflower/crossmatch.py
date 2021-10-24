import os
import re
import sys
import math
import random

import warnings

import numpy as np

from astropy import wcs
from astropy import units as u
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, join, vstack

from .io import logger
from .image import DF_pixel_scale, DF_raw_pixel_scale
from .utils import transform_coords2pixel
from .utils import crop_catalog, merge_catalog, SE_COLUMNS


def query_vizier(catalog_name, radius, columns, column_filters, header=None, coord=None):
    """ Query catalog in Vizier database with the given catalog name,
    search radius and column names. If coords is not given, look for fits header """
    from astroquery.vizier import Vizier

    # Prepare for quearyinig Vizier with filters up to infinitely many rows. By default, this is 50.
    viz_filt = Vizier(columns=columns, column_filters=column_filters)
    viz_filt.ROW_LIMIT = -1

    if coord==None:
        RA, DEC = re.split(",", header['RADEC'])
        coord = SkyCoord(RA+" "+DEC , unit=(u.hourangle, u.deg))

    # Query!
    result = viz_filt.query_region(coord, radius=radius,
                                   catalog=[catalog_name])
    return result
    
    
def cross_match(wcs_data, SE_catalog, bounds, radius=None,
                pixel_scale=DF_pixel_scale, mag_limit=15, sep=3*u.arcsec,
                clean_catalog=True, mag_name='rmag',
                catalog={'Pan-STARRS': 'II/349/ps1'},
                columns={'Pan-STARRS': ['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
                                        'objID', 'Qual', 'gmag', 'e_gmag', 'rmag', 'e_rmag']},
                column_filters={'Pan-STARRS': {'rmag':'{0} .. {1}'.format(5, 22)}},
                magnitude_name={'Pan-STARRS':['rmag','gmag']},
                verbose=True):
    """
        Cross match SExtractor catalog with Vizier Online catalog.
        
        'URAT': 'I/329/urat1'
                magnitude_name: "rmag"
                columns: ['RAJ2000', 'DEJ2000', 'mfa', 'gmag', 'e_gmag', 'rmag', 'e_rmag']
                column_filters: {'mfa':'=1', 'rmag':'{0} .. {1}'.format(8, 18)}
                
        'USNO': 'I/252/out'
                magnitude_name: "Rmag"
                columns: ['RAJ2000', 'DEJ2000', 'Bmag', 'Rmag']
                column_filters: {"Rmag":'{0} .. {1}'.format(5, 15)}
                
    """
    
    cen = (bounds[2]+bounds[0])/2., (bounds[3]+bounds[1])/2.
    coord_cen = wcs_data.pixel_to_world(cen[0], cen[1])
    
    if radius is None:
        L = math.sqrt((cen[0]-bounds[0])**2 + (cen[1]-bounds[1])**2)
        radius = L * pixel_scale * u.arcsec
    if verbose:
        msg = "Search {0} ".format(np.around(radius.to(u.deg), 3))
        msg += f"around: (ra, dec) = ({coord_cen.to_string()})"
        logger.info(msg)
    
    for j, (cat_name, table_name) in enumerate(catalog.items()):
        # Query from Vizier
        result = query_vizier(catalog_name=cat_name,
                              radius=radius,
                              columns=columns[cat_name],
                              column_filters=column_filters[cat_name],
                              coord=coord_cen)

        Cat_full = result[table_name]
        
        if len(cat_name) > 4:
            c_name = cat_name[0] + cat_name[-1]
        else:
            c_name = cat_name
        
        m_name = np.atleast_1d(mag_name)[j]
        
        # Transform catalog wcs coordinate into pixel postion
        Cat_full = transform_coords2pixel(Cat_full, wcs_data, name=c_name)
        
        # Crop catalog and sort by the catalog magnitude
        Cat_crop = crop_catalog(Cat_full, bounds, sortby=m_name,
                                keys=("X_CATALOG", "Y_CATALOG"))
        
        # catalog magnitude
        mag_cat = Cat_crop[m_name]
        
        # Screen out bright stars (mainly for cleaning duplicate source in catalog)
        Cat_bright = Cat_crop[(np.less(mag_cat, mag_limit,
                                       where=~np.isnan(mag_cat))) & ~np.isnan(mag_cat)]
        mag_cat.mask[np.isnan(mag_cat)] = True
        
        if clean_catalog:
            # Clean duplicate items in the catalog
            c_bright = SkyCoord(Cat_bright['RAJ2000'], Cat_bright['DEJ2000'], unit=u.deg)
            c_catalog = SkyCoord(Cat_crop['RAJ2000'], Cat_crop['DEJ2000'], unit=u.deg)
            idxc, idxcatalog, d2d, d3d = c_catalog.search_around_sky(c_bright, sep)
            inds_c, counts = np.unique(idxc, return_counts=True)
            
            row_duplicate = np.array([], dtype=int)
            
            # Use the measurement with min error in RA/DEC
            for i in inds_c[counts>1]:
                obj_duplicate = Cat_crop[idxcatalog][idxc==i]
#                 obj_duplicate.pprint(max_lines=-1, max_width=-1)
                
                # Remove detection without magnitude measurement
                mag_obj = obj_duplicate[m_name]
                obj_duplicate = obj_duplicate[~np.isnan(mag_obj)]
                
                # Use the detection with the best astrometry
                e2_coord = obj_duplicate["e_RAJ2000"]**2 + obj_duplicate["e_DEJ2000"]**2
                min_e2_coord = np.nanmin(e2_coord)
                
                for ID in obj_duplicate[e2_coord>min_e2_coord]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
            
            Cat_crop.remove_rows(np.unique(row_duplicate))
            #Cat_bright = Cat_crop[mag_cat<mag_limit]
            
        for m_name in magnitude_name[cat_name]:
            mag = Cat_crop[m_name]
            if verbose:
                logger.debug("%s %s:  %.3f ~ %.3f"%(cat_name, m_name, mag.min(), mag.max()))

        # Merge Catalog
        keep_columns = SE_COLUMNS + ["ID"+'_'+c_name] + magnitude_name[cat_name] + \
                                    ["X_CATALOG", "Y_CATALOG"]
        tab_match = merge_catalog(SE_catalog, Cat_crop, sep=sep,
                                  keep_columns=keep_columns)
        
        tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                         keep_columns=keep_columns)
        
        # Rename columns
        for m_name in magnitude_name[cat_name]:
            tab_match[m_name].name = m_name+'_'+c_name
            tab_match_bright[m_name].name = m_name+'_'+c_name
        
        # Join tables
        if j==0:
            tab_target_all = tab_match
            tab_target = tab_match_bright
        else:
            tab_target_all = join(tab_target_all, tab_match, keys=SE_COLUMNS,
                                 join_type='left', metadata_conflicts='silent')
            tab_target = join(tab_target, tab_match_bright, keys=SE_COLUMNS,
                              join_type='left', metadata_conflicts='silent')
            
    # Sort matched catalog by SE MAG_AUTO
    tab_target.sort("MAG_AUTO")
    tab_target_all.sort("MAG_AUTO")

    mag_all = tab_target_all[mag_name+'_'+c_name]
    mag = tab_target[mag_name+'_'+c_name]
    if verbose:
        logger.info("Matched stars with %s %s:  %.3f ~ %.3f"\
              %(cat_name, mag_name, mag_all.min(), mag_all.max()))
        logger.info("Matched bright stars with %s %s:  %.3f ~ %.3f"\
              %(cat_name, mag_name, mag.min(), mag.max()))
    
    return tab_target, tab_target_all, Cat_crop

def cross_match_PS1_DR2(wcs_data, SE_catalog, bounds,
                        band='g', radius=None, clean_catalog=True,
                        pixel_scale=DF_pixel_scale, sep=5*u.arcsec,
                        mag_limit=15, verbose=True):
    """
    Use PANSTARRS DR2 API to do cross-match with the SE source catalog.
    Note this could be (much) slower compared to cross-match using Vizier.
    
    Parameters
    ----------
    wcs_data : wcs of data
    
    SE_catalog : SE source catalog
    
    bounds : Nx4 2d / 1d array defining the cross-match region(s) [Xmin, Ymin, Xmax, Ymax]

    clean_catalog : whether to clean the matched catalog. (default True)
            The PS-1 catalog contains duplicate items on a single source with different
            measurements. If True, duplicate items of bright sources will be cleaned by
            removing those with large coordinate errors and pick the items with most
            detections in that band .
            
    mag_limit : magnitude threshould defining bright stars.
    
    sep : maximum separation (in astropy unit) for crossmatch with SE.
    
    Returns
    -------
    tab_target : table containing matched bright sources with SE source catalog
    tab_target_all : table containing matched all sources with SE source catalog
    catalog_star : PS-1 catalog of all sources in the region(s)
        
    """
    from astropy.nddata.bitmask import interpret_bit_flags
    from .panstarrs import ps1cone
    
    band = band.lower()
    mag_name = band + 'MeanPSFMag'
    c_name = 'PS'
    
    for j, bounds in enumerate(np.atleast_2d(bounds)):
        cen = (bounds[2]+bounds[0])/2., (bounds[3]+bounds[1])/2.
        coord_cen = wcs_data.pixel_to_world(cen[0], cen[1])
        ra, dec = coord_cen.ra.value, coord_cen.dec.value
        
        L = math.sqrt((cen[0]-bounds[0])**2 + (cen[1]-bounds[1])**2)
        radius = (L * pixel_scale * u.arcsec).to(u.deg)
        
        if verbose:
            msg = "Search {0} ".format(np.around(radius.to(u.deg), 3))
            msg += f"around: (ra, dec) = ({coord_cen.to_string()})"
            logger.info(msg)
        
        #### Query PANSTARRS start ####
        constraints = {'nDetections.gt':1, band+'MeanPSFMag.lt':22}

        # strip blanks and weed out blank and commented-out values
        columns = """raMean,decMean,raMeanErr,decMeanErr,nDetections,ng,nr,
                    gMeanPSFMag,gMeanPSFMagErr,gFlags,rMeanPSFMag,rMeanPSFMagErr,rFlags""".split(',')
        columns = [x.strip() for x in columns]
        columns = [x for x in columns if x and not x.startswith('#')]
        results = ps1cone(ra, dec, radius.value, release='dr2', columns=columns, **constraints)
        
        Cat_full = ascii.read(results)
        for filter in 'gr':
            col = filter+'MeanPSFMag'
            Cat_full[col].format = ".4f"
            Cat_full[col][Cat_full[col] == -999.0] = np.nan
        for coord in ['ra','dec']:
            Cat_full[coord+'MeanErr'].format = ".5f"
            
        #### Query PANSTARRS end ####

        Cat_full.sort(mag_name)
        Cat_full['raMean'].unit = u.deg
        Cat_full['decMean'].unit = u.deg
        Cat_full = transform_coords2pixel(Cat_full, wcs_data, name=c_name,
                                          RA_key="raMean", DE_key="decMean")
        
        # Crop catalog and sort by the catalog magnitude
        Cat_crop = crop_catalog(Cat_full, bounds, sortby=mag_name,
                                keys=("X_CATALOG", "Y_CATALOG"))
        
        # Remove detection without magnitude
        has_mag = ~np.isnan(Cat_crop[mag_name])
        Cat_crop = Cat_crop[has_mag]
        
        # Pick out bright stars
        mag_cat = Cat_crop[mag_name]
        Cat_bright = Cat_crop[mag_cat<mag_limit]
        
        if clean_catalog:
            # A first crossmatch with bright stars in catalog for cleaning
            tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                             RA_key="raMean", DE_key="decMean")
            tab_match_bright.sort(mag_name)
            
            # Clean duplicate items in the catalog
            c_bright = SkyCoord(tab_match_bright['X_WORLD'],
                                tab_match_bright['Y_WORLD'], unit=u.deg)
            c_catalog = SkyCoord(Cat_crop['raMean'],
                                 Cat_crop['decMean'], unit=u.deg)
            idxc, idxcatalog, d2d, d3d = \
                        c_catalog.search_around_sky(c_bright, sep)
            inds_c, counts = np.unique(idxc, return_counts=True)
            
            row_duplicate = np.array([], dtype=int)
            
            # Use the measurement following some criteria
            for i in inds_c[counts>1]:
                obj_dup = Cat_crop[idxcatalog][idxc==i]
                obj_dup['sep'] = d2d[idxc==i]
                #obj_dup.pprint(max_lines=-1, max_width=-1)
                
                # Use the detection with mag
                mag_obj_dup = obj_dup[mag_name]
                obj_dup = obj_dup[~np.isnan(mag_obj_dup)]
                                
                # Use the closest match
                good = (obj_dup['sep'] == min(obj_dup['sep']))
                
### Extra Criteria
#                 # Coordinate error of detection
#                 err2_coord = obj_dup["raMeanErr"]**2 + \
#                             obj_dup["decMeanErr"]**2
                
#                 # Use the detection with the best astrometry
#                 min_e2_coord = np.nanmin(err2_coord)
#                 good = (err2_coord == min_e2_coord)
                
#                 # Use the detection with PSF mag err
#                 has_err_mag = obj_dup[mag_name+'Err'] > 0
                
#                 # Use the detection > 0
#                 n_det = obj_dup['n'+band]
#                 has_n_det = n_det > 0
                
#                 # Use photometry not from tycho in measurement
#                 use_tycho_phot =  extract_bool_bitflags(obj_dup[band+'Flags'], 7)
                
#                 good = has_err_mag & has_n_det & (~use_tycho_phot)
###
                    
                # Add rows to be removed
                for ID in obj_dup[~good]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
                
                obj_dup = obj_dup[good]
                
                if len(obj_dup)<=1:
                    continue
                
                # Use brightest detection
                mag = obj_dup[mag_name]
                for ID in obj_dup[mag>min(mag)]['ID'+'_'+c_name]:
                    k = np.where(Cat_crop['ID'+'_'+c_name]==ID)[0][0]
                    row_duplicate = np.append(row_duplicate, k)
            
            # Remove rows
            Cat_crop.remove_rows(np.unique(row_duplicate))
            
            # Subset catalog containing bright stars
            Cat_bright = Cat_crop[Cat_crop[mag_name]<mag_limit]
        
        # Merge Catalog
        keep_columns = SE_COLUMNS + ["ID"+'_'+c_name] + columns + \
                                    ["X_CATALOG", "Y_CATALOG"]
        tab_match = merge_catalog(SE_catalog, Cat_crop, sep=sep,
                                  RA_key="raMean", DE_key="decMean", keep_columns=keep_columns)
        tab_match_bright = merge_catalog(SE_catalog, Cat_bright, sep=sep,
                                         RA_key="raMean", DE_key="decMean", keep_columns=keep_columns)
        
        if j==0:
            tab_target_all = tab_match
            tab_target = tab_match_bright
            catalog_star = Cat_crop
            
        else:
            tab_target_all = vstack([tab_target_all, tab_match], join_type='exact')
            tab_target = vstack([tab_target, tab_match_bright], join_type='exact')
            catalog_star = vstack([catalog_star, Cat_crop], join_type='exact')
            
    # Sort matched catalog by matched magnitude
    tab_target.sort(mag_name)
    tab_target_all.sort(mag_name)
    
    if verbose:
        logger.info("Matched stars with PANSTARRS DR2 %s:  %.3f ~ %.3f"\
              %(mag_name, np.nanmin(tab_target_all[mag_name]),
                np.nanmax(tab_target_all[mag_name])))
        logger.info("Matched bright stars with PANSTARRS DR2 %s:  %.3f ~ %.3f"\
              %(mag_name, np.nanmin(tab_target[mag_name]),
                np.nanmax(tab_target[mag_name])))
    
    return tab_target, tab_target_all, catalog_star
        
def cross_match_PS1(band, wcs_data,
                    SE_cat_target, bounds_list,
                    pixel_scale=DF_pixel_scale,
                    sep=None, mag_limit=15, n_attempt=3,
                    use_PS1_DR2=False, verbose=True):
                    
    b_name = band.lower()
    
    if sep is None:
        sep = pixel_scale
    
    if use_PS1_DR2:
        from urllib.error import HTTPError
        # Give 3 attempts in matching PS1 DR2 via MAST.
        # This could fail if the FoV is too large.
        for attempt in range(n_attempt):
            try:
                tab_target, tab_target_full, catalog_star = \
                            cross_match_PS1_DR2(wcs_data,
                                                SE_cat_target,
                                                bounds_list,
                                                pixel_scale=pixel_scale,
                                                sep=sep * u.arcsec,
                                                mag_limit=mag_limit,
                                                band=b_name,
                                                verbose=verbose)
            except HTTPError:
                logger.warning('Gateway Time-out. Try again.')
            else:
                break
        else:
            msg = f'504 Server Error: {n_attempt} failed attempts. Exit.'
            logger.error(msg)
            sys.exit()
            
    else:
        mag_name = b_name+'mag'
        tab_target, tab_target_full, catalog_star = \
                            cross_match(wcs_data,
                                        SE_cat_target,
                                        bounds_list,
                                        pixel_scale=pixel_scale,
                                        sep=sep * u.arcsec,
                                        mag_limit=mag_limit,
                                        mag_name=mag_name,
                                        verbose=verbose)
                                        
    return tab_target, tab_target_full, catalog_star
