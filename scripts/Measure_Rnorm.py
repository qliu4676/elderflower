#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preparatory script doing the following tasks:
1) Crossmatch a source table (e.g. from SExtractor, 
SEP, etc.) with star catalog(s), by default using 
PANSTARRS DR2 catalog. A new table will be saved which 
contains corrected source magnitude (e.g. MAG_AUTO).
2) Make enlarged segmentation map(s) for sub-region(s) 
from the star catalog (by default < 23 mag).
3) Measure normalization (i.e. flux scaling) at a 
certain radius for bright stars (by default < 15 mag).

> Parameter
[-f][--FILTER]: filter of image to be crossmatched. g/G/r/R for Dragonfly.
[-b][--IMAGE_BOUNDS]: bounds of region(s) to be processed in pixel coordinate.[Xmin, Ymin, Xmax, Ymax],[...],...
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
# from src.utils import *
# from src.modeling import *
# from src.plotting import *
from src.task import Match_Mask_Measure

def main(argv):  
    # Default Parameters
    band = "G"
    save, draw = True, True
    use_PS1_DR2 = False
    mag_thre, r_scale = 15, 12
    
    image_bounds = [3000, 1300, 4000, 2300]  # in image coords
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv, "f:b:m:r:I:C:S:W:",
                                       ["FILTER=", "IMAGE_BOUNDS=", "IMAGE=", 
                                        "MAG_THRESHOLD=", "R_SCALE=", "PS", "DIR_NAME=",
                                        "SE_CATALOG=", "SEGMAP=", "WEIGHT_MAP="])
        opts = [opt for opt, arg in optlists]        
        
    except getopt.GetoptError as e:
        print(e)
        sys.exit('Wrong Option.')
    
    for opt, arg in optlists:
        if opt in ("-f", "--FILTER"):
            if arg in ["G", "R", "r", "g"]:
                band = arg.upper()
            else:
                sys.exit("Filter Not Available.")
                
    # Work Path
    work_dir = "/home/qliu/Desktop/PSF"
    
    # Input Path
    hdu_path = os.path.join(work_dir, "data/coadd_Sloan%s_NGC_5907.fits"%band)
    seg_map = os.path.join(work_dir, "SE_APASS/coadd_Sloan%s_NGC_5907_seg.fits"%band)
    SE_catalog = os.path.join(work_dir, "SE_APASS/coadd_Sloan%s_NGC_5907.cat"%band)
    weight_map = os.path.join(work_dir, "SE_APASS/weight_NGC5907.fits")
    
    # Output Path
    dir_name = os.path.join(work_dir, 'psf_modeling/output/Measure')
    
    # Handling Options    
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

    # Run Measurement~!
    Match_Mask_Measure(hdu_path, image_bounds, seg_map, SE_catalog,
                       weight_map=weight_map, band=band,
                       r_scale=r_scale, mag_thre=mag_thre,
                       draw=draw, use_PS1_DR2=use_PS1_DR2,
                       save=save, dir_name=dir_name)
    return opts
    
    
if __name__ == "__main__":
    main(sys.argv[1:])