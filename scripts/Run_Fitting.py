#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run 2D Bayesian PSF fitting on a sub-region with
dynamic nested sampling. The Model PSF is composed
of an inner (fixed) Moffat core and an outer (user-
specified) multi-power law aureole. The fitting 
result containing the joint PDF, samples and 
weights, etc. and diagnostic plots will be saved.

> Parameter
[-f][--FILTER]: filter of image to be crossmatched. g/G/r/R for Dragonfly.
[-b][--IMAGE_BOUNDS]: bounds of the region to be processed in pixel coordinate. [Xmin, Ymin, Xmax, Ymax]
[-I][--IMAGE]: path of image.
[-n][--N_COMP]: number of multi-power law component (default: 2).
[-r][--R_SCALE]: radius at which normalziation is measured, in pixel.
[-m][--MAG_THRE]: magnitude thresholds used to select [medium, very] bright stars (default: [14,11]). 
[-M][--MASK_TYPE]: mask core by "radius" or "brightness" (default: "radius").
[-c][--R_CORE]: inner masked radius for [medium, very] bright stars, in pixel (default: 24). A single value can be passed for both.
[-s][--SB_FIT_THRE]: inner masked surface brightness, in mag/arcsec^2 (default: 26)
[-B][--BRIGHTEST_ONLY]: whether to fit brightest stars only.
[-L]: whether to fit a 1st-order Legendre polynomial for the background.
[--PARALLEL]: whether to draw meidum bright stars in parallel.
[--N_CPU]: number of CPU used in nested sampling (default: n_cpu-1).
[--NO_PRINT]: if yes, suppress progress print.
[--DIR_MEASURE]: directory name where normalization measurements are saved.
[--DIR_NAME]: directory name for saving fitting outputs.

> Example Usage
1. In jupyter notebook / lab
%matplotlib inline
%run -i Run_Fitting.py -f 'G' -b '[3000, 1300, 4000, 2300]' -n 2 -r 12 -B
2. In bash

"""

import sys
import getopt
from src.task import Run_PSF_Fitting

def main(argv):
    # Image Parameter (default)
    band = "G"                
    pixel_scale = 2.5  # arcsec/pixel
    
    # Fitting Setup (default)
    n_cpu = 4
    n_spline = 2
    draw_real = True
    brightest_only = False
    parallel = False 
    leg2d = False
    fit_frac = False
    
    # Fitting Option (default)
    print_progress = True
    draw = True
    save = True

    # Measure Parameter
    r_scale = 12
    mag_threshold = np.array([14, 11])
    
    # Mask Setup
    mask_type = 'radius'
    r_core = 24
    r_out = None
    SB_fit_thre = 26
    wid_strip, n_strip = 24, 48
    
    # Get Script Options
    try:
        optlists, args = getopt.getopt(argv, "f:b:n:r:m:c:s:M:I:BLFCP",
                                       ["FILTER=", "IMAGE=", "IMAGE_BOUNDS=",
                                        "N_COMP=", "R_SCALE=", "MAG_THRE=",
                                        "MASK_TYPE=", "R_CORE=", "SB_FIT_THRE=", 
                                        "N_CPU=", "PARALLEL", "BRIGHTEST_ONLY", 
                                        "NO_PRINT", "W_STRIP=", "N_STRIP=",
                                        "CONV", "NO_SAVE",
                                        "DIR_NAME=", "DIR_MEASURE=", "DIR_DATA="])
        
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
    
    # Default Input/Output Path
    hdu_path = os.path.join(work_dir, "data/coadd_Sloan%s_NGC_5907.fits"%band)
    dir_name = os.path.join(work_dir, 'output/fit')
    dir_measure = os.path.join(work_dir, 'output/Measure')
                
    # Handling Options    
    for opt, arg in optlists:
        if opt in ("-I", "--IMAGE"):
            hdu_path = arg
        elif opt in ("-b", "--IMAGE_BOUNDS"):    
            image_bounds0 = np.array(re.findall(r'\d+', arg), dtype=int)
        elif opt in ("-n", "--N_COMP"):
            try:
                n_spline = np.int(arg)
            except ValueError:
                sys.exit("Model Not Available.")
        elif opt in ("-r", "--R_SCALE"):
            r_scale = np.float(arg)
        elif opt in ("-m", "--MAG_THRE"):    
            mag_threshold = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=float)
        elif opt in ("-M", "--MASK_TYPE"):    
            mask_type = arg
        elif opt in ("-c", "--R_CORE"):    
            r_core = np.array(re.findall(r"\d*\.\d+|\d+", arg), dtype=float)
        elif opt in ("-s","--SB_FIT_THRE"):    
            SB_fit_thre = np.float(arg)
        elif opt in ("--W_STRIP"):
            wid_strip = np.float(arg)
        elif opt in ("--N_STRIP"):
            n_strip = np.float(arg)
        elif opt in ("--N_CPU"):
            n_cpu = np.int(arg)
        elif opt in ("--DIR_NAME"):
            dir_name = arg
        elif opt in ("--DIR_MEASURE"):
            dir_measure = arg
            
    if 'image_bounds0' not in locals():
        sys.exit("Image Bounds Required.")
            
    if '-L' in opts: leg2d = True
    if '-F' in opts: fit_frac = True
    if ('-B' in opts)|("--BRIGHTEST_ONLY" in opts): brightest_only = True
    if ('-C' in opts)|("--CONV" in opts): draw_real = False
    if ('-P' in opts)|("--PARALLEL" in opts): parallel = True
    if ("--NO_PRINT" in opts): print_progress = False
    if ("--NO_SAVE" in opts): save = False
    
    if mask_type=='radius':
        dir_name = os.path.join(dir_name, "NGC5907-%s-R%dM%dpix_X%dY%d"\
                                %(band, r_scale, r_core, image_bounds0[0], image_bounds0[1]))
    elif mask_type=='brightness':
        dir_name = os.path.join(dir_name, "NGC5907-%s-R%dB%.1f_X%dY%d"\
                                %(band, r_scale, SB_fit_thre, image_bounds0[0], image_bounds0[1]))
    if save:
        check_save_path(dir_name, make_new=False)
    
    # Run Fitting~!
    ds = Run_PSF_Fitting(hdu_path, image_bounds0, n_spline, band,
                         r_scale=r_scale, mag_threshold=mag_threshold, 
                         mask_type=mask_type, SB_fit_thre=SB_fit_thre,
                         r_core=r_core, r_out=r_out, leg2d=leg2d,
                         pixel_scale=pixel_scale, n_cpu=n_cpu,
                         wid_strip=wid_strip, n_strip=n_strip,
                         brightest_only=brightest_only, draw_real=draw_real,
                         parallel=parallel, print_progress=print_progress, 
                         draw=draw, dir_measure=dir_measure,
                         save=save, dir_name=dir_name)

    return opts

    
if __name__ == "__main__":
    main(sys.argv[1:])
