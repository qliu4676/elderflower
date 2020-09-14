import os
import math
import shutil
import subprocess
import multiprocess as mp

from elderflower.io import get_SExtractor_path, default_SE_config, default_SE_conv, default_SE_nnw
from elderflower.task import Run_Detection, Match_Mask_Measure, Run_PSF_Fitting, berry

obj_name = 'cutout'
filt = 'g'
fn  = 'cutout.fits'
work_dir = './'

bounds = ((50, 50, 450, 450))
    
    
def test_run_detection():
    
    # check SE path
    SE_executable = get_SExtractor_path()
    print(SE_executable)
    assert len(SE_executable)>0, 'SExtractor path is not found correctly.'
    
    print(default_SE_conv, default_SE_conv)
    print(default_SE_config)
    
    assert (os.path.isfile(default_SE_conv)) & (os.path.isfile(default_SE_conv)), \
            'SExtractor config files are not found correctly.'
    
    # run
    ZP = Run_Detection(fn, obj_name, filt,
                       threshold=10, work_dir=work_dir,
                       executable=SE_executable,
                       config_path=default_SE_config,
                       ZP_keyname='REFZP', ZP=None,
                       FILTER_NAME=default_SE_conv,
                       STARNNW_NAME=default_SE_nnw)
    
    assert type(ZP) is float, 'Zero-point is not float.'
    
    SE_has_run = (os.path.isfile(f'{obj_name}.cat')) & (os.path.isfile(f'{obj_name}_seg.fits'))
 
    assert SE_has_run, 'SExtractor has not run properly.'
    
    return None


def test_run_match_mask_measure(ZP=27.116):
    
    # run
    Match_Mask_Measure(fn, bounds, obj_name, filt,
                       ZP=ZP, bkg=519., field_pad=50,
                       pixel_scale=2.5, use_PS1_DR2=False,
                       draw=False, save=True, work_dir=work_dir)
    
    assert os.path.isfile(f'Measure-PS1/{obj_name}-norm_12pix_gmag15_X[50-450]Y[50-450].txt'), \
          'Brightness measurement is not saved properly.'
    
    assert os.path.isfile(f'Measure-PS1/{obj_name}-segm_gmag_catalog_X[50-450]Y[50-450].fits'), \
          'Segmentation file is not saved properly.'
    
    assert os.path.isfile(f'Measure-PS1/{obj_name}-thumbnail_gmag15_X[50-450]Y[50-450].pkl'), \
          'Star thumbnails are not saved properly.'
    
    return None


def check_val_tol(values, truths, tols, names):
    print("Values to check: ", values)
    print("Truths: ", truths)
    for val, truth, tol, name in zip(values, truths, tols, names):
        assert math.fabs(val-truth) < tol, f'{name} does not agree with known solution'

        
def test_run_psf_fitting(ZP=27.116):
    
    # dynesty will be run in parallel
    n_cpu = min(mp.cpu_count()-1, 4)
    
    # run
    samplers = Run_PSF_Fitting(fn, bounds, obj_name,
                               band="g", n_spline=2, n_cpu=n_cpu,
                               r_core=24, mag_threshold=[13.5,10.5],
                               theta_cutoff=1200, ZP=ZP, bkg=519.,
                               pad=0, pixel_scale=2.5, use_PS1_DR2=False,
                               draw=False, save=True, print_progress=False,
                               work_dir=work_dir)
    sampler = samplers[0]
    psf_fit = sampler.psf_fit
    
    # check output values
    vals = [psf_fit.n_s[0], psf_fit.n_s[1], psf_fit.theta_s[1],
            sampler.bkg_fit, sampler.bkg_std_fit]
    truths = [3.25, 1.78, 88, 519, 4.7]
    tols = [5e-2, 5e-2, 10, 0.5, 0.5]
    names = ['n0', 'n1', 'theta1', 'background', 'background std']

    check_val_tol(vals, truths, tols, names)
    
    # check save
    assert os.path.isfile(f'{obj_name}A-g-fit2p.res'), \
          'Fitting result is not saved properly.'
    
    assert os.path.isfile('starsA.pkl'), \
          'Stars pkl is not saved properly.'
    
    # clean out
    shutil.rmtree('plot/')
    shutil.rmtree('Measure-PS1/')
    for f in [f'{obj_name}A-g-fit2p.res', 'starsA.pkl', f'{obj_name}.cat', f'{obj_name}_seg.fits']:
        os.remove(f)

    return None


def test_run_berry(ZP=27.116):
    
    from elderflower.io import get_SExtractor_path, default_SE_config
    
    SE_executable = get_SExtractor_path()
    
    # initialize
    elder = berry(fn, bounds, obj_name, filt, work_dir=work_dir)

    # detection
    elder.detection(threshold=10,
                    executable=SE_executable,
                    config_path=default_SE_config,
                    ZP_keyname='REFZP', ZP=None,
                    FILTER_NAME=default_SE_conv,
                    STARNNW_NAME=default_SE_nnw)
    # run
    elder.run(n_spline=2, mag_threshold=[13.5,10.5],
              theta_cutoff=1200, ZP=ZP, bkg=519.,
              pad=0, pixel_scale=2.5, r_core=24,
              field_pad=50, use_PS1_DR2=False,
              draw=False, save=True, print_progress=False)
    
    sampler = elder.samplers[0]
    psf_fit = sampler.psf_fit
    
    # check output values
    vals = [psf_fit.n_s[0], psf_fit.n_s[1], psf_fit.theta_s[1],
            sampler.bkg_fit, sampler.bkg_std_fit]
    truths = [3.25, 1.78, 88, 519, 4.7]
    tols = [5e-2, 5e-2, 10, 0.5, 0.5]
    names = ['n0', 'n1', 'theta1', 'background', 'background std']
    
    check_val_tol(vals, truths, tols, names)
    
    # check save
    assert os.path.isfile(f'{obj_name}A-g-fit2p.res'), \
          'Fitting result is not saved properly.'
    
    assert os.path.isfile('starsA.pkl'), \
          'Stars pkl is not saved properly.'
    
    # clean out
    shutil.rmtree('plot/')
    shutil.rmtree('Measure-PS1/')
    for f in [f'{obj_name}A-g-fit2p.res', 'starsA.pkl', f'{obj_name}.cat', f'{obj_name}_seg.fits']:
        os.remove(f)
    
    return None
