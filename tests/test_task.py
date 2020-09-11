import os
import math
import shutil
import subprocess
from elderflower.task import Run_Detection, Match_Mask_Measure, Run_PSF_Fitting

obj_name = 'cutout'
filt = 'g'
fn  = 'cutout.fits'
work_dir = './'

bounds = ((50, 50, 450, 450))
    
    
def test_run_detection():
    from elderflower.io import get_SExtractor_path, default_SE_config, default_SE_conv, default_SE_nnw
    
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

def test_run_psf_fitting(ZP=27.116):
    
    # run
    samplers = Run_PSF_Fitting(fn, bounds, obj_name,
                               band="g", n_spline=2, parallel=False,
                               r_core=24, mag_threshold=[13,10.5],
                               theta_cutoff=1200, ZP=ZP, bkg=519.,
                               pad=0, pixel_scale=2.5, use_PS1_DR2=False,
                               draw=False, save=True, work_dir=work_dir)
    sampler = samplers[0]
    psf_fit = sampler.psf_fit
    
    n_tol, theta_tol = 5e-2, 10
    mu_tol, sigma_tol = 0.5, 0.5
    
    assert math.fabs(psf_fit.n_s[0]-3.25) < n_tol, \
           'n0 does not agree with known solution'
    
    assert math.fabs(psf_fit.n_s[1]-1.78) < n_tol, \
           'n1 does not agree with known solution'
    
    assert math.fabs(psf_fit.theta_s[1]-88) < theta_tol, \
           'theta1 does not agree with known solution'
    
    assert math.fabs(sampler.bkg_fit-519) < mu_tol, \
           'background does not agree with known solution'
    
    assert math.fabs(sampler.bkg_std_fit-4.7) < sigma_tol, \
           'background std does not agree with known solution'
    
    fname_fit = f'{obj_name}A-g-fit2p.res'
    fname_star = 'starsA.pkl'
    assert os.path.isfile(fname_fit), \
          'Fitting result is not saved properly.'
    
    assert os.path.isfile(fname_star), \
          'Stars pkl is not saved properly.'

    os.remove(fname_fit)
    os.remove(fname_star)
    shutil.rmtree('plot/')

    return None


def test_parallel_run_psf_fitting(ZP=27.116):
    import multiprocess as mp
    
    # dynesty will be run in parallel
    n_cpu = min(mp.cpu_count()-1, 4)

    # run
    samplers = Run_PSF_Fitting(fn, bounds, obj_name,
                               band="g", n_spline=2, n_cpu=n_cpu,
                               r_core=24, mag_threshold=[13,10.5],
                               theta_cutoff=1200, ZP=ZP, bkg=519.,
                               pad=0, pixel_scale=2.5, use_PS1_DR2=False,
                               draw=False, save=True, work_dir=work_dir)
    
    sampler = samplers[0]
    psf_fit = sampler.psf_fit
    
    n_tol, theta_tol = 5e-2, 10
    mu_tol, sigma_tol = 0.5, 0.5
    
    assert math.fabs(psf_fit.n_s[0]-3.25) < n_tol, \
           'n0 does not agree with known solution'
    
    assert math.fabs(psf_fit.n_s[1]-1.78) < n_tol, \
           'n1 does not agree with known solution'
    
    assert math.fabs(psf_fit.theta_s[1]-88) < theta_tol, \
           'theta1 does not agree with known solution'
    
    assert math.fabs(sampler.bkg_fit-519) < mu_tol, \
           'background does not agree with known solution'
    
    assert math.fabs(sampler.bkg_std_fit-4.7) < sigma_tol, \
           'background std does not agree with known solution'
    
    fname_fit = f'{obj_name}A-g-fit2p.res'
    fname_star = 'starsA.pkl'
    assert os.path.isfile(fname_fit), \
          'Fitting result is not saved properly.'
    
    assert os.path.isfile(fname_star), \
          'Stars pkl is not saved properly.'

    os.remove(fname_fit)
    os.remove(fname_star)
    shutil.rmtree('plot/')
    shutil.rmtree('Measure-PS1/')
    os.remove(f'{obj_name}.cat')
    os.remove(f'{obj_name}_seg.fits')
    
    return None
