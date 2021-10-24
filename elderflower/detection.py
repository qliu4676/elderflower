"""
Submodule for running SExtractor from dfreduce (credit: Johnny Greco).
Will be replaced by an independent module.
"""

import os
from subprocess import call

import numpy as np
from astropy.io import ascii, fits

from .io import logger
from .io import config_dir

# default SExtractor paths and files
input_file_path = os.path.join(config_dir, 'sextractor')
kernel_path = os.path.join(input_file_path, 'kernels')
default_SE_config = os.path.join(input_file_path, 'default.sex')
default_param_file = os.path.join(input_file_path, 'default.param')
default_run_config = os.path.join(input_file_path, 'default.config')
default_nnw = os.path.join(input_file_path, 'default.nnw')
default_conv = os.path.join(kernel_path, 'default.conv')

# get list of all config options
with open(os.path.join(input_file_path, 'config_options.txt'), 'r') as file:
    all_option_names = [line.rstrip() for line in file]

# get list of all SExtractor measurement parameters
default_params = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUX_RADIUS',
                  'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'FWHM_IMAGE', 'FLAGS']
with open(os.path.join(input_file_path, 'all_params.txt'), 'r') as file:
    all_param_names = [line.rstrip() for line in file]

# default non-standard options
default_options = dict(
    BACK_SIZE=128,
    DETECT_THRESH=4,
    DETECT_MINAREA=4,
    ANALYSIS_THRESH=4,
    GAIN=0.37,
    PHOT_APERTURES=6,
    PIXEL_SCALE=2.85,
    SEEING_FWHM=2.5,
    VERBOSE_TYPE='QUIET',
    MEMORY_BUFSIZE=4096,
    MEMORY_OBJSTACK=30000,
    MEMORY_PIXSTACK=3000000,
    PARAMETERS_NAME=default_param_file,
    FILTER_NAME=default_conv
)

def temp_fits_file(path_or_pixels, tmp_path='/tmp', run_label=None,
               prefix='tmp',  header=None):
    is_str = type(path_or_pixels) == str or type(path_or_pixels) == np.str_
    if is_str and header is None:
        path = path_or_pixels
        created_tmp = False
    else:
        if is_str:
            path_or_pixels = fits.getdata(path_or_pixels)
        label = '' if run_label is None else '_' + run_label
        fn = '{}{}.fits'.format(prefix, label)
        path = os.path.join(tmp_path, fn)
        fits.writeto(path, path_or_pixels, header=header, overwrite=True)
        created_tmp = True
    return path, created_tmp

def is_list_like(check):
    t = type(check)
    c = t == list or t == np.ndarray or t == pd.Series or t == pd.Int64Index
    return c

def list_of_strings(str_or_list):
    """
    Return a list of strings from a single string of comma-separated values.
    """
    if is_list_like(str_or_list):
        ls_str = str_or_list
    elif type(str_or_list) == str:
        ls_str = str_or_list.replace(' ', '').split(',')
    else:
        Exception('{} is not correct type for list of str'.format(str_or_list))
    return ls_str

def run(path_or_pixels, catalog_path=None, config_path=default_run_config,
        executable='source-extractor', tmp_path='/tmp', run_label=None, header=None,
        extra_params=None, **sextractor_options):
    """
    Run SExtractor.

    Parameters
    ----------
    path_or_pixels : str
        Full path file name to the fits image -- or -- The image pixels as
        a numpy array. In the latter case, a temporary fits file will be
        written in tmp_path with an optional run_label to make the temp file
        name unique (this is useful if you are running in parallel).
    catalog_path : str (optional)
        If not None, the full path file name of the output catalog. If None,
        a temporary catalog will be written in tmp_path with a
        run_label (if it's not None).
    config_path : str (optional)
        Full path SExtractor configuration file name.
    executable : str (optional)
        The SExtractor executable name (full path if necessary)
    tmp_path : str (optional)
        Path for temporary fits files if you pass image pixels to
        this function.
    run_label : str (optional)
        A unique label for the temporary files.
    header : astropy.io.fits.Header (optional)
        Image header if you pass image pixels to this function and want
        SExtractor to have the header information.
    extra_params: str or list-like (optional)
        Additional SE measurement parameters. The default parameters, which
        are always in included, are the following:
        X_IMAGE, Y_IMAGE, FLUX_AUTO, FLUX_RADIUS, FWHM_IMAGE, A_IMAGE,
        B_IMAGE, THETA_IMAGE, FLAGS
    **sextractor_options: Keyword arguments
        Any SExtractor configuration option.
    
    Returns
    -------
    catalog : astropy.Table
        The SExtractor source catalog.

    Notes
    -----
    You must have SExtractor installed to run this function.

    The 'sextractor_options' keyword arguments may be passed one at a time or
    as a dictionary, exactly the same as **kwargs.

    Example:

    # like this
    cat = sextractor.run(image_fn, cat_fn, FILTER='N', DETECT_THRESH=10)

    # or like this
    options = dict(FILTER='N', DETECT_THRESH=10)
    cat = sextractor.run(image_fn, cat_fn, **options)

    # extra_params can be given in the following formats
    extra_params = 'FLUX_RADIUS'
    extra_params = 'FLUX_RADIUS,ELLIPTICITY'
    extra_params = 'FLUX_RADIUS, ELLIPTICITY'
    extra_params = ['FLUX_RADIUS', 'ELLIPTICITY']
    # (it is case-insensitive)
    """
    image_path, created_tmp = temp_fits_file(path_or_pixels,
                                             tmp_path=tmp_path,
                                             run_label=run_label,
                                             prefix='se_tmp',
                                             header=header)

    logger.debug('Running SExtractor on ' + image_path)

    # update config options
    final_options = default_options.copy()
    for k, v in sextractor_options.items():
        k = k.upper()
        if k not in all_option_names:
            msg = '{} is not a valid SExtractor option -> we will ignore it!'
            logger.warning(msg.format(k))
        else:
            logger.debug('SExtractor config update: {} = {}'.format(k, v))
            final_options[k] = v

    # create catalog path if necessary
    if catalog_path is not None:
        cat_name = catalog_path
        save_cat = True
    else:
        label = '' if run_label is None else '_' + run_label
        cat_name = os.path.join(tmp_path, 'se{}.cat'.format(label))
        save_cat = False

    # create and write param file if extra params were given
    param_fn = None
    if extra_params is not None:
        extra_params = list_of_strings(extra_params)
        params = default_params.copy()
        for par in extra_params:
            p = par.upper()
            _p = p[:p.find('(')] if p.find('(') > 0 else p
            if _p not in all_param_names:
                msg = '{} is not a valid SExtractor param -> we will ignore it!'
                logger.warning(msg.format(p))
            elif _p in default_params:
                msg = '{} is a default parameter -> No need to add it!'
                logger.warning(msg.format(p))
            else:
                params.append(p)
        if len(params) > len(default_params):
            label = '' if run_label is None else '_' + run_label
            param_fn = os.path.join(tmp_path, 'params{}.se'.format(label))
            with open(param_fn, 'w') as f:
                logger.debug('Writing parameter file to ' + param_fn)
                print('\n'.join(params), file=f)
        final_options['PARAMETERS_NAME'] = param_fn

    # build shell command
    cmd = executable + ' -c {} {}'.format(config_path, image_path)
    cmd += ' -CATALOG_NAME ' + cat_name
    for k, v in final_options.items():
        cmd += ' -{} {}'.format(k.upper(), v)
    if param_fn is not None:
        cmd += ' -PARAMETERS_NAME ' + param_fn

    # run it
    logger.debug(f'>> {cmd}')
    call(cmd, shell=True)

    if 'CATALOG_TYPE' not in final_options.keys():
        catalog = ascii.read(cat_name)
    elif final_options['CATALOG_TYPE'] == 'ASCII_HEAD':
        catalog = ascii.read(cat_name)
    else:
        catalog = None

    if created_tmp:
        logger.debug('Deleting temporary file ' + image_path)
        os.remove(image_path)
    if param_fn is not None:
        logger.debug('Deleting temporary file ' + param_fn)
        os.remove(param_fn)
    if not save_cat:
        logger.debug('Deleting temporary file ' + cat_name)
        os.remove(cat_name)

    return catalog
