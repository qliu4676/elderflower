import os

# Path
package_dir = os.path.dirname(__file__)
test_dir = os.path.normpath(os.path.join(package_dir, '../tests'))
script_dir = os.path.normpath(os.path.join(package_dir, '../scripts'))
config_dir = os.path.normpath(os.path.join(package_dir, '../configs'))

# Default configuration path
default_config = os.path.join(config_dir, './config.yml')

import re
import sys
import yaml
import string
import shutil
import subprocess
import numpy as np
from datetime import datetime
from functools import partial, wraps

try:
    import dill as pickle
except ImportError:
    import pickle
from pickle import PicklingError

### LOGGING ###
import logging
from astropy.logger import AstropyLogger

class elderflowerLogger(AstropyLogger):
    def reset(self, level='INFO', to_file=None, overwrite=True):
        """ Reset logger. If to_file is given as a string, the output
        will be stored into a log file. """

        for handler in self.handlers[:]:
            self.removeHandler(handler)
            
        self.setLevel(level)
        
        if isinstance(to_file, str) is False:
            # Set up the stdout handlers
            handler = StreamHandler()
            self.addHandler(handler)
            
        else:
            if os.path.isfile(to_file) & overwrite:
                os.remove(to_file)
                
            # Define file handler and set formatter
            file_handler = logging.FileHandler(to_file)
            msg = '[%(asctime)s] %(levelname)s: %(message)s'
            formatter = logging.Formatter(msg, datefmt='%Y-%m-%d|%H:%M:%S')
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
            
        self.propagate = False
    
class StreamHandler(logging.StreamHandler):
    """ A StreamHandler that logs messages in different colors. """
    def emit(self, record):
        stream_print(record.msg, record.levelno)

def stream_print(msg, levelno=logging.INFO):
    """ Enable colored msg using ANSI escape codes based input levelno. """
    levelname = logging.getLevelName(levelno)
    stream = sys.stdout
    
    if levelno < logging.INFO:
        level_msg = '\x1b[1;30m'+levelname+': '+'\x1b[0m'
    elif levelno < logging.WARNING:
        level_msg = '\x1b[1;32m'+levelname+': '+'\x1b[0m'
    elif levelno < logging.ERROR:
        level_msg = '\x1b[1;31m'+levelname+': '+'\x1b[0m'
    else:
        level_msg = levelname+': '
        stream = sys.stderr
        
    print(f'{level_msg}{msg}', file=stream)

logging.setLoggerClass(elderflowerLogger)
logger = logging.getLogger('elderflowerLogger')
logger.reset()

######

def check_save_path(dir_name, overwrite=True, verbose=True):
    """ Check if the input dir_name exists. If not, create a new one.
        If yes, clear the content if overwrite=True. """
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    else:
        if len(os.listdir(dir_name)) != 0:
            if overwrite:
                if verbose: logger.info(f"'{dir_name}' exists. Will overwrite files.")
                #shutil.rmtree(dir_name)
            else:
                while os.path.exists(dir_name):
                    dir_name = input(f"'{dir_name}' exists. Enter a new dir name for saving:")
                    if input("exit"): sys.exit()
            
                os.makedirs(dir_name)
        
    if verbose: logger.info(f"Results will be saved in {dir_name}\n")


def get_executable_path(executable):
    """ Get the execuable path """
    
    command = f'which {executable}'
    check_exe_path = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    
    exe_path = check_exe_path.stdout.read().decode("utf-8").rstrip('\n')
    
    return exe_path


def get_SExtractor_path():
    """ Get the execuable path of SExtractor.
        Possible (sequential) alias: source-extractor, sex, sextractor """
        
    # Check_path
    SE_paths = list(map(get_executable_path,
                        ['source-extractor', 'sex', 'sextractor']))
                        
    # return the first availble path
    try:
        SE_executable = next(path for path in SE_paths if len(path)>0)
        return SE_executable
    except StopIteration:
        logger.error('SExtractor path is not found automatically.')
        return ''

def update_SE_kwargs(kwargs={},
                     kwargs_update={'DETECT_THRESH':3,
                                    'ANALYSIS_THRESH':3}):
    """ Update SExtractor keywords in kwargs """
    
    from .detection import default_conv, default_nnw
    SE_key = kwargs.keys()
    
    for key in kwargs_update.keys():
        if key not in SE_key: kwargs[key] = kwargs_update[key]
        
    if 'FILTER_NAME' not in SE_key : kwargs['FILTER_NAME'] = default_conv
    if 'STARNNW_NAME' not in SE_key : kwargs['STARNNW_NAME'] = default_nnw
    
    for key in ['CHECKIMAGE_TYPE', 'CHECKIMAGE_TYPE']:
        if key in SE_key:
            kwargs.pop(key, None)
            logger.warning(f'{key} is a reserved keyword. Not updated.')
            
    return kwargs


def find_keyword_header(header, keyword,
                        default=None, input_val=False, raise_error=False):
    """ Search keyword value in header (converted to float).
        Accept a value by input if keyword is not found. """
        
    try:
        val = np.float(header[keyword])
     
    except KeyError:
        logger.info(f"Keyname {keyword} missing in the header .")
        
        if input_val:
            try:
                val = np.float(input(f"Input a value of {keyword} :"))
            except ValueError:
                msg = f"Invalid {keyword} values!"
                logger.error(msg)
                raise ValueError(msg)
        elif default is not None:
            logger.info(f"Set {keyword} to default value = ", default)
            val = default
        else:
            if raise_error:
                msg = f"{keyword} must be specified in the header."
                logger.error(msg)
                raise KeyError(msg)
            else:
                return None
            
    return val
    
    
def DateToday():
    """ Today's date in YYYY-MM-DD """
    return datetime.today().strftime('%Y-%m-%d')

def AsciiUpper(N):
    """ ascii uppercase letters """
    return string.ascii_uppercase[:N]
    
def save_pickle(data, filename):
    """ Save data as pickle file. """
    try:
        logger.info(f"Saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except PicklingError:
        logger.error(f"Saving {filename} failed")

def load_pickle(filename):
    """ Load data as pickle file. """
    logger.info(f"Read from {filename}")
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            try:
                out = pickle.load(f)
            except ValueError as err:
                logger.error(err)
                import pickle5
                out = pickle5.load(f)
            return out
    else:
        msg = f'{filename} not found!'
        logger.error(msg)
        raise FileNotFoundError(msg)

def clean_pickling_object(keyword):
    """ Delete pickled objects defined in __main__ to avoid pickling error """
    for variable in dir():
        if keyword in variable:
            del locals()[variable]

def load_config(filename):
    """ Read a yaml configuration. """
    
    if not filename.endswith('.yml'):
        msg = f"Table {filename} is not a yaml file. Exit."
        logger.error(msg)
        sys.exit()
    
    with open(filename, 'r') as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:
            logger.error(err)

def config_kwargs(func, config_file):
    """Wrap keyword arguments from a yaml configuration file."""

    # Load yaml file
    config = load_config(config_file)
    logger.info(f"Loaded configuration file {config_file}")
    
    # Wrap the function
    @wraps(func)
    def wrapper(*args, **kwargs):
        config.update(kwargs)
        return func(*args, **config)

    return wrapper
    
