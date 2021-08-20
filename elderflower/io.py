import os
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

package_dir = os.path.dirname(__file__)
test_dir = os.path.normpath(os.path.join(package_dir, '../tests'))
script_dir = os.path.normpath(os.path.join(package_dir, '../scripts'))
config_dir = os.path.normpath(os.path.join(package_dir, '../configs'))

# Default configuration path
default_config = os.path.join(config_dir, './config.yml')

def check_save_path(dir_name, make_new=True, verbose=True):
    """ Check if the input dir_name exists. If not, create a new one.
        If yes, clear the content if make_new=True. """
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    else:
        if len(os.listdir(dir_name)) != 0:
            if make_new:
                if verbose: print("'%s' already existed. Overwrite files."%dir_name)
                shutil.rmtree(dir_name)
            else:
                while os.path.exists(dir_name):
                    dir_name = input("'%s' already existed. Enter a directory name for saving:"%dir_name)
            
            os.makedirs(dir_name)
        
    if verbose: print("Results will be saved in %s\n"%dir_name)


def get_executable_path(executable):
    """ Get the execuable path """
    
    check_exe_path = subprocess.Popen(f'which {executable}', stdout=subprocess.PIPE, shell=True)
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
        print('Warning: SExtractor path is not found automatically.')
        return ''
    

def find_keyword_header(header, keyword,
                        default=None, input_val=False, raise_error=False):
    """ Search keyword value in header (converted to float).
        Input a value by user if keyword is not found. """
        
    try:
        val = np.float(header[keyword])
     
    except KeyError:
        print("%s missing in header --->"%keyword)
        
        if input_val:
            try:
                val = np.float(input("Input a value of %s :"%keyword))
            except ValueError:
                raise ValueError("Invalid %s values!"%keyword)
        elif default is not None:
            print(f'Set {keyword} to default value = ', default)
            val = default
        else:
            if raise_error:
                raise KeyError("%s needs to be specified in the keywords."%keyword)
            else:
                return None
            
    return val
    
    
def DateToday():
    """ Today's date in YYYY-MM-DD """
    return datetime.today().strftime('%Y-%m-%d')

def AsciiUpper(N):
    """ ascii uppercase letters """
    return string.ascii_uppercase[:N]
    
def save_pickle(data, filename, printout=True):
    """ Save data as pickle file. """
    try:
        if printout: print("Saved to %s"%filename)
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except PicklingError:
        if printout: print("Saving %s failed"%filename)

def load_pickle(filename, printout=True):
    """ Load data as pickle file. """
    if printout: print("Read from %s"%filename)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            try:
                out = pickle.load(f)
            except ValueError as err:
                print(err)
                import pickle5
                out = pickle5.load(f)
            return out
    else:
        raise FileNotFoundError(f'{filename} not found!')


def load_config(filename):
    """ Read a yaml configuration. """
    
    if not filename.endswith('.yml'):
        sys.exit(f"Table {filename} is not a yaml file. Exit.")
    
    with open(filename, 'r') as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:
            print(err)

def config_kwargs(func, config_file):
    """Wrap keyword arguments from a yaml configuration file."""

    # Load yaml file
    config = load_config(config_file)
    print(f"Loaded configuration file {config_file}")
    
    # Wrap the function
    @wraps(func)
    def wrapper(*args, **kwargs):
        config.update(kwargs)
        return func(*args, **config)

    return wrapper
    
