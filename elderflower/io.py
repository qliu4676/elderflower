import os
import re
import sys
import pickle
import yaml
import numpy as np
from functools import partial, wraps

package_dir = os.path.dirname(__file__)
config_dir = os.path.normpath(os.path.join(package_dir, '../configs'))


def check_save_path(dir_name, make_new=True, verbose=True):
    """ Check if the input dir_name exists. If not, create a new one.
        If yes, clear the content if make_new=True. """
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    elif make_new:
        if len(os.listdir(dir_name)) != 0:
            while os.path.exists(dir_name):
                dir_name = input("'%s' already existed. Enter a directory name for saving:"%dir_name)
            os.makedirs(dir_name)
            
    if verbose: print("Results will be saved in %s\n"%dir_name)
    
    
def find_keyword_header(header, keyword):
    """ Search keyword value in header (converted to float).
        Input a value by user if keyword is not found. """
        
    try:
        val = np.float(header[keyword])
     
    except KeyError:
        print("%s missing in header --->"%keyword)
        
        try:
            val = np.float(input("Input a value of %s :"%keyword))
        except ValueError:
            sys.exit("Invalid %s values!"%keyword)
            
    return val
    
    
def save_pickle(data, filename):
    """ Save data as pickle file. """
    
    fname = filename+'.pkl'
    print("Save data to %s"%fname)
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """ Load data as pickle file. """

    fname = filename+'.pkl'
    print("Read data from %s"%fname)
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_config(filename):
    """ Read a yaml configuration. """
    
    if not filename.endswith('.yaml'):
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
    
