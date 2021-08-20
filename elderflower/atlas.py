"""
Query HLSP-Atlas using Casjobs Command Line tool
"""

import os
import glob
import subprocess
import numpy as np
from shutil import copyfile

from .io import script_dir, config_dir

default_atlas_config = os.path.join(config_dir, './casjobs.config')
exe_path = os.path.join(script_dir, 'casjobs.jar')

def query_atlas(ra_range, dec_range, wsid, password, mag_limit=16):
    
    # make a temp directory and copy atlas command line tool
    os.makedirs('ATLAS', exist_ok=True)
    os.chdir('ATLAS')
    copyfile(exe_path, 'casjobs.jar')
    copyfile(default_atlas_config, 'casjobs.config')
    
    # replace config wsid and password
    with open('casjobs.config', "r") as f:
        config = f.read()
        config = config.replace('YOUR_WEBSERVICES_ID', str(wsid))
        config = config.replace('YOUR_PASSWORD', str(password))

    with open('casjobs.config', "w") as f:
        f.write(config)
    
    # write script
    ra_min, ra_max = np.around([ra_range[0], ra_range[1]], 5)
    dec_min, dec_max = np.around([dec_range[0], dec_range[1]], 5)
    
    casjobs_script = f"""#!/bin/bash
casjobs_path=casjobs.jar

java -jar $casjobs_path execute "select RA, Dec, g, r from refcat2 into mydb.atlas where ra between {ra_min} and {ra_max} and dec between {dec_min} and {dec_max} and g <= {mag_limit}"
java -jar $casjobs_path extract -b atlas -F -type csv -d
java -jar $casjobs_path execute -t "mydb" -n "drop query" "drop table atlas"
"""
    with open('casjobs_atlas.sh', 'w') as f:
        f.write(casjobs_script)
    
    # run script
    out = subprocess.Popen('sh casjobs_atlas.sh', stdout=subprocess.PIPE, shell=True)
    os.chdir('../')
    
    # rename
    table = glob.glob('ATLAS/*.csv')[0]
    os.rename(table, 'ATLAS/cat_atlas.csv')
    
    return out
