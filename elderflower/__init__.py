__version__ = "0.3.3"

# Pixel scale (arcsec/pixel) for reduced and raw Dragonfly data
DF_pixel_scale = 2.5
DF_raw_pixel_scale = 2.85

# Gain (e-/ADU) of Dragonfly
DF_Gain = 0.37

try:
    __SETUP__ = True
except NameError:
    __SETUP__ = False
    

if not __SETUP__:
    from . import io
    from . import numeric
    from . import utils
    from . import modeling

    from . import image
    from . import mask
    from . import crossmatch
    from . import detection
    from . import stack
    from . import norm

    from . import sampler
    from . import container
    from . import task

    from . import plotting
    from . import parallel

    from . import panstarrs
    from . import atlas

