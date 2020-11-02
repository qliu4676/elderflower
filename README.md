# elderflower

<img src="docs/source/images/elderflower.png" width="30%">

**ELDERFLOWER** (Mod**E**l**L**ing Wi**DE**-Angle Point Sp**R**ead **F**unction in **LOW** Surfac**E** B**R**ightness) is a wide-angle PSF modeling tool for low-surface brightness imaging with Dragonfly that utilizes [Galsim](https://github.com/GalSim-developers/GalSim) and [Dynesty](https://github.com/joshspeagle/dynesty).


Doucmentation: https://elderflower.readthedocs.io/en/latest/

Installation
------------

```bash
  cd <install directory>
  git clone https://github.com/NGC4676/elderflower.git
  cd elderflower
  pip install -e .
```

Basic Usage
-----------
1. Function mode
```python
bounds = ([100,100,700,700])
obj_name = 'test'
filt = 'r'

ZP = Run_Detection('cutout.fits', obj_name, filt)
Match_Mask_Measure('cutout.fits', bounds, obj_name, filt,
                    ZP=ZP, pixel_scale=2.5)
samplers = Run_PSF_Fitting('cutout.fits', bounds, obj_name, filt,
                            n_spline=3, ZP=ZP, pixel_scale=2.5)   
```

2. Configuration mode
```python
from elderflower.task import berry

bounds = ([100,100,700,700])
elder = berry('cutout.fits', bounds,
              obj_name='test', filt='r',
              config_file='config.yaml')
elder.detection()
elder.run()
```

See doucementaton for paramter retrievement and bright star subtraction using ``samplers``.
