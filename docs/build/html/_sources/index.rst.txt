.. elderflower documentation master file, created by
   sphinx-quickstart on Mon Aug 24 16:41:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to elderflower's documentation!
=======================================


.. image:: images/elderflower.png
	:class: no-scaled-link
	:scale: 50%


Introduction
------------

``elderflower`` is a python package to fit the wide-angle point spread function (PSF) in wide-field low-surface brightness images, especially developed for the `Dragonfly telescope <https://www.dragonflytelescope.org/>`__. It mainly utilizes Galsim and Dynesty to generate forward models of PSF for bright stars in the image and fit the parameters of the PSF wing in a Bayesian way. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   tutorial.rst
   example.rst
   config.rst
   api.rst


Why called 'elderflower'?
-------------------------
Well, reason #1 -- I am a lover of elderflower (and a lover of `Fanta Shokata <https://the-soda.fandom.com/wiki/Fanta_Shokata>`__)! 

For more, check the image below taken by the Dragonfly telescope which displays the open cluster M44. Look how similar it is to the beautiful elderflowers!


.. image:: images/M44-1229.png
	:class: no-scaled-link
	:scale: 80%

The optical cleanliness of the lenses in this particular image was intentionally compromised, but it illustrates how big the impact of the scattered light from bright stars (the wide-angle PSF) could be in deep images at low surface brightness levels.


Getting Started
---------------

Here is a simple start-up with a cutout (under ``tests/``) from a Dragonfly image.

.. code-block:: python

    from elderflower.task import berry
    
    # Local path of SExtractor executable
    SE_executable = '/opt/local/bin/source-extractor'

    bounds = ([100,100,700,700])
    elder = berry('cutout.fits', bounds,
		  obj_name='test', band='r',
		  work_dir='./test',
		  config_file='config.yaml') 
    elder.detection(executable=SE_executable)
    elder.run()

.. image:: images/test_2d.png
	:align: center


For further details, please refer to the `Tutorial <tutorial.html>`__ page.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
