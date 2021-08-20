.. elderflower documentation master file, created by
   sphinx-quickstart on Mon Aug 24 16:41:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to elderflower's documentation!
=======================================


.. image:: images/elderflower.png
	:class: no-scaled-link
	:scale: 20%


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
Well, reason #1 -- I am an elderflower lover (also a lover of Fanta Shokata)! 

For another, check the image below which displays the open cluster M44 taken by the Dragonfly telescope. Look how similar it is to the beautiful picture of elderflower above!


.. image:: images/M44-1229.png
	:class: no-scaled-link
	:scale: 80%

The optical cleanliness of the lenses was bad but it illustrates how big the impact of the scattered light from bright stars (the wide-angle PSF) could be in deep images at low surface brightness levels.


Getting Started
---------------

.. code-block:: python

    from elderflower.task import berry
    
    # Local path of SExtractor executable
    SE_executable = '/opt/local/bin/source-extractor'

    bounds = ([100,100,700,700])
    elder = berry('cutout.fits', bounds,
		  obj_name='test', filt='r',
		  work_dir='./test',
		  config_file='config.yaml') 
    elder.detection(executable=SE_executable)
    elder.run()

.. image:: images/test_2d.png
	:align: center


For further examples in detail, refer to examples page.
For parameters tweaking, refer to configuration page.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
