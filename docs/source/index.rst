.. elderflower documentation master file, created by
   sphinx-quickstart on Mon Aug 24 16:41:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to elderflower's documentation!
=======================================

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
