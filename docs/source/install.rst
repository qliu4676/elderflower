Installation
=========================

Dependencies 
------------

``elderflower`` requires the following dependencies:

* numpy>=1.16.2

* scipy>=1.2.1

* matplotlib>=3.0.3

* requests>=2.21.0

* astroquery>=0.3.10

* multiprocess>=0.70.8

* astropy>=4.0

* dynesty>=0.9.7

* galsim>=2.2.3

* photutils>=0.7.2

* pandas>=1.0.1

* pyyaml>=5.3

* joblib>=0.14

* psutil

* tqdm

The above dependencies (except ``galsim``) can be installed by ``pip`` with ``requirement.txt`` in the Github repo of ``elderflower``::

	pip install -r requirements.txt 

To install ``galsim``, try ``pip install galsim``. If it reports errors, it is likely because one of its dependencies (fftw) cannot be installed by ``pip``. If you have ``conda`` installed, it is convenient to install ``galsim`` by::

	conda install -c conda-forge galsim

Or you can install it from source.

To fully run the process, it also needs ``SExtractor`` to be installed. Note using old version of ``SExtractor`` may report errors.

Temporarily, the newly developed Dragonfly data reduction pipeline  (author: Johnny Greco, Allison Merrit, et al.) ``dfreduce`` is required for running the first-step detection on the image. You can intsall it from the dfreduce Github `repository <https://github.com/johnnygreco/DFReduce>`__.

Optionally, the high performance compiler package ``numba`` can be installed (and recommended) to accelerate some numeric functions. This can be done by::

	pip install numba 


Installation
------------
``elderflower`` can be installed by cloning the GitHub `repository <https://github.com/NGC4676/elderflower>`__.
This can be done by running:

.. code-block:: python

	cd <install directory>
	git clone https://github.com/NGC4676/elderflower.git
	cd elderflower
	pip install -e .
