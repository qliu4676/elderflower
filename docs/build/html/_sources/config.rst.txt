Configuration
=============

An example configuration file looks as follows:

.. code-block:: yaml

	# General
	pixel_scale: 2.5        # arcsec/pix
	r_scale: 12             # pix
	mag_limit: 15
	draw: True
	save: True
	ZP: ~
	pad: 50

	# crossmatch
	mag_saturate: 13.5      # estimate, not required to be accurate
	use_PS1_DR2: False
	field_pad: 50

	# fitting
	mag_threshold: [13,11]  # MB /VB
	n_spline: 2
	cutoff: False
	n_cutoff: 4
	theta_cutoff: 1200

	theta_0: 5
	fit_n0: True
	fit_sigma: True
	fit_frac: False
	leg2d: False

	# mask
	r_core: 24              
	r_out: ~
	wid_strip: 24
	n_strip: 48
	mask_obj: ~

	# sampling
	brightest_only: False
	draw_real: True
	parallel: False
	n_cpu: 4
	nlive_init: ~
	sample_method: 'auto'
	print_progress: True	

Tips
----

To run ``elderflower`` we need some prior knowledge about the PSF and set the parameters properly.

A key assumption here is that the aureole follows a multi-power law. Be careful if the PSF presents a clear bump or non-neglibile artifacts -- the parametric form may not has a good representation.

- ``r_scale`` is the radius at which the intensity is measured as normalization. Make sure ``r_scale`` falls in the wing (outside the core and saturation). It is recommended to have ``r_scale`` to be small so that the S/N of intensity is high.

- ``theta_0`` is the inner flattening radius of the aureole. This should be small to avoid biasing the inner parts. 5 arcsec is used because Dragonfly has large pixel size.

- ``mag_threshold`` sets the thresholds defining the MB and VB stars. For a fast mode, set them to be lower. Increase if the region lacks bright stars (but not higher than `mag_limit`). Note it is recommended to pick regions with a decent number of bright stars.

- ``n_spline`` sets the number of components of the aureole. For a fast start-up, try ``n_spline=2``. Increase if the residual suggests a higher complexity is needed.

- ``fit_n0=False`` will fix the first component (n0) to be the value from profile fitting. This is usually set when n_spline>=3 to 1) avoid local minimum from too many parameters, and 2)save time.

- ``use_PS1_DR2=True`` will use the PAN-STARRS DR2 as the crossmatch. DR2 has a better performance in crossmatch in but the current MAST query approach using the PS1 API might result in HTTP Error in the case that the image is really large. Stars brighter than 7~8 mag may still be affected. A supplementary crossmatch with HLSP-ATLAS catalog improves (under implementation).

- ``cutoff=True`` applies a sharp cutoff (n_cutoff) beyond the outermost components (> theta_cutoff). Often in actual data the scattered starlight does not extend to infinity because some scale of sky has been subtracted in the reduction step. Try this if the scattered light looks too extended.

- ``r_core`` sets the mask aperture size for MB and/or VB stars because the cores are not well modelled. This can be a list: [VB, MB].

- Set ``brightest_only=False`` for a faster fitting on VB stars only. Sometimes useful when normalization for MB stars are not good.

- Set ``draw=False`` to disable the redundant plots. They can be generated with the output.

