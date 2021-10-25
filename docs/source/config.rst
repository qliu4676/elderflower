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
	n_spline: 3
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
	print_progress: False	


