import os
import numpy as np
import matplotlib.pyplot as plt

class Container:
    """ A container storing the prior, the loglikelihood function and fitting setups.
        The container is to be passed to the sampler. """
    
    def __init__(self,
                 n_spline=2,
                 fit_sigma=True,
                 fit_frac=False,
                 leg2d=False,
                 brightest_only=False,
                 parallel=False,
                 draw_real=True):
        
        self.n_spline = n_spline
        self.fit_sigma = fit_sigma
        self.fit_frac = fit_frac
        self.leg2d = leg2d
        
        self.brightest_only = brightest_only
        self.parallel = parallel
        self.draw_real = draw_real
        
    def set_prior(self, n_est, mu_est, std_est,
                  n_min=1, theta_in=50, theta_out=240):
        """ Setup priors for fitting and labels for displaying the results"""
        from modeling import set_prior
        from utils import set_labels
    
        prior_tf = set_prior(n_est, mu_est, std_est,
                             n_spline=self.n_spline, leg2d=self.leg2d,
                             fit_sigma=self.fit_sigma, fit_frac=self.fit_frac,
                             n_min=1, theta_in=50, theta_out=240)
        
        self.prior_transform = prior_tf

        labels = set_labels(n_spline=self.n_spline, leg2d=self.leg2d,
                            fit_sigma=self.fit_sigma, fit_frac=self.fit_frac)
        
        self.labels = labels

        ndim = len(labels)
        self.ndim = ndim
    
    def set_likelihood(self, Y, mask_fit,
                       psf_tri, stars_tri,
                       norm='brightness',
                       z_norm=None,
                       psf_range=[None, None],
                       image_base=None):
        """ Setup likelihood function for fitting """
        from modeling import set_likelihood
        
        if image_base is None:
            image_base = np.zeros_like(mask_fit)
        
        self.image_base = image_base
        
        loglike = set_likelihood(Y, mask_fit,
                                 psf_tri, stars_tri,
                                 psf_range=psf_range,
                                 image_base=image_base,
                                 n_spline=self.n_spline,
                                 leg2d=self.leg2d,
                                 fit_sigma=self.fit_sigma,
                                 fit_frac=self.fit_frac,
                                 norm='brightness',
                                 z_norm=z_norm,
                                 brightest_only=self.brightest_only,
                                 parallel=self.parallel, 
                                 draw_real=self.draw_real)
        
        self.loglikelihood = loglike