import os
import numpy as np
import matplotlib.pyplot as plt

class Container:
    """ A container storing the prior, the loglikelihood function and fitting data & setups.
        The container is to be passed to the sampler. """
    
    def __init__(self,
                 n_spline=2,
                 leg2d=False,
                 fit_sigma=True,
                 fit_frac=False,
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
        
    def __str__(self):
        return "A Container Class"

    def __repr__(self):
        if hasattr(self, 'ndim'):
            return f"{self.__class__.__name__} p={self.ndim}"
        else:
            return f"{self.__class__.__name__}"
            
        
    def set_prior(self, n_est, mu_est, std_est,
                  n_min=1, theta_in=50, theta_out=300):
        """ Setup priors for fitting and labels for displaying the results"""
        from .modeling import set_prior
        
        fit_sigma = self.fit_sigma
        fit_frac = self.fit_frac
        
        n_spline = self.n_spline
        leg2d = self.leg2d
    
        prior_tf = set_prior(n_est, mu_est, std_est,
                             n_spline=n_spline, n_min=n_min, leg2d=leg2d,
                             theta_in=theta_in, theta_out=theta_out,
                             fit_sigma=fit_sigma, fit_frac=fit_frac)
        
        self.prior_transform = prior_tf

        labels = set_labels(n_spline=n_spline, leg2d=leg2d,
                            fit_sigma=fit_sigma, fit_frac=fit_frac)
        
        self.labels = labels
        self.ndim = len(labels)
        
        self.std_est = std_est
    
    def set_likelihood(self,
                       data, mask_fit,
                       psf, stars,
                       norm='brightness',
                       psf_range=[None, None],
                       G_eff=None,
                       image_base=None):
                       
        """ Setup likelihood function for fitting """
        
        from .modeling import set_likelihood
        
        if image_base is None:
            image_base = np.zeros_like(mask_fit)
        
        self.image_base = image_base
        
        # Copy psf and stars to preserve the orginal ones
        stars_tri = stars.copy()
        psf_tri = psf.copy()
        
        # Set up likelihood function
        loglike = set_likelihood(data, mask_fit,
                                 psf_tri, stars_tri,
                                 norm=norm,
                                 psf_range=psf_range,
                                 std_est=self.std_est,
                                 G_eff=G_eff,
                                 image_base=image_base,
                                 n_spline=self.n_spline,
                                 leg2d=self.leg2d,
                                 fit_sigma=self.fit_sigma,
                                 fit_frac=self.fit_frac,
                                 brightest_only=self.brightest_only,
                                 parallel=self.parallel, 
                                 draw_real=self.draw_real)
        
        self.loglikelihood = loglike
        
        
def set_labels(n_spline, fit_sigma=True, fit_frac=False, leg2d=False):
    
    """ Setup labels for cornerplot """
    
    K = 0
    if fit_frac: K += 1
    if fit_sigma: K += 1
    
    if n_spline=='m':
        labels = [r'$\gamma_1$', r'$\beta_1$']
    elif n_spline==1:
        labels = [r'$n0$']
    elif n_spline==2:
        labels = [r'$n0$', r'$n1$', r'$\theta_1$']
    elif n_spline==3:
        labels = [r'$n0$', r'$n1$', r'$n2$', r'$\theta_1$', r'$\theta_2$']
    else:
        labels = [r'$n_%d$'%d for d in range(n_spline)] \
               + [r'$\theta_%d$'%(d+1) for d in range(n_spline-1)]
        
    labels += [r'$\mu$']
        
    if leg2d:
        labels.insert(-1, r'$\log\,A_{01}$')
        labels.insert(-1, r'$\log\,A_{10}$')
    
    if fit_sigma:
        labels += [r'$\log\,\sigma$']
        
    if fit_frac:
        labels += [r'$\log\,f$']
        
    return labels
