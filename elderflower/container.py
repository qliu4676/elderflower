import os
import numpy as np
import matplotlib.pyplot as plt
from .io import logger

class Container:
    """
    A container storing the prior, the loglikelihood function and
    fitting data & setups. The container is passed to the sampler.
    """
    
    def __init__(self,
                 n_spline=2,
                 leg2d=False,
                 fit_sigma=True,
                 fit_frac=False,
                 brightest_only=False,
                 parallel=False,
                 draw_real=True):
                 
        if n_spline is float:
            if n_spline <=1:
                sys.exit('n_spline needs to be >=2!')
        
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
                  n_min=1.2, d_n0=0.2,
                  theta_in=50, theta_out=300):
                  
        """ Setup priors for Bayesian fitting."""
        
        from .modeling import set_prior
        
        fit_sigma = self.fit_sigma
        fit_frac = self.fit_frac
        
        n_spline = self.n_spline
        leg2d = self.leg2d
    
        prior_tf = set_prior(n_est, mu_est, std_est,
                             n_spline=n_spline, leg2d=leg2d,
                             n_min=n_min, d_n0=d_n0, fix_n0=self.fix_n0,
                             theta_in=theta_in, theta_out=theta_out,
                             fit_sigma=fit_sigma, fit_frac=fit_frac)
        
        self.prior_transform = prior_tf
        
        # Set labels for displaying the results
        labels = set_labels(n_spline=n_spline, leg2d=leg2d,
                            fit_sigma=fit_sigma, fit_frac=fit_frac)
        
        self.labels = labels
        self.ndim = len(labels)
        
        self.n_est = n_est
        self.mu_est = mu_est
        self.std_est = std_est
    
    def set_MLE_bounds(self, n_est, mu_est, std_est,
                       n_min=1.2, d_n0=0.2,
                       theta_in=50, theta_out=300):
        
        """ Setup p0 and bounds for MLE fitting """
        
        for option in ['fit_sigma', 'fit_frac', 'leg2d']:
            if getattr(self, option):
                logger.warning(f"{option} not supported in MLE. Will be turned off.")
                exec('self.' + option + ' = False')
            
        n0 = n_est
        n_spline = self.n_spline

        log_t_in, log_t_out = np.log10(theta_in), np.log10(theta_out)
        log_theta_bounds = [(log_t_in, log_t_out) for i in range(n_spline-1)]
        
        bkg_bounds = [(mu_est-3*std_est, mu_est+3*std_est)]
        
        if n_spline == 2:
            self.param0 = np.array([n0, 2.2, 1.8, mu_est])
            n_bounds = [(n0-d_n0, n0+d_n0), (n_min, 3.)]
        elif n_spline == 3:
            self.param0 = np.array([n0, 2.5, 2., 1.8, 2., mu_est])
            n_bounds = [(n0-d_n0, n0+d_n0), (2., 3.), (n_min, 2+d_n0)]
        else:
            n_guess = np.linspace(3., n_min, n_spline-1)
            theta_guess = np.linspace(1.8, log_t_out-0.3, n_spline-1)
            self.param0 = np.concatenate([[n0], n_guess, theta_guess, [mu_est]])
    
            n_bounds = [(n0-d_n0, n0+d_n0), (2., n0-d_n0)] + [(n_min, n0-d_n0) for i in range(n_spline-2)]
            
            logger.warning("Components > 3. The MLE might reach maxiter or maxfev.")

        self.MLE_bounds = tuple(n_bounds + log_theta_bounds + bkg_bounds)

        self.n_est = n_est
        self.mu_est = mu_est
        self.std_est = std_est
        
        self.ndim = 2 * n_spline
        
    def set_likelihood(self,
                       image, mask_fit,
                       psf, stars,
                       norm='brightness',
                       psf_range=[None, None],
                       G_eff=1e5,
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
        loglike = set_likelihood(image, mask_fit,
                                 psf_tri, stars_tri,
                                 norm=norm,
                                 psf_range=psf_range,
                                 fix_n0=self.fix_n0,
                                 std_est=self.std_est,
                                 G_eff=G_eff,
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
