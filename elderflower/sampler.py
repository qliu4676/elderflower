import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from .io import save_pickle, load_pickle
from .plotting import colorbar


class Sampler:

    def __init__(self, container,
                 sample='auto', bound='multi',
                 n_cpu=None, n_thread=None,
                 run=True, results=None):
                 
        """ A class for runnning the sampling and plotting results """
                 
        # False if a previous run is read
        self.run = run
        
        self.container = container
        self.image = container.image
        self.ndim = container.ndim
        self.labels = container.labels
        
        if run:
            if n_cpu is None:
                n_cpu = min(mp.cpu_count()-1, 10)
                
            if n_thread is not None:
                n_thread = max(n_thread, n_cpu-1)
            
            if n_cpu > 1:
                self.open_pool(n_cpu)
                self.use_pool = {'update_bound': False}
            else:
                self.pool = None
                self.use_pool = None
            
            self.prior_tf = container.prior_transform
            self.loglike = container.loglikelihood
            
            dsampler = dynesty.DynamicNestedSampler(self.loglike, self.prior_tf, self.ndim,
                                                    sample=sample, bound=bound,
                                                    pool=self.pool, queue_size=n_thread,
                                                    use_pool=self.use_pool)
            self.dsampler = dsampler
            
        else:
            self._results = results # use existed results
        

    def run_fitting(self,
                    nlive_init=100,
                    maxiter=10000,
                    nlive_batch=50,
                    maxbatch=2,
                    wt_kwargs={'pfrac': 0.8},
                    close_pool=True,
                    print_progress=True):
        
        if not self.run: return None
        
        print("Run Nested Fitting for the image... Dim of params: %d"%self.ndim)
        start = time.time()
   
        dlogz = 1e-3 * (nlive_init - 1) + 0.01
        
        self.dsampler.run_nested(nlive_init=nlive_init, 
                                 nlive_batch=nlive_batch, 
                                 maxbatch=maxbatch,
                                 maxiter=maxiter,
                                 dlogz_init=dlogz, 
                                 wt_kwargs=wt_kwargs,
                                 print_progress=print_progress) 
        
        end = time.time()
        self.run_time = (end-start)
        
        print("\nFinish Fitting! Total time elapsed: %.3g s"%self.run_time)
        
        if (self.pool is not None) & close_pool:
            self.close_pool()
        
    def open_pool(self, n_cpu):
        print("\nOpening new pool: # of CPU used: %d"%(n_cpu))
        self.pool = mp.Pool(processes=n_cpu)
        self.pool.size = n_cpu
    
    def close_pool(self):
        print("\nPool Closed.")
        self.pool.close()
        self.pool.join()

    @property
    def results(self):
        """ Results of the dynesty dynamic sampler class """
        if self.run:
            return getattr(self.dsampler, 'results', {})
        else:
             return self._results
    
    def get_params(self, return_sample=False):
        return get_params_fit(self.results, return_sample)
    
    def save_results(self, filename, fit_info=None, save_dir='.'):
        """ Save fitting results """
        
        if not self.run: return None
        
        res = {}
        if fit_info is not None:
            res['fit_info'] = {'run_time': round(self.run_time,2)}
            for key, val in fit_info.items():
                res['fit_info'][key] = val
        
        res['fit_res'] = self.results         # fitting results
        res['container'] = self.container     # a container for prior and likelihood
        
        # Delete local prior and loglikelihood function which can't be pickled
        for attr in ['prior_transform', 'loglikelihood']:
            delattr(res['container'], attr)
        
        save_pickle(res, os.path.join(save_dir, filename))
        
    @classmethod
    def load_results(cls, filename):
        """ Read saved fitting results """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = load_pickle(filename, printout=False)
            
        print(f"Read fitting results {filename}\n", res['fit_info'])
        
        return cls(res['container'], run=False, results=res['fit_res'])
        
    
    def cornerplot(self, truths=None, figsize=(16,15),
                   save=False, save_dir='.', suffix='', **kwargs):
        from .plotting import draw_cornerplot
        
        draw_cornerplot(self.results, self.ndim,
                        labels=self.labels, truths=truths, figsize=figsize,
                        save=save, save_dir=save_dir, suffix=suffix, **kwargs)
        
    def cornerbounds(self, figsize=(10,10),
                    save=False, save_dir='.', suffix='', **kwargs):
        from .plotting import draw_cornerbounds
        
        if hasattr(self, 'prior_tf'):
            draw_cornerbounds(self.results, self.ndim, self.prior_tf,
                              labels=self.labels, figsize=figsize,
                              save=save, save_dir=save_dir, suffix=suffix, **kwargs)
                                    
    
    def plot_fit_PSF1D(self, psf, **kwargs):
        from .plotting import plot_fit_PSF1D
        n_spline = self.container.n_spline
        
        plot_fit_PSF1D(self.results, psf, n_spline=n_spline, **kwargs)
    
    def generate_fit(self, psf, stars, norm='brightness'):
    
        """ Build psf from fitting results """
        
        from .utils import make_psf_from_fit
        from .modeling import generate_image_fit
        
        ct = self.container
        image_size = ct.image_size
        image_base = ct.image_base
        
        psf_fit, params = make_psf_from_fit(self.results, psf,
                                            n_spline=ct.n_spline,
                                            leg2d=ct.leg2d,
                                            sigma=ct.std_est,
                                            fit_sigma=ct.fit_sigma,
                                            fit_frac=ct.fit_frac)
        
        self.bkg_fit = psf_fit.bkg
        self.bkg_std_fit = psf_fit.bkg_std
        
        
        image_stars, image_fit, bkg_image \
                   = generate_image_fit(psf_fit, stars, image_size,
                                        norm=norm, leg2d=ct.leg2d,
                                        brightest_only=ct.brightest_only,
                                        draw_real=ct.draw_real)

        noise_image = image_fit - image_stars

        image_fit += image_base + bkg_image
        
        # Images constructed from fitting
        self.image_fit = image_fit
        self.image_stars = image_stars
        self.bkg_image = bkg_image
        self.noise_image = noise_image
        
        # PSF constructed from fitting
        self.psf_fit = psf_fit
        
    
    def calculate_reduced_chi2(self, Gain=456.95):
        
        """Calculate reduced Chi^2"""
        
        from .utils import calculate_reduced_chi2
        
        ct = self.container
        mask_fit = getattr(ct.mask, 'mask_comb', ct.mask.mask_deep)
        data = ct.data
        
        data_pred = (self.image_fit[~mask_fit]).ravel()
        
        uncertainty = np.sqrt(self.bkg_std_fit**2+(data-self.bkg_fit)/Gain)
        
        calculate_reduced_chi2(data_pred, data, uncertainty)
        
    def draw_comparison_2D(self, **kwargs):
        from .plotting import draw_comparison_2D
        
        ct = self.container
        image = ct.image
        mask = ct.mask
        
        if hasattr(self, 'image_fit'):
            draw_comparison_2D(self.image_fit, image, mask, self.image_stars,
                               self.noise_image, **kwargs)
        
    def draw_background(self, save=False, save_dir='.', suffix=''):
        plt.figure()
        if hasattr(self, 'bkg_image'):
            im = plt.imshow(self.bkg_image); colorbar(im)
        if save:
            plt.savefig(os.path.join(save_dir,'Background2D%s.png'%(suffix)), dpi=80)
        else:
            plt.show()


# (Old) functional way
def Run_Dynamic_Nested_Fitting(loglikelihood, prior_transform, ndim,
                               nlive_init=100, sample='auto', 
                               nlive_batch=50, maxbatch=2,
                               pfrac=0.8, n_cpu=None, print_progress=True):
    
    print("Run Nested Fitting for the image... #a of params: %d"%ndim)
    
    start = time.time()
    
    if n_cpu is None:
        n_cpu = min(mp.cpu_count()-1, 10)
        
    with mp.Pool(processes=n_cpu) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu))
        pool.size = n_cpu

        dlogz = 1e-3 * (nlive_init - 1) + 0.01

        pdsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim,
                                                 sample=sample, pool=pool,
                                                 use_pool={'update_bound': False})
        pdsampler.run_nested(nlive_init=nlive_init, 
                             nlive_batch=nlive_batch, 
                             maxbatch=maxbatch,
                             print_progress=print_progress, 
                             dlogz_init=dlogz, 
                             wt_kwargs={'pfrac': pfrac})
        
    end = time.time()
    print("Finish Fitting! Total time elapsed: %.3gs"%(end-start))
    
    return pdsampler


def get_params_fit(results, return_sample=False):
    samples = results.samples                                 # samples
    weights = np.exp(results.logwt - results.logz[-1])        # normalized weights 
    pmean, pcov = dyfunc.mean_and_cov(samples, weights)       # weighted mean and covariance
    samples_eq = dyfunc.resample_equal(samples, weights)      # resample weighted samples
    pmed = np.median(samples_eq,axis=0)
    
    if return_sample:
        return pmed, pmean, pcov, samples_eq
    else:
        return pmed, pmean, pcov
        

    
def merge_run(res_list):
    return dyfunc.merge_runs(res_list)

