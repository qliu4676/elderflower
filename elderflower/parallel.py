"""
Submodule for parallel computing
Adapted from https://github.com/pycroscopy/sidpy (S.Somnath, C.Smith)
"""

import numpy as np
import joblib

def parallel_compute(data, func, cores=None, lengthy_computation=False, func_args=None, func_kwargs=None, verbose=False):
    """
    Computes the provided function using multiple cores using the joblib library
    Parameters
    ----------
    data : numpy.ndarray
        Data to map function to. Function will be mapped to the first axis of data
    func : callable
        Function to map to data
    cores : uint, optional
        Number of logical cores to use to compute
        Default - All cores - 1 (total cores <= 4) or - 2 (cores > 4) depending on number of cores.
    lengthy_computation : bool, optional
        Whether or not each computation is expected to take substantial time.
        Sometimes the time for adding more cores can outweigh the time per core
        Default - False
    func_args : list, optional
        arguments to be passed to the function
    func_kwargs : dict, optional
        keyword arguments to be passed onto function
    verbose : bool, optional. default = False
        Whether or not to print statements that aid in debugging
    Returns
    -------
    results : list
        List of computational results
    """

    if not callable(func):
        raise TypeError('Function argument is not callable')
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')
    if func_args is None:
        func_args = list()
    else:
        if isinstance(func_args, tuple):
            func_args = list(func_args)
        if not isinstance(func_args, list):
            raise TypeError('Arguments to the mapped function should be specified as a list')
    if func_kwargs is None:
        func_kwargs = dict()
    else:
        if not isinstance(func_kwargs, dict):
            raise TypeError('Keyword arguments to the mapped function should be specified via a dictionary')

    req_cores = cores
    rank = 0
    cores = recommend_cpu_cores(data.shape[0],
                                requested_cores=cores,
                                lengthy_computation=lengthy_computation,
                                verbose=verbose)

    if verbose:
        print('Rank {} starting computing on {} cores (requested {} cores)'.format(rank, cores, req_cores))

    if cores > 1:
        values = [joblib.delayed(func)(x, *func_args, **func_kwargs) for x in data]
        results = joblib.Parallel(n_jobs=cores, backend='multiprocessing')(values)

        # Finished reading the entire data set
        if verbose:
            print('Rank {} finished parallel computation'.format(rank))

    else:
        if verbose:
            print("Rank {} computing serially ...".format(rank))
        # List comprehension vs map vs for loop?
        # https://stackoverflow.com/questions/1247486/python-list-comprehension-vs-map
        results = [func(vector, *func_args, **func_kwargs) for vector in data]

    return results


def recommend_cpu_cores(num_jobs, requested_cores=None, lengthy_computation=False, min_free_cores=None, verbose=False):
    """
    Decides the number of cores to use for parallel computing
    Parameters
    ----------
    num_jobs : unsigned int
        Number of times a parallel operation needs to be performed
    requested_cores : unsigned int (Optional. Default = None)
        Number of logical cores to use for computation
    lengthy_computation : Boolean (Optional. Default = False)
        Whether or not each computation takes a long time. If each computation is quick, it may not make sense to take
        a hit in terms of starting and using a larger number of cores, so use fewer cores instead.
        Eg- BE SHO fitting is fast (<1 sec) so set this value to False,
        Eg- Bayesian Inference is very slow (~ 10-20 sec)so set this to True
    min_free_cores : uint (Optional, default = 1 if number of logical cores < 5 and 2 otherwise)
        Number of CPU cores that should not be used)
    verbose : Boolean (Optional.  Default = False)
        Whether or not to print statements that aid in debugging
    Returns
    -------
    requested_cores : unsigned int
        Number of logical cores to use for computation
    """
    from multiprocess import cpu_count
    logical_cores = cpu_count()

    if min_free_cores is not None:
        if not isinstance(min_free_cores, int):
            raise TypeError('min_free_cores should be an unsigned integer')
        if min_free_cores < 0 or min_free_cores >= logical_cores:
            raise ValueError('min_free_cores should be an unsigned integer less than the number of logical cores')
        if verbose:
            print('Number of requested free CPU cores: {} was accepted'.format(min_free_cores))
    else:
        if logical_cores > 4:
            min_free_cores = 2
        else:
            min_free_cores = 1
        if verbose:
            print('Number of CPU free cores set to: {} given that the CPU has {} logical cores'
                  '.'.format(min_free_cores, logical_cores))

    max_cores = max(1, logical_cores - min_free_cores)

    if requested_cores is None:
        # conservative allocation
        if verbose:
            print('No requested_cores given.  Using estimate of {}.'.format(max_cores))
        requested_cores = max_cores
    else:
        if not isinstance(requested_cores, int):
            raise TypeError('requested_cores should be an unsigned integer')
        if verbose:
            print('{} cores requested.'.format(requested_cores))
        if requested_cores < 0 or requested_cores > logical_cores:
            # Respecting the explicit request
            requested_cores = max(min(int(abs(requested_cores)), logical_cores), 1)
            if verbose:
                print('Clipped explicit request for CPU cores to: {}'.format(requested_cores))

    if not isinstance(num_jobs, int):
        raise TypeError('num_jobs should be an unsigned integer')
    if num_jobs < 1:
        raise ValueError('num_jobs should be greater than 0')

    jobs_per_core = max(int(num_jobs / requested_cores), 1)
    min_jobs_per_core = 10  # I don't like to hard-code things here but I don't have a better idea for now
    if verbose:
        print('computational jobs per core = {}. For short computations, each core must have at least {} jobs to '
              'warrant parallel computation.'.format(jobs_per_core, min_jobs_per_core))

    if not lengthy_computation:
        if verbose:
            print('Computations are not lengthy.')
        if requested_cores > 1 and jobs_per_core < min_jobs_per_core:
            # cut down the number of cores if there are too few jobs
            jobs_per_core = 2 * min_jobs_per_core
            # intelligently set the cores now.
            requested_cores = max(1, min(requested_cores, int(num_jobs / jobs_per_core)))
            if verbose:
                print('Not enough jobs per core. Reducing cores to {}'.format(requested_cores))

    return int(requested_cores)
