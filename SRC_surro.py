#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:07:14 2021

@author: tobraun
"""

#------------------ PACKAGES ---------------------------#

# Standard library imports
from scipy.stats import beta as BETA
import numpy as np

#------------------ FUNCTIONS ---------------------------#


def SRC_surrogates(t, ts, Nsurr, step=0.15, Nit=1000, alpha0=1, beta=1):
    """
    Generates an ensemble of time axis and time series surrogates that have 
    the same number of samples per unit time interval. This is achieved by drawing 
    from the sampling interval distribution (with replacement). For time intervals
    with exceptionally many samples, uniform weights are continuously deformed 
    into beta-distr. weights. For each sampling interval, the corresponding amlitude
    difference is drawn as well and cumulated afterwards.
     
        
    Parameters
    ----------    
        t: 1D array (float)
            (scalar) time series 
        ts : 1D array (int)
            (scalar) time series
        Nsurr: int
             number of surrogates
        step: int
            step size of shape parameter for beta-distr. deformation (too low: no convergence)
        Nit: int
            upper limit for number of iterations for a unit time interval to sample desired 
            samples/interval (if chosen too low: no convergence)
        alpha0: float
            start value for shape parameter of beta-distr. (alpha, beta = 1, 1 uniform distr.)
        beta:
            fixed value for scale parameter of beta-distr. (alpha, beta = 1, 1 uniform distr.)
    Returns
    -------
        a_tsurr : array (float)
            time axis surrogates
        a_asurr : array (float)
            time axis surrogates    

        
    Dependencies
    ------
        Makes use of beta-distr. PDF from scipy.stats

    Examples
    --------
    
    """
    ### PARAMETERS
    # split time axis into unit time intervals
    T = t.size
    a_t0, a_split = unit_split(t)
    Tw = a_split.size
    # number of samples per unit interval
    a_spi = np.hstack([a_split[i].size for i in range(Tw)])
    # time and amplitude differences
    a_interv = np.diff(t)
    a_adiff = np.diff(ts)
    # sort sampling intervals from largest to smallest
    a_argsorted = np.flip(np.argsort(a_interv))
    # regular x-grid for weight distribution
    a_x = np.linspace(0, 1, T-1)
    
    ### SURROGATE LOOPS
    a_tsurr, a_asurr = np.zeros((Nsurr, T)), np.zeros((Nsurr, T))
    for nsurr in (range(Nsurr)):
        # for successive construction of surrogate, initialize empty arrays
        tmp_tsurr, tmp_asurr = np.empty(0), np.empty(0)
        ## loop over unit intervals
        for i in range(Tw):
            # pick an interval and extract number of samples
            k = a_spi[i]
            # WEIGHTED MC-SAMPLING
            # loop as long as samples/unit interval does not match (while not exceeding 'Nit')
            count = 0; s = 2 #>1
            # start with uniform distribution: alpha=1
            alpha = alpha0
            while (s > 1) and (count < Nit):
                # draw indices (with replacement) from sorted sampling intervals with current beta-distr. weights
                tmp_idx = beta_sampling(x=a_x, n=k, population=a_argsorted, alpha=alpha, beta=beta)
                # sampling intervals and amplitude differences
                tmp_tsmpl, tmp_asmpl = a_interv[tmp_idx], a_adiff[tmp_idx]
                # UPDATE: do the sampling intervals sum to less than the unit interval length?
                s = np.sum(tmp_tsmpl)
                count += 1
                alpha += step
            # WARNING: given 'step', 'Nit' might not be sufficient. This generates a warning.
            if count == Nit:
                print('WARNING: increase number of iterations or step size for alpha!')
            # cumulate: surrogate time axes should have same starting points 't0' as real time axis
            tmp_ct = np.cumsum(tmp_tsmpl)
            t0 = a_t0[i]
            # stacking
            tmp_tsurr = np.hstack([tmp_tsurr, t0 + tmp_ct])
            tmp_asurr = np.hstack([tmp_asurr, np.cumsum(tmp_asmpl)])
        # surrogate realizations
        a_tsurr[nsurr,] = tmp_tsurr
        a_asurr[nsurr,] = tmp_asurr
    ### OUTPUT
    return a_tsurr, a_asurr



def beta_sampling(x, n, population, alpha, beta):
    """
    Generates beta-distributed weights for efficient random sampling of 
    time axis sampling intervals.
        
    Parameters
    ----------    
        x: 1D array (float)
            regular x-grid for weight distribution
        n: int
            numbers of sampling intervals to be sampled
        population: 1D array (int)
            indeces of sampling interval population, sorted in descending order
        alpha: float
            shape parameter
        beta: float
            scale parameter
            
    Returns
    -------
        a_idx : array (int)
            indices of randomly sampled sampling intervals/amplitude differences
    """
    # generate beta-distributed weights with beta=1 and current alpha value
    tmp_weights = BETA.pdf(x, alpha, beta)
    tmp_probs = tmp_weights/tmp_weights.sum()
    # draw indices (with replacement) from sorted sampling intervals with current beta-distr. weights
    a_idx = np.random.choice(population, n, replace=True, p=tmp_probs)
    return a_idx





def unit_split(t):
    """
    Splits a time axis into unequally (or equally) sized unit time intervals.
        
    Parameters
    ----------    
        t: 1D array (float)
            (scalar) time axis 

    Returns
    -------
        a_int : array (int)
            unique integer unit sampling intervals
        a_chunks : array (float)
            time axis, split into unit intervals
    """
    # find all unique integer unit sampling intervals
    a_int = np.unique(t.astype(int))
    # assign non-integer time instances to their correspondong unit interval
    a_assign = np.digitize(t, a_int)
    a_range = np.arange(np.nanmin(a_assign), np.nanmax(a_assign)+1, 1)
    # split according to unequally sized unit intervals
    a_chunks = np.array([t[np.where(a_assign==b)] for b in a_range], dtype=object)
    return a_int, a_chunks




def AR1_irreg(t, tau, sdev=1, mean=0, x0 = 'random'):
    """
    Generate a realization of an autoregressive process of first order on an irregular
    time axis. We follow the Fortran 90 implementation by M. Mudelsee (REDFIT, 2002).
        
    Parameters
    ----------    
        t: 1D array (float)
            (scalar) time axis 
        tau: float 
            autocorrelation time
        sdev: float
            standard deviation of white noise
        mean:  float
            mean value of white noise
        x0: float
            initial value
        
    Returns
    -------
        a_x : array (float)
            AR(1)-time series
    """
    T = t.size
    a_x = np.empty(T)
    
    # intial cond.
    if x0 == 'random':
        a_x[0] = np.random.rand(1)
    else:
        a_x[0] = x0
    
    # gaussian white noise
    a_rnorm = np.random.normal(loc=mean, scale=sdev, size=T)

    # generate AR(1)-iterations
    for i in range(1, T):    
        dt = t[i] - t[i-1]
        sigma = 1.0 - np.exp(-2.0 * dt / tau)
        a_x[i] =  np.exp(-dt/tau) * a_x[i-1] + np.sqrt(sigma) * a_rnorm[i]
    # output
    return a_x

