# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:55:20 2019

@author: tunde
"""
import numpy as np
import warnings

def bic(log_like, n, k):

    """
    Compute the bayesian information criteria
    
    Parameters:
    -----------
    
    log_like: array-like;
        The maximized value of the likelihood function of the model M.
        
    n: array-like;
        Number of data points in the observation.
        
    k: array-like;
        Number of parameters estimated by the model
        
    Returns:
    --------
    
    BIC: array-like input;
        Rhe value of the Bayesian information Criterion. Model with lowest bic is preferred
        
        
    """
    import numpy as np
    return -2. * log_like + k * np.log(n)
    
    
def aic(log_like, n, k):
    """
    The Aikake information criterion.
    A model comparison tool based of infomormation theory. It assumes that N is large i.e.,
    that the model is approaching the CLT.
    """

    val = -2. * log_like + 2 * k
    val += 2 * k * (k + 1) / float(n - k - 1)

    if not np.isfinite(val):
        val = 0

        warnings.warn('AIC was NAN. Recording zero, but you should examine your fit.')


    return val
    

def rmse(error):
    """
    Calculate the root-mean-square of the imputed error array
    """
    
    return np.sqrt(np.mean(np.square(error)))
    
def sse(error):
    
    return np.sum( np.square(error) )