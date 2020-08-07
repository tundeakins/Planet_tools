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
        The value of the Bayesian information Criterion. Model with lowest bic is preferred
        
        
    """
    import numpy as np
    return -2. * log_like + k * np.log(n)
    
    
def aic(log_like, n, k):
    """
    The Aikake information criterion.
    A model comparison tool based of infomormation theory. It assumes that N is large i.e.,
    that the model is approaching the CLT.
    
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
    
    AIC: array-like input;
        The value of the Akaike Information Criterion. Model with lowest aic is preferred
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
    
    Parameters:
    -----------
    
    error : array-like;
            array of error or residual values (observed - calculated)

    Return:
    --------
    rmse : float;
            root mean square error value

    """
    
    return np.sqrt(np.mean(np.square(error)))
    
def sse(error):
    
    return np.sum( np.square(error) )
				
def mse(error):
    """
    Calculate the mean-square error of imputed error array

	Parameters:
    ----------
    error : array-like;
            array of error or residual values (observed - calculated)

    Return:
    --------
    mse : float;
            mean square error value
    """        
    return np.mean(np.square(error))
    
def mae(error):
    """
    Calculate the mean-absolute error of imputed error array

	Parameters:
    ----------
    error : array-like;
            array of error or residual values (observed - calculated)

    Return:
    --------
    mae : float;
            mean absolute error value
    """        
    return np.mean(np.abs(error))
    
def rse(obs, calc):
    """
    Calculate the relative-square error from the observed and calculated values

	Parameters:
    ----------
    obs : array-like;
            array of observed values
    calc : array-like;
            array of calculated values e.g from model
    Return:
    --------
    rse : float;
            relative square error value
    """        
    return np.mean(np.square((obs-calc)/obs))
    
def rae(obs, calc):
    """
    Calculate the relative-absolute error from the observed and calculated values

	Parameters:
    ----------
    obs : array-like;
            array of observed values
    calc : array-like;
            array of calculated values e.g from model
    Return:
    --------
    rae : float;
            relative absolute error value
    """        
    return np.mean(np.abs((obs-calc)/obs))
    
def r_squared(obs, calc):
    """
    Calculate the R2_score commonly referred to as coefficient of determination. It measure how close the regression line is to the  observed values. Best possible value is 1.0
    
	Parameters:
    ----------
    obs : array-like;
            array of observed values
    calc : array-like;
            array of calculated values e.g from model
    Return:
    --------
    r2 : float;
            r2_score value
    """      
  
    return 1 - (np.sum(np.square(obs-calc)) / np.sum(np.square(obs-np.mean(obs))) )