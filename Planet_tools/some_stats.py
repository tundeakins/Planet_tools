# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:55:20 2019

@author: tunde
"""
import numpy as np
import warnings
from scipy.stats import rv_continuous

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
    
def red_chisquare(data, model, error, npar):
    """
    Calculate the reduced chisquare. x2_red.
    if x2_red ~ 1, the model fits the data well.
    if x2_red << 1, the errors are overestimated, or the model is overfitting the data.
    if x2_red >> 1, the errors are underestimated, or the model is underfitting the data.
    
    Parameters:
    -----------
    
    data : array;
        the observed data.
        
    model : array-like data;
        calculated model to explain the data.
        
    error : array-like data;
        error on the observed data points.
        
    npar : int;
        number of fitted parameters in the model.
        
    Returns:
    --------
    
    x2_red : float;
        the reduced chisquare given by sum(((data-model)/error)**2) / (len(data)-npar)
        
    """
    
    return np.sum(((data-model)/error)**2) / (len(data)-npar)
    
    
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




class MixtureModel(rv_continuous):

    def __init__(self, submodels, *args, weights = None, **kwargs):
        """
        create a distribution composed of a number of scipy.stats distribitions
        #copied from https://stackoverflow.com/questions/47759577/creating-a-mixture-of-probability-distributions-for-sampling
        
        Parameters
        ----------
        submodels : scipy frozen
            scipy frozen distribution e.g. norm, uniform,...
        weights : list of floats
            weight of each distribution, by default None

        Examples
        -------
        >>> mixture_gaussian_model = MixtureModel([uniform(-3, 3), norm(3, 1)]) #weights = [0.3, 0.5, 0.2]
        >>> x_axis = np.arange(-6, 6, 0.001)
        >>> mixture_pdf = mixture_gaussian_model.pdf(x_axis)
        >>> mixture_rvs = mixture_gaussian_model.rvs(100000)

        """

        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

    

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs
