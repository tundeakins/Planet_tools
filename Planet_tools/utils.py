from scipy.signal import medfilt
import numpy as np

def clip_outliers(x, y, yerr = None, clip=5, width=15, verbose=True):

    """
    Remove outliers using a running median method. Points > clip*M.A.D are removed
    where M.A.D is the mean absolute deviation from the median in each window
    
    Parameters:
    ----------
    x: array_like;
        dependent variable.
        
    y: array_like; same shape as x
        Depedent variable. data on which to perform clipping
        
    yerr: array_like(x);
        errors on the dependent variable
        
    clip: float;
       cut off value above the median. Default is 5
    
    width: int;
        Number of points in window to use when computing the running median. Must be odd. Default is 15
        
    Returns:
    --------
    x_new, y_new, yerr_new: Each and array with the remaining points after clipping
    
    """
    dd = abs( medfilt(y, width) - y)
    mad = dd.mean()
    ok= dd < clip * mad

    if verbose:
        print('\nRejected {} points more than {:0.1f} x MAD from the median'.format(sum(~ok),clip))
    
    if yerr is None:
            return x[ok], y[ok]
    
    return x[ok], y[ok], yerr[ok]


def phase_fold(t, period, t0):
    """

    Phasefold data on the give period
    
    Parameters:
    ----------- 
    t: array_like;
        array of times
        
    period: float;
        period
        
    t0: float;
    	reference time
    
    Returns:
    --------
    
    phases: array_like;
        array of phases (not sorted)	
    
    """
    return ((t - t0 + 0.5*period)%period - 0.5*period )/period



