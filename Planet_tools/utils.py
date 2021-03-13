from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

def clip_outliers(x, y, yerr = None, clip=5, width=15, verbose=True, return_clipped_indices = False):

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
    dd = abs( medfilt(y-1, width)+1 - y)   #medfilt pads with zero, so filtering at edge is better if flux level is taken to zero(y-1)
    mad = dd.mean()
    ok= dd < clip * mad

    if verbose:
        print('\nRejected {} points more than {:0.1f} x MAD from the median'.format(sum(~ok),clip))
    
    if yerr is None:
        if return_clipped_indices:
            return x[ok], y[ok], ~ok
            
        return x[ok], y[ok]
    
    if return_clipped_indices:
        return x[ok], y[ok], yerr[ok], ~ok
    
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


def plot_emcee_chains(sampler, labels=None, thin=1, discard=0, figsize=None, alpha=0.05 ):
    """
    Plot chains from emcee sampler run.
    
    Parameters:
    -----------
    sampler: array-like; shape: (nsteps, nwalkers, ndim)
    	Sampler from emcee run
    
    labels: array/list of len ndim
    	Label for the parameters of the chain
    	
    Return:
    -------
    fig
    	
    """
    samples = sampler.get_chain(thin = thin, discard=discard)
    ndim, nwalkers = samples.shape[2], samples.shape[1]
    if figsize is None: figsize = (12,7+int(ndim/2))
    fig, axes = plt.subplots(ndim, sharex=True, figsize=figsize)
    
    if thin > 1 and discard > 0:
        axes[0].set_title(f"Discarded first {discard} steps & thinned by {thin}", fontsize=14)
    elif thin > 1 and discard == 0:
        axes[0].set_title(f"Thinned by {thin}", fontsize=14)
    else:
        axes[0].set_title(f"Discarded first {discard} steps", fontsize=14)
    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:,:,i],"k", alpha=alpha)
        ax.set_xlim(0,len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number", fontsize=14);
    
    return fig
    
def oversampling(time, oversample_factor, exp_time):

    """
    oversample time of data of long integration time and rebin the data after computation with oversampled time 
    
    Parameters:
    ----------
    
    time : ndarray;
        array of time to oversample
    
    oversampler_factor : int;
        number of points subdividing exposure
    
    exp_time: float;
        exposure time of current data in same units as input time

    Returns:
    --------
    ovs : oversampling object with attributes containing oversampled_time and function to rebin the dependent data back to original cadence.
    
    
    Example:
    --------
    
       t = np.arange(0,1000,10)
       #some function to generate data based on t
       fxn = lambda t: np.random.normal(1,100e-6, len(t))
       #divide each 10min point in t into 30 observations
       ovs = oversampling(t, 30, 10 )
       t_oversampled = ovs.oversampled_time

       #generate value of function at the oversampled time points
       f_ovs = fxn(t_oversampled)
       #then rebin f_ovs back to cadence of observation t
       f = ovs.rebin_data(f_ovs)

    """
    assert isinstance(time, np.ndarray), f'time must be a numpy array and not {type(time)}'
    t_offsets = np.linspace(-exp_time/2., exp_time/2., oversample_factor)
    t_oversample = (t_offsets + time.reshape(time.size, 1)).flatten()
    result = SimpleNamespace(oversampled_time=t_oversample)
    
    def rebin_data(data):
        rebinned_data = np.mean(data.reshape(-1,oversample_factor), axis=1)
        return rebinned_data
    
    result.rebin_data = rebin_data
    
    return result
    


