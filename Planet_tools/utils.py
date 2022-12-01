from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from pycheops.core import load_config
import os

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
    
    
def plot_corner_lines(fig, ax, values, ndim=3, color="red", show_titles=False, title_fontsize=15,
                      labels=None):
    """
    Plot indicator lines in the axes, ax, of a corner plot figure, fig.
    
    Parameters:
    -----------
    
    fig : object;
        corner plot object
        
    ax : array;
        array of corner plot axes
        
    values : array-like (ndim, len(ax));
        array of values to plot. ndim can be 1-3 so as to plot also the Credible Intervals.
        0 is -1sigma, 1 is the median of maxlikelihood, 2 is +1sigma limits
    
    """
    assert len(ax) == values.T.shape[0]
    for i in range(len(ax)):
        #ML
        fig.axes[(len(ax)+1)*i].axvline(values[1][i], c=color)
        if ndim > 1:
            #CIs
            [fig.axes[(len(ax)+1)*i].axvline(values[n][i], ls="dashed", c=color) for n in [0,2]]
            
        if show_titles and ndim>1:
            lb = values[1][i] - values[0][i]
            ub = values[2][i] - values[1][i]
            fig.axes[(len(ax)+1)*i].set_title(f"{labels[i]} = {values[1][i]:.4f}$_{{-{lb:.4f}}}^{{+{ub:.4f}}}$",
                                             fontsize=title_fontsize)
            

    
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
    
def dynesty_results(res, q = 0.5):
    from dynesty import utils as dyfunc

    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])

    return [dyfunc.quantile(samples[:,i], q, weights)[0] for i in range(samples.shape[1])]
    
  
def bin_data(time, flux, err=None, nbins=20, statistic="mean"):

    """
    Calculate average flux and error in time bins of equal width.
    
    Parameters:
    -----------
    time : array;
        array of times to bin
    
    flux : array-like time;
        fluxes to perform the statistics on within each bin
        
    err : array-like time;
        err on the flux. It is binned using np.mean(x)/np.sqrt(len(x)).
        where x are the errors values in each bin.
        
    nbins: int;
        Number of bins to to split the data into.
        
    statistic: "mean", "median";
    	statistic to compute for the flux values in each bin. 
        
    Returns:
        t_bin, y_bin, err_bin
    
    """
    from scipy.stats import binned_statistic

    y_bin, y_binedges, _ = binned_statistic(time, flux, statistic=statistic, bins=nbins)
    bin_width = y_binedges[2] - y_binedges[1]
    t_bin = y_binedges[:-1] + bin_width/2.
    nans = np.isnan(y_bin)

    if err is not None:
        err_bin, _, _= binned_statistic(time, err, statistic = lambda x: 1/np.sqrt(np.sum(1/x**2)), bins=nbins)
        return t_bin[~nans], y_bin[~nans], err_bin[~nans]

    return t_bin[~nans], y_bin[~nans]
    
    
def MaxLL_result_CI(chain, weights=None, dims=None, labels=None, stat="max_central"):
    
    """
    Function to get maximum likelihood estimate of chain given results from dynesty.
    
    Parameters:
    -----------
    chain : dict, array;
        2D samples from chain or the result dict from dynesty.
        
    weights: array;
        weights of samples. If chain is a dict with including weights,
        this is used else weights should be supplied if required
        
    dims: list;
    	list of indexes to specify parameters to calculate
        
    stats: str;
        statistic to use in computing the 68.27% CI around the maximum likelihood. default is 'max_central'
        options are ['max', 'mean', 'cumulative', 'max_symmetric', 'max_shortest', 'max_central'].
        FOr definitions, See figure 6 in Andrae(2010) - https://arxiv.org/pdf/1009.2755.pdf.
        
    Returns:
    -------
    MLL: array (n_pars, 3):
        array containing [LB, mll, UB] for each parameter in samples
    """
    from chainconsumer import ChainConsumer
    from dynesty import utils as dyfunc
    
    samples=chain
    if isinstance(chain, dict):
        samples = chain.samples
        weights = np.exp(chain.logwt - chain.logz[-1])
        
    if dims is not None:
        samples = samples[:,dims]
            
    c=ChainConsumer()
    c.add_chain(samples, weights=weights, parameters=labels)
    
    c.configure(statistics=stat)
    
    summary = c.analysis.get_summary()
    mll = [summary[key] for key in summary.keys()]
        
    return np.array(mll)

def decontaminate(F,contam_frac):
    """
    decontaminate flux F following prescription by kipping & Tinetti https://doi.org/10.1111/j.1365-2966.2010.17094.x
    Fcorr = F*(1+Fcont/F_st)- Fcont/F_st
    
    Parameters:
    -----------
    
    F: array-like;
    	contaminated flux that needs correction
    
    contam_frac: float;
    	fraction of contamination in aperture Fcontam/(Ftarget+Fcontam)
    
    Returns:
    --------
    F_corr: array-like;
    	decontaminated flux
    """
    target_frac =  1-contam_frac
    return F*(1+contam_frac/target_frac) - contam_frac/target_frac

def CONAN_input_from_pycheops(d, prefix=None,outfile="CONAN_data", verbose=False):
    """
    create inputfile for CONAN from pycheops data

    Parameters
    ----------
    d : pycheops dataset object
        single visit data
    prefix : str, optional
        str to put in front of file name, by default None and d.target is used
    outfile : str, optional
        folder to save the txt file, by default "CONAN_data". folder is created if it doesnt exist in working directory.
    """
    import os
    rescale = lambda x: (x - np.min(x))/np.ptp(x) # between 0 and 1 
    rescale_r = lambda x: (2*x-(x.min()+x.max()))/np.ptp(x)  #between -1 and 1 
    def resort_roll(x):
        #make roll-angle continuous
        x = x.copy()
        phi = np.sort(x)
        gap = np.diff(phi)
        if max(gap)>20:
            brk_pt = phi[np.argmax(gap)]+0.5*max(gap)
        else: brk_pt = 360
#         x = (360 + x - brk_pt)%360
        x[x > brk_pt] -= 360
        return x

    
    if not os.path.exists(outfile): os.mkdir(outfile)
    
    t = d.lc["time"]+d.bjd_ref-2457000
    roll =  resort_roll(d.lc["roll_angle"])
    
    lc_ = np.stack((t, d.lc["flux"], d.lc["flux_err"], 
                d.lc["xoff"],d.lc["yoff"], roll, d.lc["bg"], d.lc["contam"], 
                d.lc["deltaT"])).T
    fkey = d.file_key.split("TG")[-1].split("_V")[0]
    if prefix is None: prefix = d.target
    fname = outfile+"/"+prefix+"_"+fkey+".txt"
    np.savetxt(fname, lc_, fmt="%.8f", 
        header=f'time-2457000 {"flux":10s} {"flux_err":10s} {"x_off":10s} {"y_off":10s} {"roll":10s} {"bg":10s} {"contam":10s} {"deltaT":10s}')
    if verbose: print(f"saved file as {fname}")  


def log_jitter(y, yerr):
    """
    estimate jitter in data as sigma = sqrt(std(y)**2 - median(yerr)**2). i.e extra white noise to be added to yerr to explain the observed stdev in y

    Parameters
    ----------
    y : array
        observation y or the residual after model has been subtracted.
    
    yerr : array
        uncertainties on y.

    Returns
    --------
    log_sigma : float
        log of the jitter.
    """

    return np.log( np.sqrt(np.std(y)**2 -  np.median(yerr)**2 ))



def pipe_data(name, fileid, conta_lc ="U1", smear_lc="U2",overwrite=False):
    """
    Parameters:
    -----------
    name : str
        name of the fits file
        containing the PIPE data
    fileid : str
        unique file id for the target visit
    conta_lc : str
        replace contamination with data from PIPE reduction. default is "U1"
    smear_lc : str
        replace smear with data from PIPE reduction. default is "U2"
    overwrite : Bool
        overwrite previous fits file with same name
-----------
    return:
        tgz file in pycheops data folder readable to
        pycheops
    """
    config = load_config()
    p3 = config['DEFAULT']['data_cache_path']

    # Changing the version of fileid
    fileid = fileid[:-4] + '0000' 
    hdul = fits.open(name)
    hdr = hdul[1].header
    dta = Table.read(hdul[1])
    # Masking datasets
    flg = np.asarray(dta['FLAG'])                 # Flagged data
    msk = np.where(flg==0)[0]                     # Creating a mask to remove flg>0 values
    # Extracting data from file
    mjd1, bjd1 = np.asarray(dta['MJD_TIME'])[msk], np.asarray(dta['BJD_TIME'])[msk]                 # Various time formats stored
    bjd2 = Time(bjd1, format='jd')                                                                  # Converting to UTC time
    utc1 = bjd2.to_value('fits')
    # Storing Data
    fl, fle = np.asarray(dta['FLUX'])[msk], np.asarray(dta['FLUXERR'])[msk]
    roll, xc, yc, tft2 = np.asarray(dta['ROLL'])[msk], np.asarray(dta['XC'])[msk],\
        np.asarray(dta['YC'])[msk], np.asarray(dta['thermFront_2'])[msk]
    bg = np.asarray(dta['BG'])[msk]
    conta = np.asarray(dta[conta_lc])[msk] if conta_lc in dta.keys() else np.zeros_like(fl)
    smear = np.asarray(dta[smear_lc])[msk] if smear_lc in dta.keys() else np.zeros_like(fl)
    locx, locy = np.ones_like(fl)*np.median(xc), np.ones_like(fl)*np.median(yc)
    # For relative weights of the first arbitrary PSF principal components
    # Us_n = np.array([])      # To store the names of U
    # Us = []                  # To store the values of U
    # cols = dta.colnames      # Name of all columns
    # for j in range(len(cols)):
    #     if cols[j][0] == 'U':
    #         Us_n = np.hstack((Us_n, cols[j]))
    # for j in range(len(Us_n)):
    #     usn = np.asarray(dta[Us_n[j]])[msk]
    #     Us.append(usn)

    # Making an arrays of zero for other elements
    status, event, dark, conta_err, smerr =\
        np.zeros_like(fl), np.zeros_like(fl), np.zeros_like(fl), np.zeros_like(fl), np.zeros_like(fl)
    # Creating an astropy table for storing the data
    tab = Table()
    tab['UTC_TIME'], tab['MJD_TIME'], tab['BJD_TIME'], tab['FLUX'], tab['FLUXERR'], tab['STATUS'],\
        tab['EVENT'], tab['DARK'], tab['BACKGROUND'], tab['CONTA_LC'], tab['CONTA_LC_ERR'], tab['SMEARING_LC'],\
        tab['SMEARING_LC_ERR'], tab['ROLL_ANGLE'], tab['LOCATION_X'], tab['LOCATION_Y'], tab['CENTROID_X'], tab['CENTROID_Y'] =\
        utc1, mjd1, bjd1, fl, fle, status, event, dark, bg, conta, conta_err, smear, smerr, roll, locx, locy, xc, yc
    # for j in range(len(Us_n)):
    #     tab[Us_n[j]] = Us[j]
    # Saving the table first
    tab.write(fileid + '_PIPE.fits', format='fits')
    # Making changes to the saved file
    tb_fits = fits.open(fileid + '_PIPE.fits')
    tb_fits_hdr = tb_fits[1].header
    hdr1 = hdr[8:]
    hdr2 = hdr1[:-47]
    crds = hdr2.cards
    for j in range(len(crds)):
        tb_fits_hdr.append(crds[j])
    tb_fits_hdr.append(('AP_RADI', 25.0, '(px) Radius of Aperture/NA -- PSF Photometry'))                 # Aperture radius
    nn1 = tb_fits_hdr['TARGNAME']
    os.system('rm ' + fileid + '_PIPE.fits')
    tb_fits.writeto('PIPE_' + fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits',overwrite=overwrite)
    os.system('tar -cvzf PIPE_' + fileid + '.tgz PIPE_' + fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits')
    os.system('mv PIPE_' + fileid + '.tgz ' + p3 + '/PIPE_' + fileid + '.tgz')
    #os.system('mv PIPE_' + fileid + '.tgz ' + p3 + '/' + fileid + '.tgz')
    os.system('rm PIPE_' + fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits')
    #tb_fits.writeto(fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits')
    #os.system('tar -cvzf ' + fileid + '.tgz ' + fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits')
    #os.system('mv ' + fileid + '.tgz ' + p3 + '/' + fileid + '.tgz')
    #os.system('rm ' + fileid[:-6] + '_SCI_COR_Lightcurve-DEFAULT_V0000.fits')
    print('-----------------------------------------------------------------------------')
    print('Name of the file\t\t\t\t.tgz file')
    print('-----------------------------------------------------------------------------')
    print(nn1 + '\t\t\t\tPIPE_' + fileid + '.tgz')
    #print(nn1 + '\t\t\t\t' + fileid + '.tgz')
    # For meta files
    tab1 = Table()
    tab1['thermFront_2'] = tft2
    tab1.write('PIPE_' + fileid + '-meta.fits')
    os.system('mv PIPE_' + fileid + '-meta.fits ' + p3 + '/PIPE_' + fileid + '-meta.fits')
    #os.system('mv PIPE_' + fileid + '-meta.fits ' + p3 + '/' + fileid + '-meta.fits')
    print('meta\t\t\t\t\tPIPE_' + fileid + '-meta.fits')
    #print('meta\t\t\t\t\t' + fileid + '-meta.fits')

