import numpy as np

def photo_granulation(M,R,Teff):

   """
   Estimate the amplitude and timescale of granulation noise in photometric observations
   as given by Gilliland 2011
   
   Parameters:
   ----------
   
   M: Mass of the star in Solar masses
   
   R: Radius of the star in solar radii
   
   Teff: Effective temperature of the star in Kelvin
   
   Returns
   -------
   
   amp, tau = Amplitude (in ppm) and timescale (in seconds) of the granulation noise in photometry
   
   """
   M, R, Teff= np.array(M), np.array(R), np.array(Teff)
   amp = 75 * M**(-0.5) * R * (Teff/5777.)**0.25  
   
   tau = 220 * M**(-1.) * R**2 * (Teff/5777.)**0.5
   
   return amp, tau
   
def chaplin_exptime(L,Teff,logg):
    """
    Function to compute the optimal exposure time to reduce stellar p-mode oscillation amplitude in the given star to 0.1m/s and 0.09m/s according to Chaplin et al. 2019.
    
    Parameters:
    -----------
    Teff : array_like; size [N]
        An array of stellar effective temperatures in Kelvin.

    logg : array_like; size [N]
        An array of log g's in dex.

    L : array_like; size [N]
        An array of luminosities in units of solar luminosity.

    Returns
    -------
    result : numpy array size [N, 2]
        A 2D array containing tp1 and tpE for each star. tp1 and tpE are exposure times to reach residual amplitudes of 0.1m/s and 0.09m/s to detect earth induced RV. 

    """
    from chaplinfilter import filter
    f = filter(verbose=True)
    results = f(Teff, logg, L)
    return results
    
def rv_precision_degrade(vsini,spec_type):
    """Function to calculate factor by which RV precision of a stellar spectral type degrades due to vsini.
    It compares RV precision at vsini=0km/s to that at *vsini*  in argument of function. It interpolates using quality factor degradation from Table 2 of Bouchy et al.(2001). It fits a 1-D interpolating spline through all the provided data points. 
    
    Parameters:
    ----------
    vsini: vsini of the star
    
    spec_type: Spectral type of the star. Options: "K7V", "K5V", "K2V", "G8V", "F9V", "F5V", "F2V"
    
    Returns
    --------
    Factor by which to multiply the RV precision of the star to get its precision at vsini in argument.
    
    Examples
    --------
    
    For a K2V star whose theoretical RV precision is calculated to be 0.2m/s. This function can be used to estimate the the RV precision at 5km/s.
    
    ::
    
        rv_prec = 0.2
        vsini = 5
        factor=rv_precision_degrade(vsini,K2V)
    
        rv_prec_at_5kms= rv_prec*factor
    
    
    """ 
    import scipy.interpolate as inter    
    
    #table 2 of bouchy et al. (2001)    
    VSINI = [0.,4.,8.,12.,16.,20.]
    
    stars = {'F2V': [14430.0, 9000.0, 4750.0, 2925.0, 2045.0, 1530.0],
             'F5V': [19245.0, 10670.0, 5240.0, 3230.0, 2270.0, 1715.0],
             'F9V': [24450.0, 12685.0, 6105.0, 3785.0, 2690.0, 2045.0],
             'G8V': [30415.0, 14700.0, 7020.0, 4385.0, 3140.0, 2410.0],
             'K2V': [33445.0, 16140.0, 7815.0, 4930.0, 3545.0, 2740.0],
             'K5V': [34940.0, 17080.0, 8440.0, 5380.0, 3885.0, 3020.0],
             'K7V': [31150.0, 15605.0, 8015.0, 5185.0, 3785.0, 2975.0]}    
    
    int_fxn = inter.InterpolatedUnivariateSpline(VSINI,stars[spec_type],k=1)   #
    factor = stars[spec_type][0]/int_fxn(vsini)
    return factor
    
