# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:27:41 2018

@author: tunde
"""
import numpy as np

def a_r(P,R,M,format='days'):
    """
    function to convert period to scaled semi-major axis.
    
    Parameters:
    ----------
    P: Period of the planet.
    
    R: Radius of the star in units of solar radii.
    
    M: Mass of star in units of solar masses
    
    format: Unit of P (str). Specify "days" or "years"
    
    
    
    Returns
    -------
    a_r: Scaled semi-major axis.
    
    """
    AU_factor=1.496e8/(R*695510)
    if format=='days':
        P=P/365.
        
    return P**(2/3.)*M**(1/3.)*AU_factor
    

def AU_to_a_r(AU,R):
    """
    function to convert semi-major axis in AU to scaled semi-major axis a/R*.
    
    Parameters:
    ----------
    AU: Semi major axis of the planet in AU.
    
    R: Radius of the star in units of solar radii.
    
    
    Returns
    -------
    a_r: Scaled semi-major axis.
    
    """
    return AU*1.496e8/(R*695510)

    

def impact_parameter(inc,a,format='deg'):
    """
    Function to convert inclination to impact parameter b.
    input format of angles as 'deg' or 'rad'
    
    
    Parameters:
    ----------
    inc: inclination of the planetary orbit
    
    a: scaled semi-major axis in units of solar radii    
    
    format: (str) unit of inclincation angle  - "deg" or "rad"
    
    Returns
    -------
    
    b: impact parameter
    """
    import numpy as np

#    ecc_factor=(1-e**2)/(1+e*np.sin(np.deg2rad(w)))
    if format == 'deg':
        inc = np.radians(inc)

    return a*np.cos(inc)

def inclination(b,a):
    """
    Function to convert impact parameter b to inclination in degrees.
    
        
    Parameters:
    ----------
    b: Impact parameter of the transit.
    
    a: Scaled semi-major axis i.e. a/R*
    
    Returns
    --------
    
    inc: The inclination of the planet orbit in degrees.
    
    """
    return round(np.rad2deg(np.arccos(float(b)/a)),2)
    
def vsini(prot,st_rad):
    """
    Function to convert stellar rotation period to vsini in km/s.
    
    
    Parameters:
    ----------
    prot: Rotation period of star in days.
    
    st_rad: Stellar radius in units of solar radii
    
    Returns
    --------
    
    vsini: projected velocity of the star in km/s.
    
    """
    prot=np.array(prot)
    vsini=(2*np.pi*st_rad*696000.)/(prot*24*60*60)
    
    return vsini

def prot(vsini,st_rad):
    """
    Function to convert stellar rotation velocity vsini in km/s to rotation period in days.
    
    
    Parameters:
    ----------
    vsini: Rotation velocity of star in km/s.
    
    st_rad: Stellar radius in units of solar radii
    
    Returns
    ------
    Prot: Period of rotation of the star in days.
    
    """
    vsini=np.array(vsini)

    prot=(2*np.pi*st_rad*696000.)/(vsini*24*60*60)
    
    return prot
    
def kipping_ld(u1,u2):
    """
    Re-parameterize quadratic limb darkening parameters $u_{1}$ and $u_{2}$ according to Kipping (2013)
    
    
    Parameters:
    ----------
    u1: linear limb darkening coefficient.
    
    u2: quadratic limb darkening coefficient.
    
    Returns
    --------
    q1, q2 : Tuple containing the reparametrized limb darkening coefficients
    
    """
    
    q1 = (u1+u2)**2
    q2= u1/(2*(u1+u2))
    
    return round(q1,4), round(q2,4)
    
def ldtk_ldc(lambda_min,lambda_max,Teff,Teff_unc, logg,logg_unc,z,z_unc):
    """
    Function to estimate quadratic limb darkening coefficients for a given star
    
    Parameters:
    ----------
    lambda_min: Start wavelength of the bandpass filter.
    
    lambda_max: End  wavelength of the bandpass filter.

    Teff: Effective temperature of the star.

    Teff_unc: Uncertainty in Teff.

    logg: Surface gravity of the star.

    logg_unc: Uncertainty in logg.

    z: Metallicity of the star.

    z_unc: Uncertainty in z.
    

    Returns
    -------
    cq, eq : Each an array giving the 2 quadractic limb darkening parameters and the errors associated with them 

    
    """
    
    from ldtk import LDPSetCreator, BoxcarFilter
    
    # Define your passbands. Boxcar filters useful in transmission spectroscopy
    filters = [BoxcarFilter('a', lambda_min, lambda_max)] 

    sc = LDPSetCreator(teff=(Teff,   Teff_unc),    # Define your star, and the code
                   logg=(logg, logg_unc),    # downloads the uncached stellar
                      z=(z, z_unc),    # spectra from the Husser et al.
                     filters=filters)    # FTP server automatically.

    ps = sc.create_profiles()                # Create the limb darkening profiles
    cq,eq = ps.coeffs_qd(do_mc=True)         # Estimate quadratic law coefficients

    #lnlike = ps.lnlike_qd([[0.45,0.15],      # Calculate the quadratic law log
#                       [0.35,0.10],      # likelihood for a set of coefficients
#                       [0.25,0.05]])     # (returns the joint likelihood)

    #lnlike = ps.lnlike_qd([0.25,0.05],flt=0) # Quad. law log L for the first filter
    return cq, eq
    

def sigma_CCF(res):
    """
    Function to obtain the CCF width of non-rotating star in km/s based on resolution of spectrograph
    
    
    Parameters:
    ----------
    res: Resolution of spectrograph
    
    Returns
    -------
    sigma: CCF Width of non-rotating star in km/s 
    """
    
    return 3e5/(res*2*np.sqrt(2*np.log(2)))
    
def transit_duration(P,Rp,b,a):
    
    """
    Function to calculate the transit duration
    
    Parameters:
    ----------
    
    P: Period of planet orbit in days

    Rp: Radius of the planet in units of stellar radius

    b: Impact parameter of the planet transit [0, 1+Rp]

    a: scaled semi-major axis of the planet in units of solar radius

    Returns
    -------
    Tdur: duration of transit in hours    
    
    """
    
    tdur= (P*24/np.pi) * (np.arcsin(np.sqrt((1+Rp)**2-b**2)/a))
    
    return  tdur
    
def ingress_duration(P,R,M,Rp,format="days"):
    """
    Function to calculate the duration of ingress/egress.
    
    Parameters:
    ----------
    P: Period of the planet.
    
    R: Radius of the star in units of solar radii.
    
    M: Mass of star in units of solar masses
    
    Rp: Radius of the planet in unit of the stellar radius.
    
    format: Unit of P (str). Specify "days" or "years"
    
    
    
    Returns
    -------
    ingress_dur: Duration of ingres/egress in minutes.       
    
    
    """
    
    if format=='days':
        P=P/365.

    vel= 2*np.pi* a_r(P,R,M,format='years')/float(P)
    
    ingress_dur= 2* Rp/vel  *365*24*60
    
    return ingress_dur
    
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
    Teff: array_like; size [N]
        An array of stellar effective temperatures in Kelvin.

    logg: array_like; size [N]
        An array of log g's in dex.

    L: array_like; size [N]
        An array of luminosities in units of solar luminosity.

    Returns
    -------
    result: numpy array size [N, 2]
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
    
    spec_type: Spectral type of the star. Options: K7V, K5V, K2V, G8V, F9V, F5V, F2V
    
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
    K7V = [31150.,15605.,8015.,5185.,3785.,2975.]
    K5V = [34940.,17080.,8440.,5380.,3885.,3020.]
    K2V = [33445.,16140.,7815.,4930.,3545.,2740.]
    G8V = [30415.,14700.,7020.,4385.,3140.,2410.]
    F9V = [24450.,12685.,6105.,3785.,2690.,2045.]
    F5V = [19245.,10670.,5240.,3230.,2270.,1715.]
    F2V = [14430.,9000.,4750.,2925.,2045.,1530.]
    
    int_fxn = inter.InterpolatedUnivariateSpline(VSINI,spec_type,k=1)   #
    factor = spec_type[0]/int_fxn(vsini)
    return factor
    
    
def a_r_to_rho_star(P,a_r):
    """
    Compute the transit derived stellar density from the planet period and 
    scaled semi major axis
    
    
    Parameters:
    -----------
    
    P: array-like;
        The planet period in days
    
    a_r: array-like;
        The scaled semi-major axis of the planet orbit
        
    Returns:
    --------
    
    rho: array-like;
        The stellar density in cgs units
        
    """
    import astropy.constants as c
    import astropy.units as u
    
    st_rho=(3*np.pi/(c.G*(P*u.day)**2)*a_r**3).cgs
    return st_rho
    
def rho_to_a_r(rho,P):
    """
    convert stellar density to semi-major axis of planet with a particular period
    
    Parameters:
    -----------
    
    rho: array-like;
        The density of the star in g/cm^3.
        
    P: array-like;
        The period of the planet in days.
        
    Returns:
    --------
    
    a_r: array-like;
        The scaled semi-major axis of the planet.
    """
    
    import astropy.constants as c
    import astropy.units as u
    
    a_r=(((rho*u.g/u.cm**3*(c.G*(P*u.day)**2))/(3*np.pi)).cgs)**(1/3.)
    return a_r.value
    

def T_eq(T_st,a_r):
    """
    calculate equilibrium temperature of planet in Kelvin
    
    Parameters
    ----------
    
    T_st: Array-like;
        Effective Temperature of the star
        
    a_r: Array-like;
        Scaled semi-major axis of the planet orbit
        
    Returns
    -------
    
    T_eq: Array-like;
        Equilibrium temperature of the planet
    """
    return T_st*np.sqrt(0.5/a_r)

def R_roche(rho_pl, rho_sat):
    """
    Compute roche radius of a planet as a function of the planet's radius
    
    Parameters:
    ----------
    rho_pl: Array-like;
        density of the planet
        
    rho_sat: Array-like;
        density of the satellite
          
    Returns
    -------
    
    R_roche: Array-like;
        Roche radius of the planet
    """    
    
    return  2.44*((1.0*rho_pl)/rho_sat)**(1/3.)
    
    
def R_hill(mp, m_st, a_r,rp):
    """
    compute the hill radius of a planet
    
    Parameters:
    ----------
    
    mp: array-like;
        mass of the planet in same unit as m_st
        
    m_st: array-like;
        mass of the star
        
    a_r: array-like;
        scaled semi-major axis
    
    rp: array_like;
        ratio of planetary radius to stellar radius
        
    Returns
    --------
    
    R-hill: array-like;
        radius of hill sphere in unit of planetary radius
    
    """
    return ( mp/(3*m_st) )**(1/3.) * a_r/rp