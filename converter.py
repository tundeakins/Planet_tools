# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:27:41 2018

@author: tunde
"""
import numpy as np

def a_r(P,R,M,format='days'):
    """
    function to convert period to scaled semi-major axis.
    
    Parameters
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
    
    Parameters
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
    
    
    Parameters
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
    
        
    Parameters
    ----------
    b: Impact parameter of the transit.
    
    a: Scaled semi-major axis i.e. a/R*
    
    """
    return round(np.rad2deg(np.arccos(float(b)/a)),2)
    
def vsini(prot,st_rad):
    """
    Function to convert stellar rotation period to vsini in km/s.
    
    
    Parameters
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
    
    
    Parameters
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
    
    
    Parameters
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
    
    Parameters
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
    
    
    Parameters
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
    
    Parameters
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
    
    Parameters
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
       
       Parameters
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
       
