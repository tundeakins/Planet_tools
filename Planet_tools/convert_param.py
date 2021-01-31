from uncertainties.umath import acos, radians, degrees, sin, cos
import astropy.constants as c
import astropy.units as u
import numpy as np

rsun = c.R_sun.to( u.km).value

def P_to_aR(P,R,M,format='days'):
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
    aR: Scaled semi-major axis.
    
    """
    AU_factor = c.au.to(u.km).value/(R*rsun)
    if format == 'days':
        P=P/365.
        
    return P**(2/3.)*M**(1/3.)*AU_factor
    

def AU_to_aR(AU,R):
    """
    function to convert semi-major axis in AU to scaled semi-major axis a/R*.
    
    Parameters:
    ----------
    AU: Semi major axis of the planet in AU.
    
    R: Radius of the star in units of solar radii.
    
    
    Returns
    -------
    aR: Scaled semi-major axis.
    
    """
    return AU*c.au.to(u.km).value / (R*rsun)

    

def impact_parameter(inc, a, e=0, w=90, format='deg'):
    """
    Function to convert inclination to impact parameter b.
    input format of angles as 'deg' or 'rad'.
    see eq. 1.19 in https://www.astro.ex.ac.uk/people/alapini/Publications/PhD_chap1.pdf 
    
    Parameters:
    ----------
    inc: float;
        inclination of the planetary orbit
    
    a: float;
        scaled semi-major axis in units of solar radii 
    
    e: float;
        eccentricity of the orbit.
    
    w: float;
        longitude of periastron
      
    
    format: str;  - "deg" or "rad"
        unit of the `inc` and `w`  
    
    Returns
    -------
    
    b: impact parameter
    """

    if format == 'deg':
        inc = radians(inc)
        w = radians(w)

    ecc_factor=(1-e**2)/(1+e*sin(w))  
    return a*cos(inc)*ecc_factor

def inclination(b, a, e=0, w=90):
    """
    Function to convert impact parameter b to inclination in degrees.
    
        
    Parameters:
    ----------
    b: Impact parameter of the transit.
    
    a: Scaled semi-major axis i.e. a/R*.

    e: float;
        eccentricity of the orbit.
    
    w: float;
        longitude of periastron in degrees
    
    Returns
    --------
    
    inc: The inclination of the planet orbit in degrees.
    
    """
    ecc_factor=(1-e**2)/(1+e*sin(radians(w)))  
    inc = degrees(acos( float(b) / (a*ecc_factor)) )
    return round(inc, 2)
    
def vsini(prot, st_rad):
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
    import numpy as np
    prot=np.array(prot)
    vsini=(2*np.pi*st_rad*rsun)/(prot*24*60*60)
    
    return vsini

def prot(vsini, st_rad):
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
    import numpy as np
    vsini=np.array(vsini)

    prot=(2*np.pi*st_rad*rsun)/(vsini*24*60*60)
    
    return prot
    
def kipping_LD(u1,u2):
    """
    Re-parameterize quadratic limb darkening parameters u1 and u2 according to Kipping (2013) [1]     
    
    Parameters:
    ----------
    u1: linear limb darkening coefficient.
    
    u2: quadratic limb darkening coefficient.
    
    Returns
    --------
    q1, q2 : Tuple containing the reparametrized limb darkening coefficients
    
    Note:
    ------
    Conditions: 0< q1<1, 0<=q2<1
    
    References:
    -----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    
    q1 = (u1+u2)**2
    q2= u1/(2*(u1+u2))
    
    return q1, q2
				
def kipping_to_quadLD(q1,q2):
    """
    Re-parameterize kipping 2013 ldcs q1 and q2 to the usual quadratic limb darkening parameters u1 and u2. according to Kipping (2013) [1] 
    
    
    Parameters:
    ----------
    q1: linear limb darkening coefficient.
    
    q2: quadratic limb darkening coefficient.
    
    Returns:
    --------
    u1, u2 : Tuple containing the quadratic limb darkening coefficients
    
    Note:
    ------
    Conditions: u1+u2<1, u1>0, u1+2u2>0.
    
    References:
    -----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    u1 = 2 *q1**0.5 *q2 
    u2 = q1**0.5*(1-2*q2)
    return u1, u2
    

def kipping_to_Power2LD(q1, q2):
    """
    Re-parameterize kipping (2013)[1] ldcs q1 and q2 to the Power-2 limb darkening parameters h1 and h2 according to (Maxted 2018) [2] and Donald et al 2019 [3].
    
    Parameters:
    ----------
    q1: linear limb darkening coefficient.
    
    q2: quadratic limb darkening coefficient.
    
    Returns:
    --------
    h1, h2 : Tuple containing the the transformed limb darkening coefficients
    
    Note:
    -------
    Conditions: h1>0,  h2>0,  h2 <= h1, and h1<1.
    
    References:
    ----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    [2] https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..39M/abstract
    
    [3] https://ui.adsabs.harvard.edu/abs/2019RNAAS...3..117S/abstract
    
    """
    h1 = 1 - q1**0.5 + q2*q1**0.5 
    h2 = 1 - q1**0.5
    return h1, h2
    
def Power2_to_kippingLD(h1,h2):
    """
    Transform Power-2 limb darkening parameters h1 and h2 (Maxted 2018 [1]) to Kipping (2013 [2]) coefficients. Conditions h1>0,  h2>0,  h2 <= h1, and h1<1. 
    
    
    Parameters:
    ----------
    h1: linear limb darkening coefficient.
    
    h2: quadratic limb darkening coefficient.
    
    Returns
    --------
    q1, q2 : Tuple containing the reparametrized limb darkening coefficients
    
    Note:
    ------
    Conditions: 0< q1<1, 0<=q2<1
    
    References
    ----------
    [1] https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..39M/abstract
    
    [2] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    
    q1 = (1-h2)**2
    q2= (h1-h2)/(1-h2)
    
    return q1, q2
				
    
def aR_to_rho_star(P,aR):
    """
    Compute the transit derived stellar density from the planet period and 
    scaled semi major axis
    
    
    Parameters:
    -----------
    
    P: float, ufloat, array-like;
        The planet period in days
    
    aR: float, ufloat, array-like;
        The scaled semi-major axis of the planet orbit
        
    Returns:
    --------
    
    rho: array-like;
        The stellar density in cgs units
        
    """
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    G = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps = P*(u.day.to(u.second))
    
    st_rho=3*np.pi*aR**3 / (G*Ps**2) 
    return st_rho
    
def rho_to_aR(rho,P):
    """
    convert stellar density to semi-major axis of planet with a particular period
    
    Parameters:
    -----------
    
    rho: float, ufloat, array-like;
        The density of the star in g/cm^3.
        
    P: float, ufloat, array-like;
        The period of the planet in days.
        
    Returns:
    --------
    
    aR: array-like;
        The scaled semi-major axis of the planet.
    """
    
    G = (c.G.to(u.cm**3/(u.g*u.second**2))).value
    Ps = P*(u.day.to(u.second))
    aR = ((rho*G*Ps**2)/(3*np.pi)) **(1/3.)
    return aR
    
    
   

def timeperi_to_timetrans(tp, per, ecc, omega, secondary=False):
    """
    Convert Time of Periastron passage `tp` to Time of Transit i.e time of conjunction `tc`.
    For circular orbits with ecc=0 and w=90. tc = tp.
    Adopted from radvel: https://github.com/California-Planet-Search/radvel
    see also: http://www.sternwarte.uni-erlangen.de/~hanke/science/VelaX-1/ellipticalOrbits.pdf

    
    Parameters:
    -----------
    tp: float;
        time of periastron passage
        
    per: float;
        Planet period in same unit as tp. if tp is a phase then per = 1.
        
    ecc: float;
        eccentricity
        
    omega (float): argument of periastron (degrees)
        
    secondary: bool; 
        calculate time of secondary eclipse instead
    
    Returns:
    -------
    
    tc: float; 
        time of inferior conjunction (time of transit if system is transiting)
    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass

    if secondary:
        f = 3*np.pi/2 - omega*np.pi/180.                                       # true anomaly during secondary eclipse
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        #time of conjunction is time when true anomaly is pi/2. - omega
        f = np.pi/2 - omega*np.pi/180                       
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

    tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))         # time of conjunction

    return tc


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit (conjunction/midtransit) to Time of Periastron Passage (Taken from radvel)
    https://github.com/California-Planet-Search/radvel
    
    Parameters:
    ----------
    tc: float;
        time of midtransit (conjunction)
    
    per: float;
        period in same unit as tc. if tc is a phase, use per = 1
    
    ecc: float;
        eccentricity
    
    omega: float;
        longitude of periastron (degrees)
    
    Returns:
    --------
    float: time of periastron passage
    
    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi/2 - omega*np.pi/180
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron

    return tp

def r1r2_to_bp(r1,r2,pl=0.01, pu=0.25):
    """
    Convert uniform samling of r1 and r2 to impact parameter b and and radius ratio p
    following Espinoza 2018, https://iopscience.iop.org/article/10.3847/2515-5172/aaef38/meta
    
    Paramters:
    -----------
    r1, r2: float;
        uniform parameters in from u(0,1)
    
    pl, pu: float;
        lower and upper limits of the radius ratio
        
    
    Return:
    -------
    b, p: tuple;
        impact paramter and radius ratio
    """
    
    assert 0<r1<=1 and 0<r2<=1, f"r1 and r2 needs to be u(0,1) but r1={r1}, r2={r2}"
    
    Ar = (pu-pl)/(2+pu+pl)
    
    if r1 > Ar:
        b = (1+pl) * (1 + (r1-1)/(1-Ar) )
        p = (1-r2)*pl + r2*pu
    
    elif r1 <= Ar:
        q1 = r1/Ar
        
        b = (1+pl) + q1**0.5 * r2*(pu-pl)
        p = pu + (pl-pu)* q1**0.5*(1-r2)
    return b, p
    
    
