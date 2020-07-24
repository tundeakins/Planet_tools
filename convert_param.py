

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

    

def impact_parameter(inc,a,format='deg',e=0,w=90):
    """
    Function to convert inclination to impact parameter b.
    input format of angles as 'deg' or 'rad'.
    see eq. 1.19 in https://www.astro.ex.ac.uk/people/alapini/Publications/PhD_chap1.pdf 
    
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

    ecc_factor=(1-e**2)/(1+e*np.sin(np.deg2rad(w)))  
    if format == 'deg':
        inc = np.radians(inc)

    return a*np.cos(inc)*ecc_factor

def inclination(b,a,e=0,w=90):
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
    ecc_factor=(1-e**2)/(1+e*np.sin(np.deg2rad(w)))  
    inc = np.rad2deg(np.arccos( float(b) / (a*ecc_factor)) )
    return round(inc, 2)
    
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
    vsini=(2*np.pi*st_rad*695510.)/(prot*24*60*60)
    
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

    prot=(2*np.pi*st_rad*695510.)/(vsini*24*60*60)
    
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
    
