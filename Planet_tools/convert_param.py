

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
    AU_factor=1.496e8/(R*695510)
    if format=='days':
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
    vsini=np.array(vsini)

    prot=(2*np.pi*st_rad*695510.)/(vsini*24*60*60)
    
    return prot
    
def kipping_LD(u1,u2):
    """
    Re-parameterize quadratic limb darkening parameters $u_{1}$ and $u_{2}$ according to Kipping (2013) [1]     
    
    Parameters:
    ----------
    u1: linear limb darkening coefficient.
    
    u2: quadratic limb darkening coefficient.
    
    Returns
    --------
    q1, q2 : Tuple containing the reparametrized limb darkening coefficients
    
    References:
    -----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    
    q1 = (u1+u2)**2
    q2= u1/(2*(u1+u2))
    
    return round(q1,4), round(q2,4)
				
def kipping_to_quadLD(q1,q2):
    """
    Re-parameterize kipping 2013 ldcs $q_{1}$ and $q_{2}$ to the usual quadratic limb darkening parameters $u_{1}$ and $u_{2}$. according to Kipping (2013) [1] 
    
    
    Parameters:
    ----------
    q1: linear limb darkening coefficient.
    
    q2: quadratic limb darkening coefficient.
    
    Returns
    --------
    u1, u2 : Tuple containing the quadratic limb darkening coefficients
    
    References:
    -----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    import numpy as np
    u1 = 2 * np.sqrt(q1)*q2 
    u2 = np.sqrt(q1)*(1-2*q2)
    return round(u1,4), round(u2,4)
    

def kipping_to_Power2LD(q1,q2):
    """
    Re-parameterize kipping (2013)[1] ldcs $q_{1}$ and $q_{2}$ to the Power-2 limb darkening parameters $h_{1}$ and $h_{2}$ according to (Maxted 2018) [2] and Donald et al 2019 [3].
    
    Parameters:
    ----------
    q1: linear limb darkening coefficient.
    
    q2: quadratic limb darkening coefficient.
    
    Returns
    --------
    h1, h2 : Tuple containing the the transformed limb darkening coefficients
    
    References
    ----------
    [1] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    [2] https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..39M/abstract
    
    [3] https://ui.adsabs.harvard.edu/abs/2019RNAAS...3..117S/abstract
    
    """
    import numpy as np
    h1 = 1 - np.sqrt(q1) + q2*np.sqrt(q1) 
    h2 = 1 - np.sqrt(q1)
    return round(h1,4), round(h2,4)
    
def Power2_to_kippingLD(h1,h2):
    """
    Transform Power-2 limb darkening parameters $h_{1}$ and $h_{2}$ (Maxted 2018 [1]) to Kipping (2013 [2]) coefficients 
    
    
    Parameters:
    ----------
    h1: linear limb darkening coefficient.
    
    h2: quadratic limb darkening coefficient.
    
    Returns
    --------
    q1, q2 : Tuple containing the reparametrized limb darkening coefficients
    
    References
    ----------
    [1] https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..39M/abstract
    
    [2] https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    
    """
    
    q1 = (1-h2)**2
    q2= (h1-h2)/(1-h2)
    
    return round(q1,4), round(q2,4)
				
    
def aR_to_rho_star(P,aR):
    """
    Compute the transit derived stellar density from the planet period and 
    scaled semi major axis
    
    
    Parameters:
    -----------
    
    P: array-like;
        The planet period in days
    
    aR: array-like;
        The scaled semi-major axis of the planet orbit
        
    Returns:
    --------
    
    rho: array-like;
        The stellar density in cgs units
        
    """
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    st_rho=(3*np.pi/(c.G*(P*u.day)**2)*aR**3).cgs
    return st_rho
    
def rho_to_aR(rho,P):
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
    
    aR: array-like;
        The scaled semi-major axis of the planet.
    """
    
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    aR=(((rho*u.g/u.cm**3*(c.G*(P*u.day)**2))/(3*np.pi)).cgs)**(1/3.)
    return aR.value
    
