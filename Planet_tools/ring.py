import numpy as np
import astropy.constants as c
import astropy.units as u
from uncertainties.umath import  asin, sqrt, log, radians, sin, cos



def effective_ringed_planet_radius(rp,rin,rout,ir):
    """
    Calculate effective radius of a ringed planet accounting for possible overlap between ring and planet. - eqn (2) from zuluaga+2015 http://dx.doi.org/10.1088/2041-8205/803/1/L14
    
    Parameters:
    -----------
    rp : float, ufloat;
        Radius of the planet hosting the ring
    
    rin, rout : float, ufloat;
        Inner and outer radii of the ring in units of rp.
        
    ir : float, ufloat;
        Inclination of the ring from the skyplane. 0 is face-on ring, 90 is edge on
    
    Returns:
    --------
    eff_R : float, ufloat;
        effective radius of the ringed planet in same unit as rp
    	
    """	
    
    cosir = cos(radians(ir))
    sinir = sin(radians(ir))
    y = lambda r: sqrt(r**2 -1)/(r*sinir)

    def eff(r):
        if r*cosir > 1:
            eff_r = r**2 * cosir - 1
        else:
            eff_r = (r**2*cosir * 2/np.pi*asin(y(r))) - (2/np.pi*asin(y(r)*r*cosir))
        
        return eff_r
    rout_eff2 = eff(rout)
    rin_eff2 = eff(rin) 
    
    Arp = rp**2 + rp**2*(rout_eff2 - rin_eff2)
    return sqrt(Arp)
    
    
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
    
    return  2.46*((1.0*rho_pl)/rho_sat)**(1/3.)
    
 
def max_ring_density(rho_pl, f=1, verbose=True):
    """
    Use Roche formular to compute maximum density of ring material given inferred planet density in g/cm3.
    since rings cannot be sustained beyond roche limit, we assume the observed radius is the roche limit of possible ringed planet. 
    
    Parameters:
    ----------
    rho_pl: ufloat, float, array-like;
        Inferred density of the planet given the observed radius and estimated mass. ufloat from uncertainties package allows the uncertainties in planet density to be propagated
        to the resulting ring density 
    
    f: float;
    	factor to account for 

    Returns
    -------
    
    rho_ring: Array-like;
        maximum ring density a planet can have. 
    """    
    rho_r = rho_pl * (2.46* f**0.5)**3 
    if verbose: print(f"Any ring around this planet must have density below {rho_r} in order to be within the Roche limit. Check ```T_eq``` of this planet to ensure that ring of such density can exist.")
    return  rho_r
    
    
def ring_density(Rout, rho_pl):
    """
    Calculate the density of the rings material around a ringed planet given the outer ring radius and the density of the host planet.
    
    Parameters:
    ------------
    
    Rout: float, ufloat, array-like;
    	outer ring radius in units of the planet radius Rp.
    	
    rho_pl: float, ufloat, array-like;
    	Density of the host planet.
    	
    Returns:
    --------
    
    rho_r: 
    	ring material density in same unit as rho_pl
    
    """
    
    return (2.46/Rout)**3 * rho_pl
    
    
    
    
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
    
def RL_Rroche(j2,mp_ms,a_r,rp_rj,rho_r):
    """
    Calculate the ratio of the laplace radius RL to the Roche radius RR to determine the ring plane.
    The ring plane aligns with the equatorial plane of the planet if RL/RR < 1 since the entirety of the ring will be within the laplace radius.
    RL/RR < 1 implies that the rings extend beyond the laplace radius and thus will transition to lying in the orbital plane of the planet.
    Ref. Schlichting & Chang 2011 - https://iopscience.iop.org/article/10.1088/0004-637X/734/2/117
    
    Parameters:
    -----------
    
    j2: float;
        quadrupole moment of the planet (ranges from∼0.003 for Uranus and Neptune to∼0.01 for Jupiter and Sat-urn (Carter & Winn 2010)) 
    
    mp_ms: float;
        Ratio of the planet-stellar mass.
        
    a_r: float;
        Scaled semi-major axis(unit of stellar radius)
    
    rp_rj: float;
        planet radius in unit of jupiter radii
        
    rho_r: float;
        density of ring particles in g/cm3.
        
    Return:
    -------
    
    RL/RR : float;
        Ratio of Laplace radius to roche radius
    """
    return 0.75*(j2/0.01)**(1/5.)*(mp_ms/0.001)**(-2/15.)*(rp_rj)**(2/5.)*(a_r/21.5)**(3/5.)*(rho_r/3)**(1/3.)    
    
