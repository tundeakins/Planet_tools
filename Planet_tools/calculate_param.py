import numpy as np
import astropy.constants as c
import astropy.units as u
from uncertainties.umath import  asin, sqrt, log, radians
from .convert_param import P_to_aR
			
def planet_prot(f,req,mp,j2):
    """
	Function to calculate period of rotation of a planet
	
	Parameters
	-----------
	
	f : float;
		planet oblateness
	req : float;
		Equatorial radius of planet in jupiter radii.
	mp : float;
		mass of planet in jupiter masses
	j2 : float;
		quadrupole moment of the planet
		
    Returns:
    --------
    prot: rotation peropd of the planet in hours
		
    """
	

    radius=req*c.R_jup
    mass= mp*c.M_jup
    prot=2*np.pi*sqrt(radius**3 / (c.G*mass*(2*f-3*j2)))
	
    return prot.to(u.hr).value
    
    
def transit_prob(Rp, aR, e=0, w=90):
    """
	Function to calculate period of rotation of a planet
	
	Parameters
	-----------
    Rp: float;
        radius of the planet in unit f the stellar radius
    
    aR: float;
        Scaled semi-major axis i.e. a/R*.

    e: float;
        eccentricity of the orbit.
    
    w: float;
        longitude of periastron in degrees
    
		
    """
	
    prob = (1 + Rp)/aR * (1 + e*sin(radians(w))/(1-e**2)  )

	
    return prob
    
    
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
    
    return 3e5/(res*2*sqrt(2*log(2)))
    
def transit_duration(P, Rp, b, a):
    
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
    
    tdur= (P*24/np.pi) * (asin(sqrt((1+Rp)**2-b**2)/a))
    
    return  tdur
    
def ingress_duration(Rp,P,R,M,format="days"):
    """
    Function to calculate the duration of ingress/egress.
    
    Parameters:
    ----------
    Rp: Radius of the planet in unit of the stellar radius.

    P: Period of the planet.
    
    R: Radius of the star in units of solar radii.
    
    M: Mass of star in units of solar masses
    
   
    format: Unit of P (str). Specify "days" or "years"
    
    
    
    Returns
    -------
    ingress_dur: Duration of ingres/egress in minutes.       
    
    
    """
    
    if format=='days':
        P=P/365.
        
    vel= 2*np.pi* P_to_aR(P,R,M,format='years')/float(P)
    
    ingress_dur= 2* Rp/vel  *365*24*60
    
    return ingress_dur
    

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
    print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
    return T_st*sqrt(0.5/a_r)

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
    
def RL_Rroche(j2,mp_ms,a_r,rp_rj,rho_r):
    return 0.75*(j2/0.01)**(1/5.)*(mp_ms/0.001)**(-2/15.)*(rp_rj)**(2/5.)*(a_r/21.5)**(3/5.)*(rho_r/3)**(1/3.)    
    

def phase_fold(time, t0, P):
    """
    Given the observation time and period, return the phase of each observation time
    
    Parameters:
    ----------
    
    time: array-like;
        time of observation

    t0: array-like;
        reference time
        
    P: float;
        Period to use in folding in same unit as t
        
    Returns
    --------
    phases: aaray-like;
        phases of the observation    
    
    """
    
    
    return ( (time-t0) % P) / float(P)
    


#def true_anomaly(t, tp, per, e):
    """
    Calculate the true anomaly for a given time, period, eccentricity.
    from radvel: https://github.com/California-Planet-Search/radvel
    
    Parameters:
    -----------
    t: array;
        array of times in JD
    tp: float;
        time of periastron, same units as t
    
    per: float;
        orbital period in days
        
    e: float;
        eccentricity

    Returns:
    -------
    array: true anomoly at each time
    """
#	import radvel
    # f in Murray and Dermott p. 27
#    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
#    eccarr = np.zeros(t.size) + e
#    e1 = radvel.kepler.kepler(m, eccarr)
#    n1 = 1.0 + e
#    n2 = 1.0 - e
#    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))

#    return nu
    

