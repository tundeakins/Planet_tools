import numpy as np
import astropy.constants as c
import astropy.units as u
from uncertainties.umath import  asin, sqrt, log, radians, sin, cos
from .convert_param import P_to_aR, inclination
			
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
	#eqn 5 - Kane & von Braun 2008, https://core.ac.uk/reader/216127860
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
    
def transit_duration(P, Rp, a, b=0, e=0, w=90, inc=None, total=True):
    
    """
    Function to calculate the total (T14) or full (T23) transit duration
    
    Parameters:
    ----------
    
    P: Period of planet orbit in days

    Rp: Radius of the planet in units of stellar radius

    b: Impact parameter of the planet transit [0, 1+Rp]

    a: scaled semi-major axis of the planet in units of solar radius
    
    inc: inclination of the orbit. Optional. If given b is not used. if None, b is used to calculate inc
    
    total: if True calculate the the total transit duration T14, else calculate duration of full transit T23

    Returns
    -------
    Tdur: duration of transit in hours    
    
    """
    #eqn 30 and 31 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
        
    factor = (1-e**2)/(1+e*sin(radians(w)))
    if inc == None:
        inc = inclination(b,a,e,w)
    
    if total is False:
        Rp = -Rp
        
    sini = sin(radians(inc))
    cosi = cos(radians(inc))
    
    denom = a*factor*sini
    
    tdur= (P*24/np.pi) * (factor**2/sqrt(1-e**2)) * (asin ( sqrt((1+Rp)**2 - (a*factor*cosi)**2)/ denom ) )
    
    return  tdur
    
def ingress_duration_appox(P, Rp, aR):
    """
    Function to calculate the duration of ingress/egress.
    
    Parameters:
    ---------
    P: Orbital period of the planet in days    
    
    Rp: Radius of the planet in unit of the stellar radius.

    aR: Scaled semi-major axis of the orbit
    
    
    Returns
    -------
    ingress_dur: very rough approximate of the Duration of ingres/egress in minutes.       
    
    
    """
        
    vel= 2*np.pi* aR/P
    
    ingress_dur= 2* Rp/vel  *24*60
    
    return ingress_dur
    
def ingress_duration(P, Rp, a, b=0, e=0, w=90, inc=None, total=True):
    
    """
    Function to calculate the ingress/egress duration T12/23 assuming symmetric light curve
    
    Parameters:
    ----------
    
    P: Period of planet orbit in days

    Rp: Radius of the planet in units of stellar radius

    b: Impact parameter of the planet transit [0, 1+Rp]

    a: scaled semi-major axis of the planet in units of solar radius
    
    inc: inclination of the orbit. Optional. If given b is not used. if None, b is used to calculate inc
    
    total: if True calculate the the total transit duration T14, else calculate duration of full transit T23

    Returns
    -------
    Tdur: duration of transit in hours    
    
    """
    #eqn 43 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
        
    T14 = transit_duration(P, Rp, a, b=b, e=e, w=w, inc=inc, total=True)
    T23 = transit_duration(P, Rp, a, b=b, e=e, w=w, inc=inc, total=False)
    
    return  (T14 - T23)/2.

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
    

