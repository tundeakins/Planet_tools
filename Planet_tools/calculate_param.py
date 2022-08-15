import numpy as np
import astropy.constants as c
import astropy.units as u
from uncertainties.umath import  asin, sqrt, log, radians, sin, cos
from .convert_param import P_to_aR, inclination
from uncertainties import ufloat
			
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
    
    
def ldtk_ldc(Teff, logg, Z, filter, ld_law="qd", dataset="vis-lowres", nsamp=5000,unc_mult=10,nz=300):
    """
    Function to estimate different limb darkening coefficients for a given star using LDTk
    
    Parameters:
    ----------
    Teff : tuple
        Effective temperature of the star.

    logg : tuple
        Surface gravity of the star.

    Z : tuple
        Metallicity of the star.

    filter : filter transmission from SVO filter service(http://svo2.cab.inta-csic.es/theory/fps/)
            should be an ldtk.filters.SVOFilter object or np.array of shape (2, N) such that wl = flt[0] \
            and transmission= flt[1]

    ld_law : str
        ld_law to calculate. must be one of ["ln","qd","tq","p2","p2mp","nl"]

    dataset : str
        one of 'vis', 'vis-lowres', 'visir', and 'visir-lowres'

    nsamp : int
        number of limb darkening profiles to generate

    unc_mult : int
        uncertainty multiplier. default is 10

    nz : int
        number of points for resampling from mu to z.  

    Returns
    -------
    cq, eq : Each an array giving the 2 quadractic limb darkening parameters and the errors associated with them 

    
    """
    
    from ldtk import LDPSetCreator, BoxcarFilter

    assert ld_law in ["ln","qd","tq","p2","p2mp","nl"],f'ld_law must be one of ["ln","qd","tq","p2","p2mp","nl"] but {ld_law} given.' 
    assert dataset in ['vis', 'vis-lowres', 'visir','visir-lowres'],f"dataset must be one of ['vis', 'vis-lowres', 'visir','visir-lowres'] but {dataset} given."
    
    # Define your passbands. Boxcar filters useful in transmission spectroscopy
    # filters = [BoxcarFilter('a', lambda_min, lambda_max)] 

    sc = LDPSetCreator(teff=Teff, logg=logg, z=Z,    # spectra from the Husser et al.
                        filters=filter, dataset=dataset)             # FTP server automatically.

    ps = sc.create_profiles(nsamp)                # Create the limb darkening profiles\
    ps.set_uncertainty_multiplier(unc_mult)
    ps.resample_linear_z(nz)

    #calculate ld profiles
    if   ld_law == "qd":   c, e = ps.coeffs_qd(do_mc=True, n_mc_samples=100000,mc_burn=1000)         # Estimate quadratic law coefficients
    elif ld_law == "p2":   c, e = ps.coeffs_p2(do_mc=True, n_mc_samples=100000,mc_burn=1000)
    elif ld_law == "tq":   c, e = ps.coeffs_tq(do_mc=True, n_mc_samples=100000,mc_burn=1000)
    elif ld_law == "p2mp": c, e = ps.coeffs_p2mp(do_mc=True, n_mc_samples=100000,mc_burn=1000)
    elif ld_law == "nl":   c, e = ps.coeffs_nl(do_mc=True, n_mc_samples=100000,mc_burn=1000)
    elif ld_law == "ln":   
        c, e = ps.coeffs_ln(do_mc=True, n_mc_samples=100000,mc_burn=1000)
        return ufloat(c,e)
    
    coeffs = []
    for i in range(len(c[0])):
        coeffs.append(ufloat(round(c[0][i],4), round(e[0][i],4)) )
    return coeffs
    

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
    Tdur: duration of transit in same unit as P   
    
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
    
    tdur= (P/np.pi) * (factor**2/sqrt(1-e**2)) * (asin ( sqrt((1+Rp)**2 - (a*factor*cosi)**2)/ denom ) )
    
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
    Tdur: duration of transit in same unit as P    
    
    """
    #eqn 43 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
        
    T14 = transit_duration(P, Rp, a, b=b, e=e, w=w, inc=inc, total=True)
    T23 = transit_duration(P, Rp, a, b=b, e=e, w=w, inc=inc, total=False)
    
    return  (T14 - T23)/2.

def T_eq(T_st,a_r, A_b =0 , f = 1/4):
    """
    calculate equilibrium/dayside temperature  of planet in Kelvin
    
    Parameters
    ----------
    
    T_st: Array-like;
        Effective Temperature of the star
        
    a_r: Array-like;
        Scaled semi-major axis of the planet orbit

    A_b: Array-like;
        Bond albedo pf the planet. default is zero

    f: Array-like;
        heat redistribution factor.default is 1/4 for uniform heat distribution and 2/3 for None. 
        
    Returns
    -------
    
    T_eq: Array-like;
        Equilibrium temperature of the planet
    """
    print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
    return T_st*sqrt(1/a_r)* ((1-A_b)*f)**0.25


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
    
def A_g(dF, Rp, aR):
    """
    Geometric albedo of a planet from occultation depth measurement

    Parameters:
    -----------
    dF : occultation depth in ppm

    Rp : planet-to-star radius ratio

    aR : scaled semi-major axis

    """
    return (aR/Rp)**2 * dF*1e-6

def albedo_temp_relation(Tst,Tpl,wl, L, aR, RpRs):
    """
    calculate the albedo of a planet as a function of dayside temperature from eclipse depth measurement 
    
    Parameters:
    -----------
    Tst: stellar surface temeprature in K
    Tpl: planet dayside equilibrium temperature in K
    L : eclipse depth in ppm
    aR: scaled semi-major axis a/Rs
    RpRs: planet-to-star radius ratio Rp/Rs
    
    Returns:
    --------
    Ag : planet albedo
    """
    from astropy import units as u
    from astropy.modeling.models import BlackBody
#     from astropy.visualization import quantity_support

    bb = BlackBody(temperature=Tst*u.K)
    wav = wl * u.AA
    flux_st = bb(wav)
    bb_p = BlackBody(temperature=Tpl*u.K)
    flux_p = bb_p(wav)
    ag = L*1e-6*(aR/RpRs)**2 - flux_p.value/flux_st.value * aR**2
    return ag

def msini(K,e,P,Mst, return_unit = "jupiter"):
	
    """
    Calculate the minimum mass of a planet using eqn 1 of torres et al 2008 ( https://iopscience.iop.org/article/10.1086/529429/pdf)
    Paramters:
    ----------
    K : float;
        radial velocity semi amplitude in m/s
    
    e : float;
        eccentricity
    
    P : float;
        Orbital Period in days
        
    Mst : float;
        Stellar mass in solar masses
        
    Returns:
    --------
    Mp_sini : float;
        The minimum mass of the planet in jupiter masses if return_unit is "jupiter" 
        or the minimum planet-to-star mass ratio if return_unit is "star"
    """
    Mj_sini  = 4.919*1e-3 *K * (1-e**2)**0.5 * P**(1/3) * Mst**(2/3)
    
    if return_unit == "jupiter": return Mj_sini
    if return_unit == "star": return Mj_sini*u.M_jup.to(u.M_sun)/Mst
    


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
    

