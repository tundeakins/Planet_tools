import numpy as np
import astropy.constants as c
import astropy.units as u
from uncertainties.umath import  asin, sqrt, log, log10,radians, sin, cos
from uncertainties import ufloat
import uncertainties
UFloat = uncertainties.core.Variable
from ldtk import SVOFilter
from scipy.stats import norm
import pandas as pd
import os


def blackbody(Teff, wl):
    """
    black body spectrum of a body as a function of wavelength

    Parameters
    ----------
    Teff : float,
        effective temperature of star in K
    wl : float, 
        wavelength in nm

    Returns
    -------
    float, array
        Back body radiance [W m^-3 sr^-1]
    """
    from astropy.constants  import c,h,k_B
    bb = (2*h*c**2/(wl*u.nm)**5 / (np.exp(h*c/(wl*u.nm*k_B*Teff*u.K))-1)).to(u.W/u.m**3)
    return bb.value

def doppler_beaming_amplitude(K,Teff,flt,stellar_lib="Husser2013"):
    """
    Calculate amplitude of doppler beaming

    Parameters
    ----------
    K : float, ufloat
        RV semi-amplitude in m/s
    Teff : float, ufloat
        effective temperature of the star
    flt : SVOfilter object
        transmission curve of instrument  used to calculate 
    stellar_lib : str, optional
        library to use for stellar spectrum. one of ["Husser2013",'BT-Settl','blackbody'], by default "Husser2013"
    
    Returns
    -------
    A_DB : float, ufloat
        Doppler beaming amplitude in ppm
    """
    from pytransit.utils.phasecurves import doppler_beaming_factor
    from astropy.constants  import c,h,k_B
    assert stellar_lib in ["Husser2013",'BT-Settl','blackbody'],f"stellar lib has to be one of ['Husser2013','BT-Settl','blackbody']"
    
    if stellar_lib == 'blackbody':
        #eqn(6) of https://iopscience.iop.org/article/10.1088/1538-3873/aa7112/meta
        wl, tr =  np.array(flt.wavelength)*1e-9, np.array(flt.transmission)
        x = lambda T: (h*c/(wl*u.m*k_B*T*u.K)).value
        alpha_BB = lambda T: np.average( x(T)*np.exp(x(T))/(np.exp(x(T))-1)/4, weights=tr)

    if isinstance(Teff, UFloat):
        if stellar_lib=="blackbody":
            alpha_array = np.array( [alpha_BB(T) for T in norm(Teff.n, Teff.s).rvs(500)])
        else:
            alpha_array = np.array([doppler_beaming_factor(T,flt, stellar_lib)/4 for T in norm(Teff.n, Teff.s).rvs(500)])
        
        alpha  = ufloat(np.mean(alpha_array), np.std(alpha_array))
    
    else: 
        alpha = alpha_BB(Teff) if stellar_lib=="blackbody" else doppler_beaming_factor(Teff,flt, stellar_lib)/4 

    return alpha * 4* K/c.value *1e6



def ellipsoidal_variation_amplitude( qm:(float,UFloat), aR:(float,UFloat),inc:(float,UFloat),u:(float,UFloat), g:(float,UFloat)):
    """
    calculate theoretical ampltiude of ellipsodial variation
    eqn 8 and 9 of shporer+19 https://doi.org/10.3847/1538-3881/ab0f96.
    Use function `gravity_darkening_coefficient()` to obtain `g` for CHEOPS and TESS passband.

    Parameters
    ----------
    qm : float,UFloat
        planet-to-star mass ratio
    aR : float,UFloat
        scaled semi-major axis
    inc : float,UFloat
        inclination of the orbit
    u : float,UFloat
        linear limb darkening coefficient
    g : float,UFloat
        gravity darkening coefficient

    Returns
    -------
    Aev: float, UFloat
        amplitude of ellipsoidal variation in ppm
    """

    alpha = 0.15 *  ((15+u)*(1+g))/(3-u)  
    return alpha * qm*sin(inc*np.pi/180)**2 / aR**3 * 1e6

def albedo_temp_relation(Tst,Tpl,flt, L, aR, RpRs):
    """
        calculate the albedo of a planet as a function of dayside temperature from eclipse depth measurement 
        eqn 3 of https://www.aanda.org/articles/aa/full_html/2019/04/aa35079-19/aa35079-19.html

        Parameters:
        -----------
        Tst: stellar surface temeprature in K
        Tpl: planet dayside equilibrium temperature in K
        flt: filter transmission from SVO filter service(http://svo2.cab.inta-csic.es/theory/fps/)
             should be an ldtk.filters.SVOFilter object or np.array of shape (2, N) such that wl = flt[0] \
             and transmission= flt[1]
        L : eclipse depth in ppm
        aR: scaled semi-major axis a/Rs
        RpRs: planet-to-star radius ratio Rp/Rs

        Returns:
        --------
        Ag : planet albedo
    """
    from astropy import units as u

    #bb = lambda T,l: (2*h*c**2/(l*u.nm)**5 / (np.exp(h*c/(l*u.nm*k_B*T*u.K))-1)).to(u.W/u.m**3)
    bb  = blackbody
    
    if isinstance(flt, np.ndarray): wl, tr = flt 
    else: wl,tr = np.array(flt.wavelength), flt.transmission
    
    #star
    flux_st = bb(Tst,wl) 

    #planet
    flux_p = bb(Tpl,wl)
    
    # emission_ratios weighted by the filter transmission
    em_ratio =  np.average(flux_p.value/flux_st.value, weights=tr)

    ag = L*1e-6*(aR/RpRs)**2 -  em_ratio * aR**2
    return ag


def gravity_darkening_coefficient(Teff:tuple, logg:tuple, Z:tuple=None, band="TESS"):
    """
    get gravity darkening coefficients for TESS and CHEOPS using ATLAS stellar models
    use tables:
    TESS: claret2017 (https://www.aanda.org/articles/aa/full_html/2017/04/aa29705-16/aa29705-16.html) for TESS 
    CHEOPS: claret2021 (https://iopscience.iop.org/article/10.3847/2515-5172/abdcb3)


    Parameters
    ----------
    Teff : tuple
        effective temperature of star given as (mean,std)
    logg : tuple
        surface gravity
    Z : tuple (optional)
        metallicity. default is None
    band : str, optional
        instrument, either 'TESS' or 'CHEOPS', by default "TESS"

    Returns
    -------
    gdc: Ufloat
        gravity darkening coefficient and uncertainty

    """
    assert band in ["TESS","CHEOPS"],f"band must be 'TESS' or 'CHEOPS'"
    logTeff = log10(ufloat(Teff[0],Teff[1]))
    
    if band == "CHEOPS":
        print(os.path.dirname(__file__)) 
        df = pd.read_csv(os.path.dirname(__file__)+'/data/ATLAS_GDC.dat', delim_whitespace=True, skiprows=1)
        df = df.iloc[::2]
        df.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y']
        denom = 4

    elif band == "TESS":
        df = pd.read_html("http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/html?J/A+A/600/A30/table29.dat.gz")[0]
        df.columns = [ 'Z','Vel','logg', 'logTeff', 'y','Mod']
        denom=1
        
    mlogg = (df.logg >= (logg[0]-logg[1])) & (df.logg <= (logg[0]+logg[1]) )
    mteff = (df.logTeff >= (logTeff.n-logTeff.s)) & (df.logTeff <= (logTeff.n+logTeff.s))
    if Z: mZ = (df.Z >= (Z[0]-Z[1])) & (df.Z <= (Z[0]+Z[1]))

    m = mlogg & mteff & mZ if Z else mlogg & mteff
    return ufloat( round(df[m].y.mean(),4), round(df[m].y.std(),4))/denom