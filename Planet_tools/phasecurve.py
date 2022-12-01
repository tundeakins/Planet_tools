import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.io import fits
from sklearn.datasets import fetch_kddcup99
from uncertainties.umath import  asin, sqrt, log, log10,radians, sin, cos
from uncertainties import ufloat, umath
import uncertainties
UFloat = uncertainties.core.Variable
from ldtk import SVOFilter
from scipy.stats import norm
import pandas as pd
import os
from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
import matplotlib.pyplot as plt


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
        Back body radiance [W cm^-3 sr^-1]
    """
    from astropy.constants  import c,h,k_B
    bb = (2*h*c**2/(wl*u.nm)**5 / (np.exp(h*c/(wl*u.nm*k_B*Teff*u.K))-1)).to(u.W/u.cm**3)
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

def ellipsoidal_variation_coefficient(u: float,g: float):
    """
    calculate alpha, the ellisoidal varation coefficient. the value is usually close to unity.
    Use function `gravity_darkening_coefficient()` to obtain `g` for CHEOPS and TESS passband and `ldtk_ldc()` for u.
    eqn 9 of shporer+19 https://doi.org/10.3847/1538-3881/ab0f96.

    Parameters
    ----------
    u : float,UFloat
        linear limb darkening coefficient
    g : float,UFloat
        gravity darkening coefficient

    Return
    ------
    alpha : float, UFloat

    """
    return 0.15 *  ((15+u)*(1+g))/(3-u) 

def ellipsoidal_variation_amplitude( qm:(float,UFloat), aR:(float,UFloat),inc:(float,UFloat), alpha:(float,UFloat)):
    """
    calculate theoretical ampltiude of ellipsodial variation
    eqn 8 and 9 of shporer+19 https://doi.org/10.3847/1538-3881/ab0f96.

    Parameters
    ----------
    qm : float,UFloat
        planet-to-star mass ratio
    aR : float,UFloat
        scaled semi-major axis
    inc : float,UFloat
        inclination of the orbit
    alpha : float,UFloat
        ellipsoidal variation coefficient

    Returns
    -------
    Aev : float, UFloat
        list containing amplitude of ellipsoidal variation in ppm
    """

    return alpha * qm*sin(inc*np.pi/180)**2 / aR**3 * 1e6

def albedo_temp_relation(Tst,Tpl,flt, L, aR, RpRs, star_spec="bb", planet_spec="bb", plot_spec=False):
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
        star_spec, planet_spec: str
            theoretical spectra to use for star and planet. either of ["bb","BT-Settl","Husser2013"]

        Returns:
        --------
        Ag : planet albedo
    """
    from astropy import units as u

    #bb = lambda T,l: (2*h*c**2/(l*u.nm)**5 / (np.exp(h*c/(l*u.nm*k_B*T*u.K))-1)).to(u.W/u.m**3)
    assert isinstance(Tst,(float, int, UFloat)), 'Tst must be either float, int or ufloat'

    bb  = blackbody
    ip = create_bt_settl_interpolator()
    ip2 = create_husser2013_interpolator()
    ip.fill_value, ip2.fill_value = None, None
    ip.bounds_error, ip2.bounds_error = False, False 

    if isinstance(flt, np.ndarray): wl, tr = flt 
    else: wl,tr = np.array(flt.wavelength), flt.transmission
    
    if isinstance(Tst, UFloat): teff    = norm(Tst.n, Tst.s)
    
    #star
    if star_spec == "bb": 
        if isinstance(Tst,UFloat): 
            fs = np.array([bb(teff.rvs(), wl) for _ in range(1000)] )
            flux_st = np.array([ufloat(v,e) for v,e in zip(fs.mean(axis=0),fs.std(axis=0))])
        else: flux_st = bb(Tst,wl) 
    elif star_spec == "BT-Settl": 
        if isinstance(Tst,UFloat): 
            fs = np.array([ip((teff.rvs(), wl))/np.pi for _ in range(1000)] )
            flux_st = np.array([ufloat(v,e) for v,e in zip(fs.mean(axis=0),fs.std(axis=0))])
        else: flux_st = ip((Tst, wl))/np.pi
    elif star_spec == "Husser2013":
        if isinstance(Tst,UFloat): 
            fs = np.array([ip2((teff.rvs(), wl))/np.pi*1e-7  for _ in range(1000)] )
            flux_st = np.array([ufloat(v,e) for v,e in zip(fs.mean(axis=0),fs.std(axis=0))])
        else: flux_st = ip2((Tst, wl))/np.pi *1e-7 #(convert from erg/s to W)
    else: raise ValueError("star_spec must be one of ['bb','BT-Settl','Husser2013']")



    #planet
    if planet_spec == "bb": flux_p = bb(Tpl,wl)
    elif planet_spec == "BT-Settl": flux_p = ip((Tpl, wl))/np.pi
    elif planet_spec == "Husser2013": flux_p = ip2((Tpl, wl))/np.pi *1e-7 #(convert from erg/s to W)
    else: raise ValueError("planet_spec must be one of ['bb','BT-Settl','Husser2013']")
        
    if plot_spec:
        if isinstance(Tst,UFloat): plt.errorbar(wl,[f.n for f in flux_st],[f.s for f in flux_st], fmt="r.-",ecolor="gray" )
        else: plt.plot(wl, flux_st,"r")
        plt.plot(wl, flux_p,"b")
        plt.xlabel("wavelength [A]")
    
    # emission_ratios weighted by the filter transmission
    # em_ratio =  np.average(flux_p/flux_st, weights=tr)
    em_ratio = sum(flux_p/flux_st * tr)/sum(tr)

    ag = L*1e-6*(aR/RpRs)**2 -  em_ratio * aR**2
    return ag


def gravity_darkening_coefficient(Teff:tuple, logg:tuple, Z:tuple=None, band="TESS"):
    """
    get gravity darkening coefficients for TESS and CHEOPS using ATLAS stellar models
    use tables:
    TESS: claret2017 (https://www.aanda.org/articles/aa/full_html/2017/04/aa29705-16/aa29705-16.html) for TESS 
    CHEOPS: table 14 of claret2021 (https://iopscience.iop.org/article/10.3847/2515-5172/abdcb3)


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
        df = pd.read_csv(os.path.dirname(__file__)+'/data/ATLAS_GDC-CHEOPS.dat', delim_whitespace=True, skiprows=1)
        df = df.iloc[::2]
        df.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y']
        denom = 4

    elif band == "TESS":
        df = pd.read_html("http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/html?J/A+A/600/A30/table29.dat.gz")[0]
        df.columns = [ 'Z','Vel','logg', 'logTeff', 'y','Mod']
        denom=1
        
    mlogg = (df.logg >= (logg[0]-3*logg[1])) & (df.logg <= (logg[0]+3*logg[1]) )
    mteff = (df.logTeff >= (logTeff.n-3*logTeff.s)) & (df.logTeff <= (logTeff.n+3*logTeff.s))
    if Z: mZ = (df.Z >= (Z[0]-3*Z[1])) & (df.Z <= (Z[0]+3*Z[1]))

    m = mlogg & mteff & mZ if Z else mlogg & mteff
    assert sum(m) >= 2,"there are no or not enough samples in the range. modify input parameters and/or their uncertanities"

    res = df[m]
    #calculate euclidean distance of the samples from the input values (teff, logg,Z)
    diff = np.array((res["logTeff"],res["logg"],res["Z"])).T - np.array([logTeff.n, logg[0],Z[0]])
    res["dist"]=np.sqrt(np.sum(diff**2,axis=1))
    #assign weights based on closest dist and normalize so sum(weights)=1
    res["weights"]=  (1-res["dist"])/sum((1-res["dist"]))

    g_mean = np.sum(res["weights"]*res["y"]) / sum(res["weights"])
    g_std  = np.std(res["weights"]*res["y"]) / sum(res["weights"])
    return ufloat(round(g_mean,4), round(g_std,4))/denom


def T_eq(T_st,a_r, A_b =0 , f = 1/4):
    """
    calculate equilibrium(f=1/4)/dayside temperature(f~=1/4)  of planet in Kelvin
    
    Parameters
    ----------
    
    T_st: Array-like;
        Effective Temperature of the star
        
    a_r: Array-like;
        Scaled semi-major axis of the planet orbit

    A_b: Array-like;
        Bond albedo pf the planet. default is zero

    f: Array-like;
        heat redistribution factor.default is 1/4 for uniform heat distribution and 2/3 for None. Note f = 
        
    Returns
    -------
    
    T_eq: Array-like;
        Equilibrium temperature of the planet
    """
    # print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
    return T_st*sqrt(1/a_r)* ((1-A_b)*f)**0.25

def T_night(T_st,a_r, A_b =0 , eps = 0):
    """
    calculate nightside temperature  of planet in Kelvin (Cowan, N. B., & Agol, E. 2011,ApJ,729, 54)
    also (wong+2021 https://doi.org/10.3847/1538-3881/ac0c7d)
    
    Parameters
    ----------
    
    T_st: Array-like;
        Effective Temperature of the star
        
    a_r: Array-like;
        Scaled semi-major axis of the planet orbit

    A_b: Array-like;
        Bond albedo pf the planet. default is zero

    e: Array-like;
        heat redistribution efficiency.1 uniform heat distribution and 0for None
        
    Returns
    -------
    
    T_eq: Array-like;
        nightside temperature of the planet
    """
    # print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
    return T_st*sqrt(1/a_r)* ((1-A_b)*eps/4)**0.25

def T_irrad(T_st,a_r):
    """
    calculate irradiation temperature  of planet in Kelvin (wong+2021 https://doi.org/10.3847/1538-3881/ac0c7d)
    
    Parameters
    ----------
    
    T_st: Array-like;
        Effective Temperature of the star
        
    a_r: Array-like;
        Scaled semi-major axis of the planet orbit
    
    Returns
    -------
    T_irr: Array-like;
        irradiation temperature of the planet
    """
    # print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
    return T_st*sqrt(1/a_r)


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

def A_B(Td,Teff,aR,e):
    """
    bold albedo of a planet

    Parameters
    ----------
    Td : float
        dayside temperature
    Teff : float
        stellar effective temperature
    aR : flaot
        scaled semi-major axis
    e : float
        heat redistribution efficiency

    Returns
    -------
    float
        bold albedo of the planet
    """
    return 1 - (Td/(Teff*(1/aR)**0.5))**4 / (2/3 - 5/12*e)


def phase_integral(Ag,Ab):
    """
    phase integral

    Parameters
    ----------
    Ag : _type_
        _description_
    Ab : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return Ab/Ag
def convert_heat_redistribution(value, conv = "e2f"):
    """
    convert between atmospheric heat redistribution efficiency e and heat redistribution factor f

    Parameters:
    -------------
    value (float): value to convert
    conv (str, optional): _description_. Defaults to "e2f".

    Raises:
        ValueError: if conv is not either of "e2f" or "f2e"
    """
    if conv == "e2f": res = 2/3 - 5/12*value
    elif conv == "f2e": res = 8/5 -12/5*value
    else: raise ValueError("conv must be either e2f or f2e")
    return res

def phase_variation(pars, t):
    """
    t0, P, Fp, Fn, delta_deg, A_ell, A_dopp  = pars
    compute different contributions to orbital phase curve.
    The atmospheric modulations is given by Fn + (Fd-Tn)/2 * (1-cos(phi+delta_deg)).
    Ellipsoidal distortion of the star yields a photometric modulation with aleading order term at 
    the first harmonic of the cosine of the orbital phase (cos_2phi),
    while Doppler boosting produces a contribution at the fundamental of the sine (sin_phi).

    Parameters
    ---------

    t0, P, Fp, Fn, delta_deg, A_ell, A_dopp  = pars
    t = time

    Returns
    -------
    atm_signal, ellps, dopp

    """
    t0, P, Fp, Fn, delta_deg, A_ell, A_dopp  = pars
    
    phi   = 2*np.pi*(t-t0)/P        #phase func
    delta = np.deg2rad(delta_deg)   #phase-offset
    
    atm_signal =  (Fp-Fn) * (1- np.cos( phi + delta))/2 + Fn
    ellps      =  -A_ell  * np.cos(2*phi) + A_ell
    dopp       =  A_dopp * np.sin(phi) + A_dopp
    return atm_signal, ellps, dopp


def TSM(rho_p, Rs, Teq, mj):
    """
    Transmission spectroscopy metric according to kempton+2018(https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K)

    Parameters
    ----------
    rho_p : float
        planet density in units of earth density 
    Rs : float
        stellar radius in units of solar radii
    Teq : float
        planet equilibrium temperature in K (full heat redistribution, zero albedo)
    mj : float
        the apparent magnitude of the host star in the J band
    """

    TSM = 1/(rho_p/5.51) * Teq/Rs**2 * 10**(-mj/5)
    return TSM

def eclipse_depth_predict(RpRs, aR,Ag, Tp, Teff, flt):
    """
    calculates expected eclipse depth of a planet given a geometric albedo and dayside temperature Tp.
    equation 1 of Wong+2021(https://doi.org/10.3847/1538-3881/ac0c7d)

    Parameters
    ----------
    RpRs : float
        planet-to-star radius ratio
    aR : float
        scaled semi-major axis
    Ag : float
        geometric albedo
    Tp : float
        planet dayside temperature
    Teff : float
        stellar equilibrium temperature
    flt : SVOfilter
        observation passband

    Returns
    -------
    D: float;
        eclipse depth in ppm
    """

    bb  = blackbody
    ip = create_bt_settl_interpolator()
    ip.fill_value = None
    ip.bounds_error = False

    fp = bb(Tp,np.array(flt.wavelength))
    fs = ip((Teff,flt.wavelength))/np.pi

    em_ratio = np.average(fp/fs, weights=flt.transmission) #emission ratio
    D = em_ratio*RpRs**2 + Ag*(RpRs/aR)**2
    return D*1e6



def get_atlas_spectrum(Teff, logg, z, lambda_range=None, plot=False, store_download=True):
    """
    get ATLAS theoretical spectra for the given stellar parameters 
    (https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/).


    Parameters
    ----------
    Teff : float
        effective temperature in Kelvin
    logg : float
        surface gravity
    z : float
        metallicity [M/H]
    lambda_range : tuple of len 2, optional
        min and max wavelength in Angstroms to return, by default None
    plot : bool, optional
        whether to plot obtained spectra, by default False
    store_download : bool, optional
        whether to keep the downloaded fits file, by default True.

    Returns
    -------

    df : pandas dataframe
        dataframe containing with keys 'wavelength(A)' and 'flux' in ers/s/cm2/A
    """
    

    if lambda_range: 
        assert (np.iterable(lambda_range) & (len(lambda_range)==2) & (np.diff(lambda_range)>0) ),"lambda_range must be iterable of size 2 with "
    
    z_array    = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.2])
    logg_array = np.arange(0,5.5,0.5)
    Teff_array = np.append(np.arange(3000, 13250, 250),np.arange(13000,50000,1000))
    
    if z not in z_array: 
        idx = np.argmin( abs(z_array - z))
        print(f"z={z} is outside computed ATLAS grid, taking closest value of z={z_array[idx]}")
        z = z_array[idx]
    if logg not in logg_array:
        idx = np.argmin( abs(logg_array-logg))
        print(f"logg={logg} is outside computed ATLAS grid, taking closest value of logg={logg_array[idx]}")
        logg = logg_array[idx]
    if Teff not in Teff_array:
        idx = np.argmin( abs(Teff_array-Teff))
        print(f"Teff={Teff} is outside computed ATLAS grid, taking closest value of Teff={Teff_array[idx]}")
        Teff = Teff_array[idx]


    s  = "m" if z<0 else "p"
    z  = float(abs(z))
    mh = str(z).replace(".","")
    logg  = float(abs(logg))
    lg    = str(logg).replace(".","")
    ttttt = str(int(Teff))
    
    http_folder= f"https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/ck{s}{mh}/"
    file  =  f"ck{s}{mh}_{ttttt}.fits"

    if not os.path.exists(file):
        print(f"\nDownloading {http_folder+file} as {file}...")
        os.system("curl " + http_folder+file + " -o " + file)
    else: print(f"\n {file} already exists in current directory. Loading the fits file")
    
    hdu = fits.open(file)
    data = hdu[1].data
    if lambda_range: indx = (data["WAVELENGTH"]>=lambda_range[0]) & ((data["WAVELENGTH"]<=lambda_range[1]))
    else: indx = [True]*len(data["WAVELENGTH"])
    
    if plot:
        plt.plot(data["WAVELENGTH"][indx], data[f"g{lg}"][indx],label="KURUCZ")
        plt.xlabel("Wavelength [A]")
        plt.ylabel("ergs/s/cm2/A")
        plt.xscale("log");
    if not store_download: os.system("rm " + file)
    return pd.DataFrame({"wavelength(A)":data["WAVELENGTH"][indx], "flux":data[f"g{lg}"][indx]})

    