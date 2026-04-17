import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.io import fits
from uncertainties.umath import  asin, sqrt, log, log10,radians, sin, cos
from uncertainties import ufloat, umath
import uncertainties
UFloat = uncertainties.core.Variable
from ldtk import SVOFilter
from scipy.stats import norm
import pandas as pd
import os
# from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.interpolate import RectBivariateSpline


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
	from astropy.constants  import c,h,k_B
	assert stellar_lib in ["Husser2013",'BT-Settl','blackbody'],f"stellar lib has to be one of ['Husser2013','BT-Settl','blackbody']"


	if stellar_lib == 'blackbody':
		# eqn(6) of https://iopscience.iop.org/article/10.1088/1538-3873/aa7112/meta
		wl, tr =  np.array(flt.wavelength)*1e-9, np.array(flt.transmission)
		x = lambda T: (h*c/(wl*u.m*k_B*T*u.K)).value
		alpha_BB = lambda T: np.average( x(T)*np.exp(x(T))/(np.exp(x(T))-1)/4, weights=tr)

	if isinstance(Teff, UFloat):
		if stellar_lib=="blackbody":
			alpha_array = np.array( [alpha_BB(T) for T in norm(Teff.n, Teff.s).rvs(500)])
		else:
			try:
				from pytransit.utils.phasecurves import doppler_beaming_factor
			except ImportError: 
				raise ImportError("pytransit is required to use the 'BT-Settl' and 'Husser2013' stellar spectra. Please install pytransit to use these features.")
			alpha_array = np.array([doppler_beaming_factor(T,flt, stellar_lib)/4 for T in norm(Teff.n, Teff.s).rvs(500)])

		alpha  = ufloat(np.mean(alpha_array), np.std(alpha_array))

	else: 
		alpha = alpha_BB(Teff) if stellar_lib=="blackbody" else doppler_beaming_factor(Teff,flt, stellar_lib)/4 

	return alpha * 4* K/c.value *1e6


def ellipsoidal_variation_coefficients(coeffs, y, input_LDlaw="quad", mu_min=0., n_powers=4, show_plot=False):
	"""
	calculate alpha1 and alpha2, the ellisoidal varation coefficients. The value of alpha2 is usually close to unity.
	Use function `gravity_darkening_coefficient()` to obtain `g` for CHEOPS, TESS, and JWST passbands and `ldtk_ldc()` for coeffs.
	eqn 12 and 13 of Esteves2013 https://doi.org/10.1088/0004-637X/772/1/51. This is expanded for the case of
	quadratic and claret 4-parameter limb darkening laws (using eqn 9 and 10 of Deline2025 https://arxiv.org/pdf/2505.01544).
	where alpha_ev=alpha2 and beta_ev=alpha1.

	- Kopal LD law --> 1 - κ1(1-μ) - κ2(1-μ^2) - κ3(1-μ^3) - κ4(1-μ^4).

	- power-2 law --> 1 - c*(1-μ^α).

	- claret 4-parameter law --> 1 - u1(1-μ^0.5) - u2(1-μ) - u3(1-μ^1.5) - u4(1-μ^2).	

	Parameters
	----------
	coeffs: list, array
		limb darkening coefficients. length 1 for linear limb darkening law (u1), length 2 for quad (u1, u2) and power-2 (c, alpha) limb darkening law, 
		and length 4 for claret ("non-linear": u1, u2, u3, u4) limb darkening law. Note that the power-2 and claret coefficients are transformed before 
		being used in EV equations since Kopal LD laws are different from these.
	y : float,UFloat
		gravity darkening coefficient
	input_LDlaw: str, optional
		limb darkening law of the input coefficients. either "quad", "power-2" or "claret". default is "quad".
	mu_min: float, optional
		minimum mu value to consider when fitting the Kopal limb darkening law to the input law. default is 0, 
		but can be set to a higher value to exclude the steep drop in intensity near the limb which is often poorly captured by the input LD laws.
	n_powers: int, optional
		number of powers (of μ) of the Kopal limb darkening law to match the input (power-2 or Claret) LD. 
		Default is 2, since the input laws have info only upto μ^2, so Kopal law cannot extract strucuture at μ^3 or μ^4 that is not in the input profile.
	show_plot: bool, optional
		if True, shows a plot of the limb darkening profile and the Kopal fit. default is False.

	Return
	------
	alpha : SimpleNamespace
		alpha1 - float, UFloat: ellipsoidal variation coefficient 1 used in the fractional constants f1 and f2
		alpha2 - float, UFloat: ellipsoidal variation coefficient 2 used in the main amplitude Aev
	"""
	
	assert input_LDlaw in ["linear", "quad","power-2","claret"], f"input_LDlaw must be one of ['linear', 'quad', 'power-2', 'claret']"
	assert n_powers <=4, f"n_powers must be less than or equal to 4 since Kopal limb darkening law has 4 powers of μ at maximum"
	assert (0.0 <= mu_min < 1.0), f"mu_min must be between 0 and 1, got {mu_min}."

	# --- if law is quadratic, Monte Carlo sampling when any coefficient or y is a UFloat ---. quadratic is directly propagated with ufloat
	if input_LDlaw != "quad":
		coeffs_has_ufloat = (
			isinstance(coeffs, UFloat) or
			(hasattr(coeffs, "__len__") and any(isinstance(c, UFloat) for c in coeffs))
		)
		if coeffs_has_ufloat or isinstance(y, UFloat):
			n_samp = 1000
			if isinstance(coeffs, UFloat):
				coeffs_samp = [norm(coeffs.n, coeffs.s).rvs(n_samp)]
			else:
				coeffs_samp = [
					norm(c.n, c.s).rvs(n_samp) if isinstance(c, UFloat) else np.full(n_samp, float(c))
					for c in coeffs
				]
			y_samp = norm(y.n, y.s).rvs(n_samp) if isinstance(y, UFloat) else np.full(n_samp, float(y))

			if show_plot:
				res_fig =  ellipsoidal_variation_coefficients([c.n if isinstance(c, UFloat) else c for c in coeffs], 
																y.n if isinstance(y, UFloat) else y, 
																input_LDlaw=input_LDlaw, mu_min=mu_min, n_powers=n_powers,show_plot=show_plot)

			alpha1_arr = np.empty(n_samp)
			alpha2_arr = np.empty(n_samp)
			for i in range(n_samp):
				c_i = float(coeffs_samp[0][i]) if isinstance(coeffs, UFloat) else [float(s[i]) for s in coeffs_samp]
				res = ellipsoidal_variation_coefficients(c_i, float(y_samp[i]), input_LDlaw=input_LDlaw, mu_min=mu_min, n_powers=n_powers, show_plot=False)
				alpha1_arr[i] = float(res.alpha1)
				alpha2_arr[i] = float(res.alpha2)
			return SimpleNamespace(
				alpha1=ufloat(np.mean(alpha1_arr), np.std(alpha1_arr)),
				alpha2=ufloat(np.mean(alpha2_arr), np.std(alpha2_arr)),
				fig=res_fig.fig
			)
			
	# ---------------------------------------------------------------
	if input_LDlaw in ["linear","quad"]:
		if isinstance(coeffs, (float, UFloat)):
			u  = [coeffs, 0.0]  # set u2=0 for linear limb darkening law
		elif hasattr(coeffs, "__len__"):
			assert len(coeffs) in [1,2], "coeffs must be of length 1(for linear), 2(for quadratic)"
			if len(coeffs)==1:
				u = [coeffs[0], 0.0]  # set u2=0 for linear limb darkening law
			else:
				u = coeffs
		ld_fig = None
		#eqn 2-39 of Kopal 1959 with modification for different quadratic parameterization gotten from Deline 2025 (eqns E.8 and E.9)
		u1, u2 = u[0], u[1]
		Xi = np.array( [ 
							2/5 * (15 + u1 + 2*u2),
							1/7 * (35*u1 + 22*u2),
							9/8 * (4*(u1-1)+u2), 
						]) / (6 - 2*u1 - u2)

	elif input_LDlaw in ["power-2", "claret"]:   #eqn 2-41 of Kopal 1959 with fit to input law to obtain the best coeffs for the eqn           

		def _inner(p, q, mu_min=mu_min):
			"""Analytic integral: int_0^1 (1 - mu^p)(1 - mu^q) dmu"""
			return (1.0-mu_min) - (1-mu_min**(1+p))/(1+p) - (1-mu_min**(1+q))/(1+q) + (1.0-mu_min**(1+p+q))/(1+p+q)

		if input_LDlaw == "claret":
			assert len(coeffs) == 4, "coeffs must be of length 4 for claret limb darkening law"
			source_amps   = list(coeffs)
			source_powers = [0.5, 1.0, 1.5, 2.0] # powers of mu for claret law

		elif input_LDlaw == "power-2":
			assert len(coeffs) == 2, "coeffs must be of length 2 for power-2 limb darkening law"
			c_p2, alpha   = coeffs
			source_amps   = [c_p2]
			source_powers = [alpha]

		# ── Build Gram matrix A (N x N) ───────────────────────────────────────
		int_powers = np.arange(1, n_powers + 1, dtype=float)
		A = np.array([[_inner(m, n,mu_min) for n in int_powers] for m in int_powers])

		# ── Build RHS vector b (N,) ───────────────────────────────────────────
		b = np.array([
			sum(amp * _inner(m, sp, mu_min)
				for amp, sp in zip(source_amps, source_powers))
			for m in int_powers
		])

		# ── Solve and compute RMS residual ────────────────────────────────────
		c = np.linalg.solve(A, b) #returns coeff
		#extract u1 to u4 depending on length of c
		u1 = c[0] if len(c) > 0 else 0.0
		u2 = c[1] if len(c) > 1 else 0.0
		u3 = c[2] if len(c) > 2 else 0.0
		u4 = c[3] if len(c) > 3 else 0.0


		#plot fit to power-2 or claret LD profiles
		if show_plot:
			mu = np.linspace(0, 1, 100)
			if input_LDlaw == "claret":
				ld_profile = 1 - sum(amp * (1 - mu**pow) for amp, pow in zip(source_amps, source_powers))
			elif input_LDlaw == "power-2":
				ld_profile = 1 - c_p2 * (1 - mu**alpha)

			ld_fit = 1 - sum(ci * (1 - mu**pi) for ci, pi in zip(c, int_powers))

			ld_fig = plt.figure(figsize=(6,4))
			plt.plot(mu, ld_profile, label=f"{input_LDlaw} LD profile\ncoeffs={coeffs}")
			plt.plot(mu, ld_fit, label=f"Kopal fit ({n_powers} powers)\n{', '.join([f'u{i+1}={c[i]:.2f}' for i in range(len(c))])}", linestyle="--")
			plt.xlabel("mu")
			plt.ylabel("I(mu)/I(1)")
			plt.title("Limb Darkening Profile and Kopal Fit")
			plt.legend()
			plt.grid()
			plt.show()

		#eqn 2-41 of Kopal 1959
		Xi = np.array( [ 
							1/7  * (210 + 14*u1 - 18*u3 - 35*u4),
							5/42 * (210*u1 + 288*u2 + 315*u3 + 320*u4),
							3/56 * ( 420*(u1-1) + 735*u2 + 932*u3 + 1050*u4) 
					]) / (30 - 10*u1 -15*u2 - 18*u3 - 20*u4)

	alpha2 = 3 / 4 * Xi[0] * (1 + y)
	alpha1 = 1/12 * Xi[1]/Xi[0] * ((y + 2) / (y + 1))
	return SimpleNamespace(alpha1=alpha1, alpha2=alpha2, fig=ld_fig if show_plot else None)


def ellipsoidal_variation_coefficients_old(u1: float, u2: float, y: float):
	"""
	calculate alpha, the ellisoidal varation coefficient. the value is usually close to unity.
	Use function `gravity_darkening_coefficient()` to obtain `g` for CHEOPS and TESS passband and `ldtk_ldc()` for u.
	eqn 12 and 13 of Esteves2013 https://doi.org/10.1088/0004-637X/772/1/51. This is expanded for the case of
	quadratic limb darkening in eqn 9 and 10 of Deline2025 https://arxiv.org/pdf/2505.01544. where alpha_ev=alpha2 and beta_ev=alpha1

	Parameters
	----------
	u1 : float,UFloat
		linear limb darkening coefficient
	u2 : float,UFloat
		quadratic limb darkening coefficient. can be set to zero if using linear limb darkening law
	y : float,UFloat
		gravity darkening coefficient

	Return
	------
	alpha : SimpleNamespace
		alpha1 - float, UFloat: ellipsoidal variation coefficient 1 used in the fractional constants f1 anf f2
		alpha2 - float, UFloat: ellipsoidal variation coefficient 2 used in the main amplitude Aev  

	"""
	alpha2  = 3/10 * ((15+u1+2*u2)/(6-2*u1-u2))  * (1+y)  
	alpha1  = 5/168 * ((35*u1+22*u2)/(15+u1+2*u2)) * ((y+2)/(y+1)) 
	return SimpleNamespace(alpha1=alpha1, alpha2=alpha2)

def ellipsoidal_variation_amplitude( qm:     float | UFloat,
									 aR:     float | UFloat,
									 inc:    float | UFloat, 
									 alpha1: float | UFloat,
									 alpha2: float | UFloat
									 ):
	"""
	calculate theoretical ampltiude of ellipsodial variation
	eqn 9 and 11 of Esteves2013 https://doi.org/10.1088/0004-637X/772/1/51

	Parameters
	----------
	qm : float,UFloat
		planet-to-star mass ratio
	aR : float,UFloat
		scaled semi-major axis
	inc : float,UFloat
		inclination of the orbit
	alpha1 : float,UFloat
		ellipsoidal variation coefficient 1 used in the fractional constants f1 and f2
	alpha2 : float,UFloat
		ellipsoidal variation coefficient 2 used in the main amplitude Aev


	Returns
	-------
	Amplitudes: SimpleNamespace
		f1 : float, UFloat. fractional constant f1
		Aev : float, UFloat. main amplitude of ellipsoidal variation in ppm
	"""
	sini = sin(inc*np.pi/180)
	Aev  = alpha2 * qm*sini**2 / aR**3 * 1e6
	f1   = 3*alpha1/aR * (5*sini**2 - 4)/sini
	return SimpleNamespace(f1=f1, Aev=Aev)


def ellipsoidal_variation_signal(phi, A_ev=0, f1_ev=0, inc_rad=1.571, e=0, w_rad=1.571):
	"""
	Calculate the ellipsoidal variation signal of a planet as a function of phase angle.
	The equation is given as F = Aev * (1 - cos(2*phi) - f1*cos(phi) - f2*cos(3*phi)).
	The semi-amplitude of the ellipsoidal variation is A, f1 and f2 are fractional coefficients controlling
	the fundamental and second harmonics respectively. see eq.8 of Esteves+2013 (https://iopscience.iop.org/article/10.1088/0004-637X/772/1/51)

	Parameters
	-----------
	phi : array-like
		phase angle (2*pi*phase for circular orbit) or true anomaly+omega-pi/2 in radians.
	A_ev : float
		semi-amplitude of the ellipsoidal variation signal
	f1_ev : float
		fractional coefficient controlling the fundamental harmonic
	inc_rad : float
		inclination of the planet orbit in radians. Default is 1.571 (90 degrees)
	e : float
	eccentricity of the planet orbit. Default is 0 (circular orbit)
	w_rad : float
	argument of periastron in radians. Default is 1.571 (90 degrees), which means that the periastron is at quadrature and the phase curve is symmetric around the secondary eclipse.

	Returns
	--------
	F_ev : array-like
		ellipsoidal variation signal as a function of phase
	"""
	true_anom = phi - w_rad + np.pi/2  # true anomaly in radians
	ecc_fac = (1 - e**2) / (1 + e * np.cos(true_anom))    # eccentricity factor missing from all instances of a/R*
	A_ev = A_ev * ecc_fac**-3  # scale the amplitude by the eccentricity factor
	f1_ev = f1_ev * ecc_fac**-1  # scale f1 by the eccentricity factor
	
	f2_ev = 5 / 3 * (f1_ev * np.sin(inc_rad) ** 2) / (5 * np.sin(inc_rad) ** 2 - 4)
	f0 = (1 - f1_ev - f2_ev)  # normalization factor to ensure stellar flux level is zero at mid occultation
	F_ev = -A_ev * (np.cos(2 * phi) + 
				 	f1_ev * np.cos(phi) + 
					f2_ev * np.cos(3 * phi) - f0)
	return F_ev


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
	if star_spec != "bb" and planet_spec != "bb":
		try:
			from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
			ip = create_bt_settl_interpolator()
			ip2 = create_husser2013_interpolator()
			ip.fill_value, ip2.fill_value = None, None
			ip.bounds_error, ip2.bounds_error = False, False 
		except ImportError:
			raise ImportError("pytransit is required to use the 'BT-Settl' and 'Husser2013' stellar spectra. Please install pytransit to use these features.")

	#bb = lambda T,l: (2*h*c**2/(l*u.nm)**5 / (np.exp(h*c/(l*u.nm*k_B*T*u.K))-1)).to(u.W/u.m**3)
	assert isinstance(Tst,(float, int, UFloat)), 'Tst must be either float, int or ufloat'

	bb  = blackbody

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


def gravity_darkening_coefficient(Teff:tuple, logg:tuple, Z:tuple=None, band="TESS", 
									beta1_ch=0.3, interp_method="linear"):
	"""
	get gravity darkening coefficients for TESS and CHEOPS using ATLAS stellar models
	use tables:
	TESS: claret2017 (https://www.aanda.org/articles/aa/full_html/2017/04/aa29705-16/aa29705-16.html) for TESS 
	CHEOPS: table 14 of claret2021 (https://iopscience.iop.org/article/10.3847/2515-5172/abdcb3)
	JWST: table 1 of Claret & Torres 2026 (https://iopscience.iop.org/article/10.3847/2515-5172/ae38df)


	Parameters
	----------
	Teff : tuple
		effective temperature of star given as (mean,std)
	logg : tuple
		surface gravity
	Z : tuple (optional)
		metallicity. default is None
	band : str, optional
		instrument, either 'TESS' or 'CHEOPS', or one of these JWST filters: ['JWST/F210M', 'JWST/F322W2', 'JWST/F444W', 
		'JWST/SOSS1', 'JWST/SOSS2', 'JWST/F277W', 'JWST/G235H', 'JWST/G235M', 'JWST/G395H', 'JWST/G395M', 'JWST/PRISM']
	beta1_ch : float, optional
		beta1 is the gravity-darkening exponent (GDE), a bolometric quantity. only used for CHEOPS 
		where the gravity darkening coefficient is given by: y = beta1_ch*y1 + y2, by default 0.3
	interp_method : str, optional
		method for interpolation. Supported are "linear", "nearest", "slinear", "cubic", "quintic" and "pchip".
		Default is "linear"
	Returns
	-------
	gdc: Ufloat
		gravity darkening coefficient and uncertainty

	"""
	jwst_filts = ['JWST/F210M', 'JWST/F322W2', 'JWST/F444W', 'JWST/SOSS1', 'JWST/SOSS2', 'JWST/F277W', 
					'JWST/G235H', 'JWST/G235M', 'JWST/G395H', 'JWST/G395M', 'JWST/PRISM']

	assert band in ["TESS","CHEOPS"]+jwst_filts,f"band must be 'TESS', 'CHEOPS' or one of the JWST filters: {jwst_filts}"
	
	logTeff = log10(ufloat(Teff[0],Teff[1]))
	
	if band == "CHEOPS":
		df = pd.read_csv(os.path.dirname(__file__)+'/data/ATLAS_GDC-CHEOPS.dat', sep='\s+', skiprows=1)
		df1 = df.iloc[::2]
		df1.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y1']
		df2 = df.iloc[1::2]
		df2.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y2']

		df  = pd.merge(df1, df2, on=['Z', 'Vel', 'logg', 'logTeff'])  
		df['y'] = beta1_ch*df['y1'] + df['y2']       
		df= df[['Z', 'logg', 'logTeff', 'y']]


	elif band == "TESS":
		df = pd.read_html("http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/html?J/A+A/600/A30/table29.dat.gz")[0]
		df.columns = [ 'Z','Vel','logg', 'logTeff', 'y','Mod']

	elif band.startswith("JWST/"):
		file_path= "https://content.cld.iop.org/journals/2515-5172/10/1/16/revision1/rnaasae38dft1_mrt.txt"
		df = pd.read_csv(file_path, sep=r'\s+', skiprows=24, names=['logg', 'logTeff'] + jwst_filts)
		df = df[['logg', 'logTeff', band]]
		df = df.rename(columns={band: 'y'})

	def get_y(logTeff, logg, Z=None):
		if Z:
			row = df[ (df['logg'] == logg) & (df['logTeff'] == logTeff) & (df['Z'] == Z)]
		else:
			row = df[ (df['logTeff'] == logTeff) & (df['logg'] == logg)]                    
		if not row.empty:
			return row['y'].values[0]
		else:
			return np.nan	

	#create interpolation function
	from scipy.interpolate import RegularGridInterpolator
	logTeff_list = sorted(df['logTeff'].unique())
	logg_list 	 = sorted(df['logg'].unique())
	if Z:
		z_list  = sorted(df['Z'].unique())
		dff     = np.empty((len(logTeff_list),len(logg_list),len(z_list)))
	else:
		dff     = np.empty((len(logTeff_list),len(logg_list)))

	for i,log_T in enumerate(logTeff_list):
		for j,log_g in enumerate(logg_list):
			if Z:
				for k,_z in enumerate(z_list):
					y_val = get_y(log_T, log_g, _z)
					dff[i,j,k] = y_val
			else:
				y_val = get_y(log_T, log_g)
				dff[i,j] = y_val

	interp_func = RegularGridInterpolator(  points = (logTeff_list, logg_list, z_list) if Z else (logTeff_list, logg_list), 
											values = dff,
											bounds_error=False, method=interp_method)
	nsamp   = 100000
	y_array      = np.empty(nsamp)
	logg_norm    = norm(*logg).rvs(nsamp)
	logTeff_norm = norm(logTeff.n, logTeff.s).rvs(nsamp)
	if Z: z_norm = norm(*Z).rvs(nsamp)
	
	for i in range(nsamp):
		if Z:
			point       = (logTeff_norm[i], logg_norm[i], z_norm[i])
		else:
			point       = (logTeff_norm[i], logg_norm[i])

		y_array[i]      = interp_func(point)
	
	y_median = interp_func( (logTeff.n, logg[0], Z[0]) ) if Z else interp_func( (logTeff.n, logg[0]) ) 
	y = ufloat(y_median, np.nanmax(np.diff(np.nanquantile(y_array, [0.16,0.5,0.84]))) )
	return y


def gravity_darkening_coefficient_old(Teff:tuple, logg:tuple, Z:tuple=None, band="TESS", beta1_ch=0.3):
	"""
	get gravity darkening coefficients for TESS and CHEOPS using ATLAS stellar models
	use tables:
	TESS: claret2017 (https://www.aanda.org/articles/aa/full_html/2017/04/aa29705-16/aa29705-16.html) for TESS 
	CHEOPS: table 14 of claret2021 (https://iopscience.iop.org/article/10.3847/2515-5172/abdcb3)
	JWST: table 1 of Claret & Torres 2026 (https://iopscience.iop.org/article/10.3847/2515-5172/ae38df)


	Parameters
	----------
	Teff : tuple
		effective temperature of star given as (mean,std)
	logg : tuple
		surface gravity
	Z : tuple (optional)
		metallicity. default is None
	band : str, optional
		instrument, either 'TESS' or 'CHEOPS', or one of these JWST filters: ['JWST/F210M', 'JWST/F322W2', 'JWST/F444W', 
		'JWST/SOSS1', 'JWST/SOSS2', 'JWST/F277W', 'JWST/G235H', 'JWST/G235M', 'JWST/G395H', 'JWST/G395M', 'JWST/PRISM']
	beta1_ch : float, optional
		beta1 is the gravity-darkening exponent (GDE), a bolometric quantity. only used for CHEOPS 
		where the gravity darkening coefficient is given by: y = beta1_ch*y1 + y2, by default 0.3

	Returns
	-------
	gdc: Ufloat
		gravity darkening coefficient and uncertainty

	Examples
	--------
	>>> from planet_tools.phasecurve import gravity_darkening_coefficient
	>>> gdc = gravity_darkening_coefficient(Teff=(6000,100), logg=(4.5,0.1), Z=(0.0,0.1), band="TESS")
	>>> print(gdc)
	0.431+/-0.013

	"""
	jwst_filts = ['JWST/F210M', 'JWST/F322W2', 'JWST/F444W', 'JWST/SOSS1', 'JWST/SOSS2', 'JWST/F277W', 
					'JWST/G235H', 'JWST/G235M', 'JWST/G395H', 'JWST/G395M', 'JWST/PRISM']

	assert band in ["TESS","CHEOPS"]+jwst_filts,f"band must be 'TESS', 'CHEOPS' or one of the JWST filters: {jwst_filts}"
	
	logTeff = log10(ufloat(Teff[0],Teff[1]))
	
	if band == "CHEOPS":
		df = pd.read_csv(os.path.dirname(__file__)+'/data/ATLAS_GDC-CHEOPS.dat', delim_whitespace=True, skiprows=1)
		df1 = df.iloc[::2]
		df1.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y1']
		df2 = df.iloc[1::2]
		df2.columns = ['Z', 'Vel', 'logg', 'logTeff', 'y2']

		df  = pd.merge(df1, df2, on=['Z', 'Vel', 'logg', 'logTeff'])        

	elif band == "TESS":
		df = pd.read_html("http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/html?J/A+A/600/A30/table29.dat.gz")[0]
		df.columns = [ 'Z','Vel','logg', 'logTeff', 'y','Mod']

	elif band.startswith("JWST/"):
		file_path= "https://content.cld.iop.org/journals/2515-5172/10/1/16/revision1/rnaasae38dft1_mrt.txt"
		df = pd.read_csv(file_path, sep=r'\s+', skiprows=24, names=['logg', 'logTeff'] + jwst_filts)
		df = df[['logg', 'logTeff', band]]
		df = df.rename(columns={band: 'y'})

		spline = RectBivariateSpline(df['logg'].unique(), df['logTeff'].unique(), 
									df.pivot(index='logg', columns='logTeff', values='y').values)
		logg_norm = norm(*logg).rvs(1000000)
		logTeff_norm = norm(logTeff.n, logTeff.s).rvs(1000000)
		y_array = spline.ev(logg_norm, logTeff_norm)
		y = ufloat(np.median(y_array), np.max(np.diff(np.quantile(y_array, [0.16,0.5,0.84]))) )
		return y
		
	mlogg = (df.logg >= (logg[0]-3*logg[1])) & (df.logg <= (logg[0]+3*logg[1]) )
	mteff = (df.logTeff >= (logTeff.n-3*logTeff.s)) & (df.logTeff <= (logTeff.n+3*logTeff.s))
	if Z: mZ = (df.Z >= (Z[0]-3*Z[1])) & (df.Z <= (Z[0]+3*Z[1]))

	m = mlogg & mteff & mZ if Z else mlogg & mteff
	assert sum(m) >= 2,"there are no or not enough samples in the range. modify input parameters and/or their uncertanities"

	res = df[m]
	#calculate euclidean distance of the samples from the input values (teff, logg,Z)
	if Z:
		diff = np.array((res["logTeff"],res["logg"],res["Z"])).T - np.array([logTeff.n, logg[0],Z[0]])
	else:
		diff = np.array((res["logTeff"],res["logg"])).T - np.array([logTeff.n, logg[0]])

	res["dist"]=np.sqrt(np.sum(diff**2,axis=1))
	#assign weights based on closest dist and normalize so sum(weights)=1
	res["weights"]=  (1-res["dist"])/sum((1-res["dist"]))

	if band == "TESS" or band.startswith("JWST/"):
		y_mean = np.sum(res["weights"]*res["y"]) / sum(res["weights"])
		y_std  = np.std(res["weights"]*res["y"]) / sum(res["weights"])
		y      = ufloat(round(y_mean,4), round(y_std,4))
	elif band == "CHEOPS":
		y1_mean = np.sum(res["weights"]*res["y1"]) / sum(res["weights"])
		y1_std  = np.std(res["weights"]*res["y1"]) / sum(res["weights"])
		y1      = ufloat(round(y1_mean,4), round(y1_std,4))

		y2_mean = np.sum(res["weights"]*res["y2"]) / sum(res["weights"])
		y2_std  = np.std(res["weights"]*res["y2"]) / sum(res["weights"])
		y2      = ufloat(round(y2_mean,4), round(y2_std,4))

		# y = beta1*y1 + y2
		y = beta1_ch*y1 + y2

	return y


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
		heat redistribution factor.default is 1/4 for uniform heat distribution and 2/3 for None. Note f = 2/3 - 5/12*eps where eps is the heat redistribution efficiency
		
	Returns
	-------
	
	T_eq: Array-like;
		Equilibrium temperature of the planet
	"""
	# print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
	return T_st*sqrt(1/a_r)* ((1-A_b)*f)**0.25


def T_day(T_st, a_r, A_b=0, eps=1, e=0, w=90):
	"""
	calculate dayside temperature  of planet in Kelvin (eq 4&5 of Cowan, N. B., & Agol, E. 2011,ApJ,729, 54)

	Parameters
	----------
	T_st: Array-like;
		Effective Temperature of the star

	a_r: Array-like;
		Scaled semi-major axis of the planet orbit

	A_b: Array-like;
		Bond albedo of the planet. default is zero

	e: Array-like;
		heat redistribution efficiency. 1 for uniform heat distribution and 0 for none

	Returns
	-------
	T_eq: Array-like;
		temperature of the planet
	"""
	# print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
	if e != 0:
		f = 3*np.pi/2 - radians(w)              # true anomaly at occultation
		r = a_r * (1 - e**2) / (1 + e * cos(f)) # star-planet distance at occultation
	else:
		r = a_r
	return T_st * sqrt(1 / r) * ((1 - A_b)) ** 0.25 * (2 / 3 - 5 / 12 * eps) ** 0.25


def T_night(T_st,a_r, A_b =0 , eps = 0, e=0, w=90): 
	"""
	calculate nightside temperature  of planet in Kelvin (eq 4&5 of Cowan, N. B., & Agol, E. 2011,ApJ,729, 54)

	
	Parameters
	----------
	T_st: Array-like;
		Effective Temperature of the star

	a_r: Array-like;
		Scaled semi-major axis of the planet orbit

	A_b: Array-like;
		Bond albedo of the planet. default is zero

	eps: Array-like;
		heat redistribution efficiency. 1 for uniform heat distribution and 0 for none

	Returns
	-------
	T_night: Array-like;
		temperature of the planet
	"""
	# print("T_st is {0:.2f}, a_r is {1:.2f}".format(T_st,a_r))
	if e != 0:
		f = np.pi/2 - radians(w)              # true anomaly at occultation
		r = a_r * (1 - e**2) / (1 + e * cos(f)) # star-planet distance at occultation
	else:
		r = a_r
	return T_st * sqrt(1 / r) * ((1-A_b))**0.25 * (eps/4)**0.25

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
	dopp       =  A_dopp * np.sin(phi) #+ A_dopp
	return atm_signal, ellps, dopp


def TSM(Rp, Mp, Rs, Teq, mj):
	"""
	Transmission spectroscopy metric according to kempton+2018(https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K)

	Parameters
	----------
	Rp : float
		planet radius in units of earth radii
	Mp : float
		planet mass in units of earth mass. if not provided, it will be estimated using the mass-radius relation from Chen & Kipping 2017(https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C)
	Rs : float
		stellar radius in units of solar radii
	Teq : float
		planet equilibrium temperature in K (full heat redistribution, zero albedo)
	mj : float
		the apparent magnitude of the host star in the J band
	"""
	if Rp <= 1.5 : 
		fac = 0.19
	elif Rp <= 2.75 : 
		fac = 1.26
	elif Rp <= 4 : 
		fac = 1.28
	else : 
		fac = 1.15

	if Mp is None:
		if Rp < 1.23 : 
			Mp = 0.9718 * Rp**3.58  
		elif Rp > 14.26: 
			Mp = 317
		else:
			Mp = 1.436 * Rp**1.70

	TSM = fac * Rp**3/Mp * Teq/Rs**2 * 10**(-mj/5)
	return TSM

def ESM(Teq, Teff, RpRs, Kmag):
	"""
	Emission spectroscopy metric according to kempton+2018(https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K)

	Parameters
	----------
	Teq : float
		planet equilibrium temperature in K (full heat redistribution, zero albedo)
	Teff : float
		stellar effective temperature in K
	RpRs : float
		planet-to-star radius ratio
	Kmag : float
		the apparent magnitude of the host star in the K band
	"""
	from astropy.modeling.models import BlackBody
	depth = (RpRs)**2 * 1e6 #occultation depth in ppm assuming planet emits as a blackbody and has the same temperature as its equilibrium temperature
	
	bb_tday     = BlackBody(temperature=1.1*Teq*u.K)
	bb_tstar    = BlackBody(temperature=Teff*u.K)
	wavelength  = 7.5*u.micron #central wavelength of the MIRI LRS mode
	planet_flux = bb_tday(wavelength)
	star_flux   = bb_tstar(wavelength)
	em_ratio    = (planet_flux / star_flux).value

	ESM = 4.29 * depth * em_ratio * 10**(-Kmag/5)
	return ESM



def eclipse_depth_predict2(RpRs, aR,Ag, Tp, Teff, flt):
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
	try:
		from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
	except ImportError:
		raise ImportError("pytransit is required to use the 'BT-Settl' and 'Husser2013' stellar spectra. Please install pytransit to use these features.")

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


def brightness_temp(RpRs, aR, D, Teff, flt, tmin=500, tmax=4500,Ag =0, st_spec="BT-Settl"):
	"""
	calculates planet brightness temperature given eclipse depth measurement D in ppm

	Parameters
	----------
	RpRs : float
		planet-to-star radius ratio
	aR : float
		scaled semi-major axis
	D  : float
		eclipse depth in ppm
	Teff : float
		stellar equilibrium temperature
	flt : SVOfilter
		observation passband
	tmin, tmax : float
		minimum and maximum planet dayside temperature to use in the fit
	Ag : float
		geometric albedo
	st_spec: string
		stellar spectrum to use. Either "BT-Settl" or "bb"
		
	Returns
	-------
	Tbr: float;
		brightness temperature in kelvin
	"""
	from scipy.optimize import minimize_scalar
	bb  = blackbody

	if st_spec != "bb":
		try:
			from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
			ip = create_bt_settl_interpolator()
			ip.fill_value = None
			ip.bounds_error = False
		except ImportError:
			raise ImportError("pytransit is required to use the 'BT-Settl' and 'Husser2013' stellar spectra. Please install pytransit to use these features.")

	if st_spec == "BT-Settl": fs = ip((Teff,flt.wavelength))/np.pi
	elif st_spec == "bb" :    fs = bb(Teff,np.array(flt.wavelength))
	else: raise(ValueError("st_spec must be either 'BT-Settl' or 'bb.'"))
		
	def minfun(tbr):
		if tbr < tmin or tbr > tmax:
			return np.inf
		fp = bb(tbr,np.array(flt.wavelength))
		em_ratio = np.average(fp/fs, weights=flt.transmission) #emission ratio
		Dfit = (em_ratio*RpRs**2 + Ag*(RpRs/aR)**2)  * 1e6
		return np.fabs(Dfit - D)
	
	return minimize_scalar(minfun, [tmin, tmax], bounds=(tmin, tmax)).x


def get_new_ATLAS_spectrum(Teff,logg, z, lib="mps1", file_path="/Users/tunde/exotic-ld_data"):
	
	assert lib in ["mps1", "mps2","kurucz"],f"mps_set can only be one of 'mps1', 'mps2','kurucz'."
	DA = locals().copy()
	_  = DA.pop("lib")
	_  = DA.pop("file_path")
	
	
	if lib in ["mps1", "mps2"]:
		data = {"z": np.array(
						[-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8,
						 -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75,
						 -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6,
						 -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
						 -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4,
						 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45,
						 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
			"Teff": np.arange(3500, 9050, 100),
			"logg":np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])}
	else:
		data = {"z": np.array(
						[-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
						 -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
			"Teff": np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000,
							  5250, 5500, 5750, 6000, 6250, 6500]),
			"logg":np.array([4.0, 4.5, 5.0])}


	# check if input teff,logg,or z input is  a point in the mps grid, if not take closest grid point
	for par in DA.keys():
		#first ensure input value is not outside grid range
		assert  min(data[par]) <= DA[par] <= max(data[par]),f"{par}={DA[par]} is outside the range [{min(data[par])},{max(data[par])}]"
		
		if DA[par] not in data[par]:
			idx = np.argmin( abs(data[par]-DA[par]) )
			print(f"{par}={DA[par]} is not a point in the {lib}-ATLAS grid, taking closest value of {data[par][idx]}")
			DA[par] = data[par][idx]
			
	file_name = os.path.join(file_path,
							 lib,f"MH{DA['z']:.1f}",f"teff{DA['Teff']:.0f}",f"logg{DA['logg']:.1f}",
							f"{lib}_spectra.dat")
	print(f"returning spectra for {DA}")
	
	mus = np.loadtxt(file_name, skiprows=1, max_rows=1)
	stellar_data = np.loadtxt(file_name, skiprows=2)
	wave, spec_mus = stellar_data[:, 0], stellar_data[:, 1:]
	return wave, mus, spec_mus*1e-8
