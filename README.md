# Planet Tools
Useful functions to perform everyday conversions and calculation of quantities in exoplanetary science. It is a work in progress, so I  am constantly adding and modifying them.

<img src="https://github.com/tundeakins/Planet_tools/blob/master/Planet_tools/planet_tools.png" width="300">

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tundeakins/Planet_tools/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/Planet-tools.svg)](https://badge.fury.io/py/Planet-tools)


To install
```bash
pip install Planet-tools
```
or

```bash
git clone https://github.com/tundeakins/Planet_tools.git
cd Planet_tools
python setup.py install
```

The **convert_params** module contains the following functions:

 - **P_to_aR** - convert period to scaled semi-major axis.
 - **aR_to_rho_star** - Compute transit derived stellar density from the planet period and scaled semi major axis
 - **impact_parameter** - Convert inclination to impact parameter b
 
 - **inclination** - Convert impact parameter b to inclination in degrees.
 - **kipping_LD** - Re-parameterize quadratic limb darkening parameters $u_{1}$ and $u_{2}$ according to Kipping (2013)
 - **kipping_to_quadLD** - transform kipping (2013) ldcs $q_{1}$ and $q_{2}$ to the usual quadratic limb darkening parameters $u_{1}$ and $u_{2}$.)
 - **kipping_to_Power2LD** - Re-parameterize kipping (2013)[1] ldcs $q_{1}$ and $q_{2}$ to the Power-2 limb darkening parameters $h_{1}$ and $h_{2}$
 - **Power2_to_kippingLD** - Transform Power-2 limb darkening parameters $h_{1}$ and $h_{2}$ (Maxted 2018) to Kipping (2013) coefficients.
 - **prot** - Convert stellar rotation velocity vsini in km/s to rotation period in days.
 - **rho_to_aR** - Convert stellar density to semi-major axis of planet with a particular period
 - **vsini** - Convert stellar rotation period to vsini in km/s.
 
The **calculate_params** module contains the follwing functions:

 - **RL_Rroche** - Calculate ratio of Laplace radius to Roche radius
 - **R_hill** - Compute the hill radius of a planet
 - **R_roche** - Compute roche radius of a planet as a function of the planet's radius
 - **T_eq** - Calculate equilibrium temperature of planet in Kelvin
 - **ingress_duration** - Calculate the duration of ingress/egress.
 - **ldtk_ldc** - Estimate quadratic limb darkening coefficients for a given star
 - **phase_fold** - Given the observation time and period, return the phase of each observation time
 - **planet_prot** - Calculate period of rotation of a planet
 - **sigma_CCF** - Calculate CCF width of non-rotating star in km/s based on resolution of spectrograph
 - **transit_duration** - Calculate the transit duration

The **estimate_effect** module contains the following functions:

 - **photo_granulation** - Estimate the amplitude and timescale of granulation noise in photometric observations as given by Gilliland 2011
 - **chaplin_exptime** - Compute the optimal exposure time to reduce stellar p-mode oscillation amplitude in the given star to 0.1m/s and 0.09m/s according to Chaplin et al. 2019.
 - **rv_precision_degrade** - Calculate factor by which RV precision of a stellar spectral type degrades due to vsini.
 
The **some_stats** module contains the following functions:

 - **bic** - Compute the bayesian information criteria
 - **aic** - Calculate the Aikake information criterion.
 - **rmse** - Calculate the root-mean-square of the inputed error array (residuals)
 - **sse** - Calculate the sum of squared error of inputed error array (residuals)
 - **mse** - Calculate the mean-square error of imputed error array
 - **mae** - Calculate the mean-absolute error of imputed error array
 - **rse** - Calculate the relative-square error from the observed and calculated values
 - **rae** - Calculate the relative-absolute error from the observed and calculated values 
 - **r_squared** - Calculate the R2_score commonly

