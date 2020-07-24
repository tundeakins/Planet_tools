# Planet Tools
Useful functions to perform everyday conversions and calculation of quantities in exoplanetary science.

The **convert_params** module contains the following functions:

 - **a_r** - convert period to scaled semi-major axis.
 - **a_r_to_rho_star** - Compute transit derived stellar density from the planet period and scaled semi major axis
 - **impact_parameter** - convert inclination to impact parameter b
 
 - **inclination** - convert impact parameter b to inclination in degrees.
 - **kipping_ld** - Re-parameterize quadratic limb darkening parameters $u_{1}$ and $u_{2}$ according to Kipping (2013)
 - **prot** - convert stellar rotation velocity vsini in km/s to rotation period in days.
 - **rho_to_a_r** - convert stellar density to semi-major axis of planet with a particular period
 - **vsini** - convert stellar rotation period to vsini in km/s.
 
The **calculate_params** module contains the follwing functions:

 - **RL_Rroche** - calculate ratio of Laplace radius to Roche radius
 - **R_hill** - compute the hill radius of a planet
 - **R_roche** - Compute roche radius of a planet as a function of the planet's radius
 - **T_eq** - calculate equilibrium temperature of planet in Kelvin
 - **ingress_duration** - calculate the duration of ingress/egress.
 - **ldtk_ldc** - estimate quadratic limb darkening coefficients for a given star
 - **phase_fold** - Given the observation time and period, return the phase of each observation time
 - **planet_prot** - Calculate period of rotation of a planet
 - **sigma_CCF** - Calculate CCF width of non-rotating star in km/s based on resolution of spectrograph
 - **transit_duration** - Calculate the transit duration
 
The **some_stats** module contains the following functions:

 - **bic** - Compute the bayesian information criteria
 - **aic** - Calculate the Aikake information criterion.
 - **rmse** - Calculate the root-mean-square of the inputed error array (residuals)
 - **sse** - Calculate the sum of squared error of inputed error array (residuals)
