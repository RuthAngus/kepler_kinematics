# Functions needed for inferring velocities.

import numpy as np
import pandas as pd
import astropy.stats as aps
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import aviary as av

from .velocities import get_solar_and_R_gal
sun_xyz, sun_vxyz, R_gal, galcen_frame = get_solar_and_R_gal()

# # Solar coords
# sun_xyz = [-8.122, 0, 0] * u.kpc
# sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s
# # sun_vxyzCD = coord.CartesianDifferential(12.9, 245.6, 7.78, u.km/u.s)

# galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
#                                     galcen_v_sun=sun_vxyz,
#                                     z_sun=0*u.pc)

# # Pre-compute the rotation matrix to go from Galactocentric to ICRS
# # (ra/dec) coordinates
# R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

# Calculate prior parameters from vx, vy, vz distributions.
import pkg_resources
vel_data = pkg_resources.resource_filename(__name__,
                                           "gaia_mc5_velocities.csv")
vels = pd.read_csv(vel_data)
m = vels.radial_velocity.values != 0
m &= np.isfinite(vels.basic_vx.values)
m &= np.isfinite(vels.basic_vy.values)
m &= np.isfinite(vels.basic_vz.values)
vels = vels.iloc[m]

# Calculate covariance between velocities
VX = np.stack((vels.basic_vx.values, vels.basic_vy.values,
               vels.basic_vz.values, np.log(1./vels.parallax.values)), axis=0)
mean = np.mean(VX, axis=1)
cov = np.cov(VX)


def proper_motion_model(params, pos):
    """
    The model. Calculates proper motion from velocities and positions.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].

    Returns:
        pm_from_v (list): The ra and dec proper motions, calculated from
            velocity model parameters and observed positions.
        rv_from_v (list): The RV, calculated from velocity model parameters
            and observed positions.

    """

    # Unpack parameters and make distance linear.
    vx, vy, vz, lnD = params
    D = np.exp(lnD)

    # Calculate XYZ position from ra, dec and parallax
    c = coord.SkyCoord(ra = pos[0]*u.deg,
                       dec = pos[1]*u.deg,
                       distance = D*u.kpc)
    galcen = c.transform_to(galcen_frame)

    # Calculate pm and rv from XYZ and V_XYZ
    V_xyz_units = [vx, vy, vz]*u.km*u.s**-1
    pm_from_v, rv_from_v = av.get_icrs_from_galactocentric(galcen.data.xyz,
                                                           V_xyz_units,
                                                           R_gal, sun_xyz,
                                                           sun_vxyz)
    return pm_from_v, rv_from_v


def lnlike_one_star(params, pm, pm_err, pos, pos_err):
    """
    log-likelihood of proper motion and position, given velocity & distance.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pm (list): Proper motion in RA and dec in mas/yr. [pmra, pmdec].
        pm_err (list): Uncertainties on proper motion in RA and dec in
            mas/yr. [pmra_err, pmdec_err]
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].
        pos_err (list): Uncertainties on positional coordinates, RA_err
            [deg], dec_err [deg] and parallax_err [mas].

    Returns:
        The log-likelihood.

    """

    # Unpack parameters and make distance linear.
    vx, vy, vz, lnD = params
    D = np.exp(lnD)

    pm_from_v, rv_from_v = proper_motion_model(params, pos)

    # Compare this proper motion with observed proper motion.
    return -.5*(pm_from_v[0].value - pm[0])**2/pm_err[0]**2 \
           -.5*(pm_from_v[1].value - pm[1])**2/pm_err[1]**2 \
           -.5*(1./D - pos[2])**2/pos_err[2]**2


def lnGauss(x, mu, sigma):
    """
    A log-Gaussian.

    """
    ivar = 1./sigma**2
    return -.5*(x - mu)**2 * ivar


def multivariate_lngaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    from scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return -fac / 2 - np.log(N)


def lnprior(params):
    """
    The log-prior over distance and velocity parameters.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
           ln(distance [kpc]).

    Returns:
        The log-prior.

    """

    vx, vy, vz, lnD = params

    # Multivariate Gaussian prior
    pos = np.stack((vx, vy, vz, lnD))
    return float(multivariate_lngaussian(pos, mean, cov))


def lnprob(params, pm, pm_err, pos, pos_err):
    """
    log-probability of distance and velocity, given proper motion and position

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pm (list): Proper motion in RA and dec in mas/yr. [pmra, pmdec].
        pm_err (list): Uncertainties on proper motion in RA and dec in
            mas/yr. [pmra_err, pmdec_err]
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].
        pos_err (list): Uncertainties on positional coordinates, RA_err [deg],
            dec_err [deg] and parallax_err [mas].

    Returns:
        The log-probability.

    """

    return lnlike_one_star(params, pm, pm_err, pos, pos_err) + lnprior(params)
