# Here are the functions I need to infer velocities with PyMC3.
# test

import numpy as np
import pandas as pd

import theano.tensor as tt
import pymc3 as pm
import exoplanet as xo

import starspot as ss

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors

import pkg_resources

def deg_to_rad(deg):
    """
    Convert angle in degrees to angle in radians

    """

    return deg * (2 * np.pi) / 360


# # Constants and global variables
# # Solar coords
# sun_xyz = [-8.122, 0, 0] * u.kpc
# sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s
# # sun_vxyz = coord.CartesianDifferential(12.9, 245.6, 7.78, u.km/u.s)

# galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
#                                     galcen_v_sun=sun_vxyz,
#                                     z_sun=0*u.pc)

# # Pre-compute the rotation matrix to go from Galactocentric to ICRS
# # (ra/dec) coordinates
# R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

from .velocities import get_solar_and_R_gal
sun_xyz, sun_vxyz, R_gal, galcen_frame = get_solar_and_R_gal()


# Coordinates of galactic centre
ra_gc_deg, dec_gc_deg = 266.4051, -28.936175
ra_gc, dec_gc = deg_to_rad(np.array([ra_gc_deg, dec_gc_deg]))

# Angle for rotating to align with plane of Galaxy.
eta_deg = 58.5986320306
eta = deg_to_rad(eta_deg)

# Distance from Galactic centre and height above midplane.
d_gc = np.abs(sun_xyz[0]).value
zsun = 0


# def get_prior():
#     """
#     Calculate mean and covariance of multivariate Gaussian prior.

#     Returns:
#         mean, cov
#     """
#     vel_data = pkg_resources.resource_filename(__name__,
#                                            "../data/gaia_mc5_velocities.csv")
#     vels = pd.read_csv(vel_data)
#     m = vels.radial_velocity.values != 0
#     m &= np.isfinite(vels.basic_vx.values)
#     m &= np.isfinite(vels.basic_vy.values)
#     m &= np.isfinite(vels.basic_vz.values)
#     vels = vels.iloc[m]

#     # Calculate covariance between velocities
#     VX = np.stack((vels.basic_vx.values, vels.basic_vy.values,
#                 vels.basic_vz.values, np.log(1./vels.parallax.values)), axis=0)
#     return np.mean(VX, axis=1), np.cov(VX)


def get_prior(cuts="all"):
    """
    Calculate mean and covariance of multivariate Gaussian prior.

    Returns:
        mean, cov
    """
    vel_data = pkg_resources.resource_filename(
        __name__, "mc_san_gaia_lam.csv")
    df = pd.read_csv(vel_data)

    lnD = np.log(1./df.parallax)
    finite = np.isfinite(df.vx.values) & np.isfinite(df.vy.values) \
        & np.isfinite(df.vz.values) & np.isfinite(lnD)

    nsigma = 3
    mx = ss.sigma_clip(df.vx.values[finite], nsigma=nsigma)
    my = ss.sigma_clip(df.vy.values[finite], nsigma=nsigma)
    mz = ss.sigma_clip(df.vz.values[finite], nsigma=nsigma)
    md = ss.sigma_clip(lnD[finite], nsigma=nsigma)
    m = mx & my & mz & md

    # vel_data = pkg_resources.resource_filename(
    #     __name__, "gaia_mc5_velocities.csv")
    # df = pd.read_csv(vel_data)
    # lnD = np.log(1./df.parallax)
    # df["vx"] = df.basic_vx.values
    # df["vy"] = df.basic_vy.values
    # df["vz"] = df.basic_vz.values
    # finite = np.isfinite(df.vx.values) & np.isfinite(df.vy.values) \
    #     & np.isfinite(df.vz.values) & np.isfinite(lnD)
    # m = np.isfinite(df.vx.values[finite])

    gmag = df.phot_g_mean_mag.values[finite][m]
    m_faint = gmag > 13.56
    m_bright = gmag < 13.56

    if cuts == "all":  # No faint or bright cuts on the prior.
        mu, cov = mean_and_var(df.vx.values[finite][m],
                               df.vy.values[finite][m],
                               df.vz.values[finite][m],
                               lnD[finite][m])
    elif cuts == "faint":
        mu, cov = mean_and_var(df.vx.values[finite][m][m_faint],
                               df.vy.values[finite][m][m_faint],
                               df.vz.values[finite][m][m_faint],
                               lnD[finite][m][m_faint])
    elif cuts == "bright":
        mu, cov = mean_and_var(df.vx.values[finite][m][m_bright],
                               df.vy.values[finite][m][m_bright],
                               df.vz.values[finite][m][m_bright],
                               lnD[finite][m][m_bright])
    return mu, cov

def mean_and_var(vx, vy, vz, lnD):
    V = np.stack((vx, vy, vz, lnD), axis=0)
    return np.mean(V, axis=1), np.cov(V)


def get_tangent_basis(ra, dec):
    """
    row vectors are the tangent-space basis at (alpha, delta, r)
    ra, dec in radians
    """

    # Precompute this matrix and save for each star.
    M = np.array([
        [-np.sin(ra), np.cos(ra), 0.],
        [-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)],
        [np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]
    ])
    return M


def cartesian_to_angular_no_units(km_s, kpc):
    """
    Unit conversion: convert velocities in km/s to proper motions in mas/yr

    Args:
        km_s (float or array): velocity in km/s
        kpc (float or array): distance in kpc

    Returns:
        mas (float or array): proper motion in mas/yr

    """

    to_yr = 365.2425 * 24 * 3600
    to_m = 1000
    to_kpc = 1./3.0856775814671917e+19
    to_deg = 360/(2*np.pi)
    to_mas = 3600 * 1000
    return km_s * to_m * to_yr * to_kpc / kpc * to_deg * to_mas


def tt_r_icrs_norm(ra_deg, dec_deg):
    """
    Calc r_icrs rotation matrix (convert from equatoral to cartesian).

    Args:
        ra_deg (float): ra in degrees.
        dec_deg (float): dec in degrees.

    Returns:
        r_icrs (array): The r_ircs rotation matrix.

    """

    ra, dec = deg_to_rad(np.array([ra_deg, dec_deg]))
    return np.array([[np.cos(ra) * np.cos(dec)],
                     [np.sin(ra) * np.cos(dec)],
                     [np.sin(dec)]])


def tt_eqtogal(ra, dec, d):
    """
    Rotate and translate from arbitrary direction to Galactocentric frame.

    Args:
        ra (float): ra in radians.
        dec (float): dec in radians.
        d (float):
    """
    r = tt_r_icrs_norm(ra, dec)
    R1 = np.array([[np.cos(dec_gc), 0, np.sin(dec_gc)],
                   [0, 1, 0],
                   [-np.sin(dec_gc), 0, np.cos(dec_gc)]])
    R2 = np.array([[np.cos(ra_gc), np.sin(ra_gc), 0],
                   [-np.sin(ra_gc), np.cos(ra_gc), 0],
                   [0, 0, 1]])
    R3 = np.array([[1, 0, 0],
                   [0, np.cos(eta), np.sin(eta)],
                   [0, -np.sin(eta), np.cos(eta)]])
    R1_R2 = np.dot(R1, R2)
    R = np.dot(R3, R1_R2)

    xhat = np.array([[1, 0, 0]]).T
    theta = np.arcsin(zsun/d_gc)
    H = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

    rdash = d * np.dot(R, r) - d_gc * xhat

    rgc = tt.dot(H, rdash)
    return rgc


def tt_get_icrs_from_galactocentric(xyz, vxyz, ra, dec, distance, M):
    """
    Calculate proper motion & RV from Galactocentric velocities and positions.

    Args:
        xyz (array): array of x, y, z position. No astropy units.
        vxyz (array): array of v_x, v_y, v_z velocity. No astropy units.
        ra (flaot): ra in radians. No astropy units.
        dec (float): dec in radians. No astropy units.
        distance (float): distance in kpc. No astropy units.
    """
    dx = xyz - sun_xyz.value
    dv = vxyz - sun_vxyz.value

    # M is precomputed for each star, R_gal is same matrix, so just do this
    # dot product.
    proj_dv = tt.dot(M, tt.dot(R_gal, dv))

    # Calculate the unit conversion using 1kms/1km and transform.
    proper = cartesian_to_angular_no_units(proj_dv[:2], distance)

    rv = proj_dv[2]

    return proper, rv


def run_pymc3_model(pos, pos_err, proper, proper_err, mean, cov):

    M = get_tangent_basis(pos[0] * 2*np.pi/360, pos[1] * 2*np.pi/360)
    # mean, cov = get_prior()

    with pm.Model() as model:

        vxyzD = pm.MvNormal("vxyzD", mu=mean, cov=cov, shape=4)
        vxyz = pm.Deterministic("vxyz", vxyzD[:3])
        log_D = pm.Deterministic("log_D", vxyzD[3])
        D = pm.Deterministic("D", tt.exp(log_D))

        xyz = pm.Deterministic("xyz", tt_eqtogal(pos[0], pos[1], D)[:, 0])

        pm_from_v, rv_from_v = tt_get_icrs_from_galactocentric(xyz, vxyz,
                                                               pos[0], pos[1],
                                                               D, M)

        pm.Normal("proper_motion", mu=pm_from_v, sigma=np.array(proper_err),
                  observed=np.array(proper))
        pm.Normal("parallax", mu=1. / D, sigma=pos_err[2], observed=pos[2])

        map_soln = xo.optimize()
        trace = pm.sample(tune=1500, draws=1000, start=map_soln,
                          step=xo.get_dense_nuts_step(target_accept=0.9))

    return trace


# if __name__ == "__main__":
#     print(get_prior())
