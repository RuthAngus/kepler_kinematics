import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np

from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors


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


def get_icrs_from_galactocentric(xyz, vxyz, R_gal, sun_xyz, sun_vxyz):

    # Parameters
    dx = xyz - sun_xyz
    dv = vxyz - sun_vxyz

    # Don't need this as already have ra and dec.
    x_icrs = coord.ICRS(
        coord.CartesianRepresentation(R_gal @ dx))

    M = get_tangent_basis(x_icrs.ra, x_icrs.dec)

    # M is precomputed for each star, R_gal is same matrix, so just do this
    # dot product.
    proj_dv = M @ R_gal @ dv

    # Calculate the unit conversion using 1kms/1km and transform.
    pm = (proj_dv[:2] / x_icrs.distance).to(u.mas/u.yr,
                                            u.dimensionless_angles())

    rv = proj_dv[2]

    return pm, rv


def cartesian_to_angular_no_units(km_s, kpc):
    to_km_yr = 365.2425 * 24*3600
    to_m_yr = 1000
    to_kpc_yr = 1./3.0856775814671917e+19
    to_rad_yr = 1./kpc
    to_deg_yr = 360/(2*np.pi)
    to_mas_yr = 3600*1000
    return np.arcsin((km_s * to_km_yr) * to_m_yr * to_kpc_yr * to_rad_yr) * to_deg_yr * to_mas_yr
