import numpy as np
from pyia import GaiaData
import pandas as pd
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u

from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors

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


def get_solar_and_R_gal():
    # Set up Solar position and motion.
    sun_xyz = [-8.122, 0, 0] * u.kpc
    sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s
    sun_vxyzCD = coord.CartesianDifferential(12.9, 245.6, 7.78, u.km/u.s)

    galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                        galcen_v_sun=sun_vxyzCD,
                                        z_sun=sun_xyz[2].value*1e3*u.pc)

    # Rotation matrix from Galactocentric to ICRS
    R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)
    return sun_xyz, sun_vxyz, R_gal, galcen_frame
sun_xyz, sun_vxyz, R_gal, galcen_frame = get_solar_and_R_gal()


def calc_vxyz(pandas_df, nsamples=1000):
    """
    Calculate vx, vy, vz samples from a pandas DataFrame.

    Args:
        df (DataFrame): pandas dataframe containing Gaia columns

    Returns:
        W (array): samples of W velocity.
            Shape = nstars x nsamples

    """

    df = Table.from_pandas(pandas_df)
    g = GaiaData(df)
    g_samples = g.get_error_samples(size=nsamples,
                                    rnd=np.random.RandomState(seed=42))
    c_samples = g_samples.get_skycoord()
    vels = c_samples.transform_to(coord.Galactocentric)
    vx = vels.v_x.value
    vy = vels.v_y.value
    vz = vels.v_z.value
    return np.mean(vx, axis=1), np.std(vx, axis=1), \
           np.mean(vy, axis=1), np.std(vy, axis=1), \
           np.mean(vz, axis=1), np.std(vz, axis=1)


def simple_calc_vxyz(ra, dec, D, pmra, pmdec, rv):
    """
    Calculate vx, vy, vz using astropy.

    Args:
        ra (array): RA in degrees.
        dec (array): dec in degrees.
        D (array): distance in kpc.
        pmra (array): proper motion (RA) in mas/yr.
        pmdec (array): proper motion (dec) in mas/yr.
        rv (array): radial velocity in km/s.

    Returns:
        galcen.data.xyz: position vector.
        galcen.velocity.vxyz: velocity vector.

    """
    c = coord.SkyCoord(ra=ra*u.deg,
                       dec=dec*u.deg,
                       distance=D*u.kpc,
                       pm_ra_cosdec=pmra*u.mas/u.yr,
                       pm_dec=pmdec*u.mas/u.yr,
                       radial_velocity=rv*u.km/u.s)

    galcen = c.transform_to(galcen_frame)
    return galcen.data.xyz, galcen.velocity.d_xyz
