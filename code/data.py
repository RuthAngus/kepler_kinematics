# Converting Exploring data into a script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.coordinates as coord

import kepler_kinematics as kek
from tools import getDust
from photometric_teff import bprp_to_teff
from dustmaps.bayestar import BayestarQuery


def load_and_merge_data():
    # Load Gaia-Kepler crossmatch.
    with fits.open("../data/kepler_dr2_1arcsec.fits") as data:
        gaia = pd.DataFrame(data[1].data, dtype="float64")
    m = gaia.parallax.values > 0
    gaia = gaia.iloc[m]

    # Round RVs down to 6dp.
    gaia["ra_6dp"] = np.round(gaia.ra.values, 6)
    gaia["dec_6dp"] = np.round(gaia.dec.values, 6)

    # Add LAMOST
    # File created using the LAMOST DR5 website: http://dr5.lamost.org/search
    lamost = pd.read_csv("../data/gaia-kepler-lamost_snr.csv")
    print(len(lamost), "lamost stars")

    # Remove one star with a giant LAMOST RV errorbar
    m = abs(lamost.stellar_rv_err.values) < 100
    lamost = lamost.iloc[m]

    # Merge Gaia and LAMOST on (rounded) RA and dec
    lamost["ra_6dp"] = lamost.inputobjs_input_ra.values
    lamost["dec_6dp"] = lamost.inputobjs_input_dec.values
    lamost_gaia = pd.merge(gaia, lamost, on=["ra_6dp", "dec_6dp"],
                           how="left", suffixes=["", "_lamost"])
    lamost_gaia = lamost_gaia.drop_duplicates(subset="source_id")

    return lamost_gaia


def combine_rv_measurements(df):
    """LAMOST RVs are overwritten by Gaia RVs
    """

    rv, rv_err = [np.ones(len(df))*np.nan for i in range(2)]

    ml = np.isfinite(df.stellar_rv.values)
    rv[ml] = df.stellar_rv.values[ml]
    rv_err[ml] = df.stellar_rv_err.values[ml]
    print(sum(ml), "stars with LAMOST RVs")

    mg = (df.radial_velocity.values != 0)
    mg &= np.isfinite(df.radial_velocity.values)
    rv[mg] = df.radial_velocity.values[mg]
    rv_err[mg] = df.radial_velocity_error.values[mg]
    print(sum(mg), "stars with Gaia RVs")

    df["rv"] = rv
    df["rv_err"] = rv_err
    return df


# S/N cuts
def sn_cuts(df):
    sn = df.parallax.values/df.parallax_error.values

    m = (sn > 10)
    m &= (df.parallax.values > 0) * np.isfinite(df.parallax.values)
    m &= df.astrometric_excess_noise.values < 5
    print(len(df.iloc[m]), "stars after S/N cuts")

    # Jason's wide binary cuts
    # m &= df.astrometric_excess_noise.values > 0
    # m &= df.astrometric_excess_noise_sig.values > 6

    # Jason's short-period binary cuts
    # m &= radial_velocity_error < 4
    # print(len(df.iloc[m]), "stars after Jason's binary cuts")
    # assert 0

    df = df.iloc[m]
    return df


def add_velocities(df):
    xyz, vxyz = kek.simple_calc_vxyz(df.ra.values, df.dec.values,
                                    1./df.parallax.values, df.pmra.values,
                                    df.pmdec.values,
                                    df.rv.values)
    vx, vy, vz = vxyz
    x, y, z = xyz

    df["vx"] = vxyz[0].value
    df["vy"] = vxyz[1].value
    df["vz"] = vxyz[2].value
    df["x"] = xyz[0].value
    df["y"] = xyz[1].value
    df["z"] = xyz[2].value
    return df


# Calculate Absolute magntitude
def mM(m, D):
    return 5 - 5*np.log10(D) + m


def deredden(df):
    print("Loading Dustmaps")
    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')

    print("Calculating Ebv")
    coords = SkyCoord(df.ra.values*u.deg, df.dec.values*u.deg,
                    distance=df.r_est.values*u.pc)

    ebv, flags = bayestar(coords, mode='percentile', pct=[16., 50., 84.],
                        return_flags=True)

    # Calculate Av
    Av_bayestar = 2.742 * ebv
    print(np.shape(Av_bayestar), "shape")
    Av = Av_bayestar[:, 1]
    Av_errm = Av - Av_bayestar[:, 0]
    Av_errp = Av_bayestar[:, 2] - Av
    Av_std = .5*(Av_errm + Av_errp)

    # Catch places where the extinction uncertainty is zero and default to an
    # uncertainty of .05
    m = Av_std == 0
    Av_std[m] = .05

    df["ebv"] = ebv[:, 1]  # The median ebv value.
    df["Av"] = Av
    df["Av_errp"] = Av_errp
    df["Av_errm"] = Av_errm
    df["Av_std"] = Av_std

    # Calculate dereddened photometry
    AG, Abp, Arp = getDust(df.phot_g_mean_mag.values,
                        df.phot_bp_mean_mag.values,
                        df.phot_rp_mean_mag.values, df.ebv.values)

    df["bp_dered"] = df.phot_bp_mean_mag.values - Abp
    df["rp_dered"] = df.phot_rp_mean_mag.values - Arp
    df["bprp_dered"] = df["bp_dered"] - df["rp_dered"]
    df["G_dered"] = df.phot_g_mean_mag.values - AG

    abs_G = mM(df.G_dered.values, df.r_est)
    df["abs_G"] = abs_G

    return df


def add_phot_teff(df):
    # Calculate photometric Teff
    teffs = bprp_to_teff(df.bp_dered - df.rp_dered)
    df["color_teffs"] = teffs
    return df


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_merge_data()
    print(len(df), "stars")

    print("Combining RV measurements...")
    df = combine_rv_measurements(df)
    print(len(df), "stars")

    print("S/N cuts")
    df = sn_cuts(df)
    print(len(df), "stars")

    print("Calculating velocities")
    df = add_velocities(df)
    print(len(df), "stars")

    print("Get dust and redenning...")
    df = deredden(df)
    print(len(df), "stars")

    print("Calculate photometric temperatures.")
    df = add_phot_teff(df)
    print(len(df), "stars")

    df = df.drop_duplicates(subset="source_id")
    grv = (np.isfinite(df.radial_velocity.values)) \
        & (df.radial_velocity.values != 0)
    print(sum(grv), "stars with Gaia RVs after cuts")
    lrv = np.isfinite(df.stellar_rv.values)
    print(sum(lrv), "stars with LAMOST RVs after cuts")
    mrv = np.isfinite(df.rv.values)
    print(sum(mrv), "stars with either rv after cuts")

    print("Saving file")
    fname = "../kepler_kinematics/gaia_kepler_lamost.csv"
    print(fname)
    df.to_csv(fname)
