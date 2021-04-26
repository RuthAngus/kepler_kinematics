# Converting Exploring data into a script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

import kepler_kinematics as kek
from tools import getDust
from photometric_teff import bprp_to_teff
from dustmaps.bayestar import BayestarQuery

# Set defaults so they don't change
import astropy.coordinates as coord
coord.galactocentric_frame_defaults.set('v4.0')


# def load_and_merge_data():
#     # Load Gaia-Kepler crossmatch.
#     with fits.open("../data/kepler_dr2_1arcsec.fits") as data:
#         gaia = pd.DataFrame(data[1].data, dtype="float64")
#     m = gaia.parallax.values > 0
#     gaia = gaia.iloc[m]

#     # Round RVs down to 6dp.
#     gaia["ra_6dp"] = np.round(gaia.ra.values, 6)
#     gaia["dec_6dp"] = np.round(gaia.dec.values, 6)

#     # Add LAMOST
#     # File created using the LAMOST DR5 website: http://dr5.lamost.org/search
#     lamost = pd.read_csv("../data/gaia-kepler-lamost_snr.csv")

#     # Remove one star with a giant LAMOST RV errorbar
#     m = abs(lamost.stellar_rv_err.values) < 100
#     lamost = lamost.iloc[m]

#     # Merge Gaia and LAMOST on (rounded) RA and dec
#     lamost["ra_6dp"] = lamost.inputobjs_input_ra.values
#     lamost["dec_6dp"] = lamost.inputobjs_input_dec.values
#     lamost_gaia = pd.merge(gaia, lamost, on=["ra_6dp", "dec_6dp"],
#                            how="left", suffixes=["", "_lamost"])
#     lamost_gaia = lamost_gaia.drop_duplicates(subset="source_id")

#     # Load apogee
#     tbl = Table.read("../data/apogeedr16_stars.fits", format='fits')
#     names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
#     apo = tbl[names].to_pandas()

#     apodf = pd.merge(apo, lamost_gaia, how="right", left_on="GAIA_SOURCE_ID",
#                      right_on="source_id", suffixes=["_apogee", ""])
#     apodf = apodf.drop_duplicates(subset="source_id")
#     return apodf


def load_and_merge_data(edr3=True):

    # Load Gaia-Kepler crossmatch
    print("Loading Gaia-Kepler crossmatch...")
    with fits.open("../data/kepler_dr2_1arcsec.fits") as data:
        gaia = pd.DataFrame(data[1].data, dtype="float64")
    m = gaia.parallax.values > 0
    gaia = gaia.iloc[m]

    # Round RVs down to 6dp.
    gaia["ra_6dp"] = np.round(gaia.ra.values, 6)
    gaia["dec_6dp"] = np.round(gaia.dec.values, 6)

    # Add LAMOST
    # File created using the LAMOST DR5 website: http://dr5.lamost.org/search
    print("Loading Lamost")
    lamost = pd.read_csv("../data/gaia-kepler-lamost_snr.csv")

    # Remove one star with a giant LAMOST RV errorbar
    m = abs(lamost.stellar_rv_err.values) < 100
    lamost = lamost.iloc[m]

    # Merge Gaia and LAMOST on (rounded) RA and dec
    lamost["ra_6dp"] = lamost.inputobjs_input_ra.values
    lamost["dec_6dp"] = lamost.inputobjs_input_dec.values

    lamost_stripped = pd.DataFrame(dict({
        "ra_6dp": lamost.inputobjs_input_ra.values,
        "dec_6dp": lamost.inputobjs_input_dec.values,
        "stellar_rv": lamost.stellar_rv.values,
        "stellar_rv_err": lamost.stellar_rv_err.values}))

    # If you want edr3, just match on kepid then match again with edr3.
    if edr3:
        print("Merging LAMOST and Gaia DR3")
        gaia_stripped = pd.DataFrame(dict({"ra_6dp": gaia["ra_6dp"],
                                           "dec_6dp": gaia["dec_6dp"],
                                           "kepid": gaia["kepid"],
                                           "r_est": gaia["r_est"],
                                           "r_lo": gaia["r_lo"],
                                           "r_hi": gaia["r_hi"]}))
        lamost_gaia = pd.merge(gaia_stripped, lamost_stripped,
                               on=["ra_6dp", "dec_6dp"],
                               how="left")

        with fits.open("../../data/kepler_edr3_1arcsec.fits") as data:
            gaia3 = pd.DataFrame(data[1].data, dtype="float64")
        m = gaia3.parallax.values > 0
        gaia3 = gaia3.iloc[m]

        # Merge DR2 and 3
        gaia23 = pd.merge(gaia_stripped, gaia3, on="kepid")

        # Merge DR3 and lamost
        lamost_gaia = pd.merge(gaia23, lamost, on=["ra_6dp", "dec_6dp"],
                               how="left", suffixes=["", "_lamost"])
        lamost_gaia["radial_velocity"] = lamost_gaia["dr2_radial_velocity"]
        lamost_gaia["radial_velocity_error"] = lamost_gaia[
            "dr2_radial_velocity_error"]

    else:
        print("Merging LAMOST and Gaia DR2")
        lamost_gaia = pd.merge(gaia, lamost_stripped,
                               on=["ra_6dp", "dec_6dp"],
                               how="left", suffixes=["", "_lamost"])

    lamost_gaia = lamost_gaia.drop_duplicates(subset="source_id")

    # Load apogee
    print("Loading APOGEE")
    if edr3:

        tbl = Table.read("../data/apogee_dr16_tmass_edr3_xmatch.fits",
                         format='fits')
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        apo3 = tbl[names].to_pandas()

        tbl = Table.read("../data/apogeedr16_stars.fits", format='fits')
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        apo2 = tbl[names].to_pandas()

        # crossmatch on dr2 source id.
        apo23 = pd.merge(apo2, apo3, how="left", left_on="GAIA_SOURCE_ID",
                         right_on="dr2_source_id")
        apo23 = apo23.drop_duplicates(subset="dr3_source_id")

        # Only keep what you need from apogee
        apo_stripped = pd.DataFrame(dict({
            "dr3_source_id": apo23.dr3_source_id.values,
            "OBSVHELIO_AVG": apo23.OBSVHELIO_AVG.values,
            "OBSVERR": apo23.OBSVERR.values}))

        apodf = pd.merge(apo_stripped, lamost_gaia, how="right",
                         left_on="dr3_source_id", right_on="source_id",
                         suffixes=["_apogee", ""])
        print(np.shape(lamost_gaia), np.shape(apodf))

    else:
        tbl = Table.read("../data/apogeedr16_stars.fits", format='fits')
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        apo = tbl[names].to_pandas()

        print("Merging Gaia and Apogee")
        apodf = pd.merge(apo, lamost_gaia, how="right",
                         left_on="GAIA_SOURCE_ID",
                         right_on="source_id", suffixes=["_apogee", ""])

    apodf = apodf.drop_duplicates(subset="source_id")
    return apodf


def fit_line(x, y, yerr):
    AT = np.vstack((np.ones(len(x)), x))
    C = np.eye(len(x))*yerr
    CA = np.linalg.solve(C, AT.T)
    Cy = np.linalg.solve(C, y)
    ATCA = np.dot(AT, CA)
    ATCy = np.dot(AT, Cy)
    w = np.linalg.solve(ATCA, ATCy)

    cov = np.linalg.inv(ATCA)
    sig = np.sqrt(np.diag(cov))
    return w, sig


def combine_rv_measurements(df):
    """
    Combine RVs from LAMOST, Gaia, and APOGEE into one column,

    LAMOST RVs are overwritten by Gaia RVs, which are overwritten by APOGEE
    RVS.
    LAMOST RVs are called 'stellar_rv, stellar_rv_err',
    Gaia are called 'radial_velocity, radial_velocity_error',
    APOGEE are called 'OBSVHELIO_AVG, OBSVERR'.
    """

    rv, rv_err = [np.ones(len(df))*np.nan for i in range(2)]

    ml = np.isfinite(df.stellar_rv.values)
    mg = np.isfinite(df.radial_velocity.values) & (df.radial_velocity.values != 0)
    ma = np.isfinite(df.OBSVHELIO_AVG.values)
    mlmgma = ml & mg & ma
    mlmg = ml & mg
    mamg = ma & mg

    # Correct LAMOST RVs
    x, y = df.radial_velocity.values[mlmg], df.stellar_rv.values[mlmg]-df.radial_velocity.values[mlmg]
    yerr = df.stellar_rv_err.values[mlmg]
    w, sig = fit_line(x, y, yerr)
    lamost_corrected = df.stellar_rv.values - (w[0] + w[1]*df.stellar_rv.values)
    df["lamost_corrected_rv"] = lamost_corrected

    rv[ml] = lamost_corrected[ml]
    rv_err[ml] = df.stellar_rv_err.values[ml]
    print(sum(ml), "stars with LAMOST RVs")

    # Overwrite LAMOST RVs with Gaia RVs
    rv[mg] = df.radial_velocity.values[mg]
    rv_err[mg] = df.radial_velocity_error.values[mg]
    print(sum(mg), "stars with Gaia RVs")

    # Correct APOGEE RVs
    # remove outliers
    tot_err = np.sqrt(df.radial_velocity_error.values[mamg]**2 + df.OBSVERR.values[mamg]**2)
    ro = abs(df.OBSVHELIO_AVG.values[mamg] - df.radial_velocity.values[mamg]) < 3 * tot_err

    x, y = df.radial_velocity.values[mamg], df.OBSVHELIO_AVG.values[mamg]-df.radial_velocity.values[mamg]
    yerr = df.OBSVERR.values[mamg]
    w, sig = fit_line(x, y, yerr)
    apogee_corrected = df.OBSVHELIO_AVG.values - (w[0] + w[1]*df.OBSVHELIO_AVG.values)
    df["apogee_corrected_rv"] = apogee_corrected

    # Overwrite Gaia RVs with APOGEE RVs
    rv[ma] = apogee_corrected[ma]
    rv_err[ma] = df.OBSVERR.values[ma]
    print(sum(ma), "stars with APOGEE RVs")

    print(sum(np.isfinite(rv)), "stars with RVs")
    print(sum(mlmg), "with LAMOST and Gaia")
    print(sum(mlmgma), "with all three")

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

    ml = np.isfinite(df.stellar_rv.values)
    print(sum(ml), "stars with LAMOST RVs after cuts")
    mg = (df.radial_velocity.values != 0)
    mg &= np.isfinite(df.radial_velocity.values)
    print(sum(mg), "stars with Gaia RVs after cuts")
    ma = np.isfinite(df.OBSVHELIO_AVG.values)
    print(sum(ma), "stars with APOGEE RVs after cuts")

    print(sum(np.isfinite(df.rv)), "stars with RVs after cuts in total")

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
                                    df.r_est.values*1e-3, df.pmra.values,
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

    print("Saving file")
    fname = "../kepler_kinematics/gaia_kepler_lamost.csv"
    print(fname)

    # Randomly shuffle the data file so it is not in order of
    # ascending Kepler id
    np.random.seed(42)
    df = df.sample(frac=1)

    df.to_csv(fname)
