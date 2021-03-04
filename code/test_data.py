import numpy as np
import matplotlib.pyplot as plt
import data as d
import pandas as pd


def test_load_and_merge_data():
    df = d.load_and_merge_data()
    print(np.shape(df))
    assert np.shape(df) == (194764, 328)
    return df

def test_combine_rv_measurements(df):
    rvdf = d.combine_rv_measurements(df)

    ml = np.isfinite(rvdf.stellar_rv.values)
    mg = np.isfinite(rvdf.radial_velocity.values) \
        & (df.radial_velocity.values != 0)
    ma = np.isfinite(rvdf.OBSVHELIO_AVG.values)
    mlmg = ml & mg
    mamg = ma & mg

    # Make sure the (corrected) LAMOST and Gaia RVs are consistent to within
    # 1 sigma.
    tot_err = np.sqrt(rvdf.radial_velocity_error.values[mlmg]**2
                      + df.stellar_rv_err.values[mlmg]**2)
    resids = df.radial_velocity.values[mlmg] \
        - df.lamost_corrected_rv.values[mlmg]
    print(np.std(resids/tot_err), np.mean(resids/tot_err))
    assert np.isclose(np.std(resids/tot_err), 1, atol=.1)
    assert np.isclose(np.mean(resids/tot_err), 0, atol=.1)

    tot_err = np.sqrt(df.radial_velocity_error.values[mamg]**2
                      + df.OBSVERR.values[mamg]**2)
    ro = abs(df.OBSVHELIO_AVG.values[mamg]
             - df.radial_velocity.values[mamg]) < 10 * tot_err
    resids2 = df.radial_velocity.values[mamg][ro] \
        - df.OBSVHELIO_AVG.values[mamg][ro]
    resids = df.radial_velocity.values[mamg][ro] \
        - df.apogee_corrected_rv.values[mamg][ro]
    print(np.std(resids/tot_err[ro]), np.mean(resids/tot_err[ro]))
    assert np.isclose(np.std(resids/tot_err[ro]), 1, atol=1)
    assert np.isclose(np.mean(resids/tot_err[ro]), 0, atol=.1)

    return rvdf

if __name__ == "__main__":
    df = test_load_and_merge_data()
    rvdf = test_combine_rv_measurements(df)
