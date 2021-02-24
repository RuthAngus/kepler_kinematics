import numpy as np
import matplotlib.pyplot as plt
import data as d
import pandas as pd


def test_load_and_merge_data():
    df = d.load_and_merge_data()
    return df

def test_combine_rv_measurements(df):
    rvdf = d.combine_rv_measurements(df)

    plt.plot(rvdf.OBSVHELIO_AVG.values, rvdf.radial_velocity, ".")
    plt.plot(df.OBSVHELIO_AVG.values, rvdf.rv, ".")
    plt.savefig("test")
    plt.close()

    return rvdf

if __name__ == "__main__":
    df = test_load_and_merge_data()
    rvdf = test_combine_rv_measurements(df)

    si = [f"{i:.0f}" for i in rvdf.source_id.values]
    names = pd.DataFrame(dict({"source_id": si}))
    names.to_csv("source_ids.csv", index=False)
