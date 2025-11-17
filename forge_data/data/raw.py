""" """

import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
import h5py
import sqlite3

# from forge_data.micro_epsilon import Recon


def process_raw_dataset():
    """
    Process a dataset.
    """
    pass


def process_linescanner_file(file):
    """
    Load raw linescanner file & process into pointcloud and mesh.
    """

    file_version = get_linescanner_file_version(file)

    if file_version == "csv-0.1.0":
        df = parse_csv_0_1_0(file)


def get_linescanner_file_version(file):
    """
    Determine which version / format the csv is so we know what dataframe ops to do.
    """

    if file.suffix == ".csv":
        df = pd.read_csv(file, header=None)

    # Determine if 11/17/25 version
    if (
        df.iloc[0, 0] == "Part Temperature (C)"
        and df.iloc[0, 1] == "Time Unix (ms):"
        and df.iloc[1, 1] == "A Axis Angle (deg):"
        and df.iloc[0, 3] == "X values (mm):"
        and df.iloc[1, 3] == "Z values (mm):"
    ):
        file_version = "csv-0.1.0"
    else:
        file_version = None

    return file_version


def parse_csv_0_1_0(file):
    df = pd.read_csv(file, header=None)

    # Assemble into reasonable data format
    df_even = df.iloc[::2].reset_index(drop=True)
    df_odd = df.iloc[1::2].reset_index(drop=True)
    timestamps = df_even.iloc[:, 2]
    temps = df_odd.iloc[:, 0]
    a_axis_angles = df_odd.iloc[:, 2]
    x_values = df_even.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)
    z_values = df_odd.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)

    df = pd.DataFrame({
        "timestamps_ms": timestamps,
        "temperature_C": temps,
        "a_axis_deg": a_axis_angles,
        "x_mm": x_values,
        "z_mm": z_values,
    })

    return df
