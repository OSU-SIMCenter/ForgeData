"""
TODO
"""

from collections import defaultdict
import pandas as pd
import re

HIT_NUM_REGEX = re.compile(r"Hitpoint (\d+)")


def process_raw_directory():
    """
    Process a dataset.
    """
    pass


def process_TP_directory(path):

    global_files, hits_data = discover_and_group_files(path)

    for global_file in global_files:
        if global_file.suffix == ".obj":
            process_obj_file(global_file)


def discover_and_group_files(path):
    global_files = []
    hits_data = defaultdict(dict)

    for item in path.rglob("*"):
        if not item.is_file():
            continue

        if item.suffix == ".txt":
            continue

        if item.parent == path:
            global_files.append(item)

        else:
            match = HIT_NUM_REGEX.search(item.name)
            if match:
                hit_num = int(match.group(1))
                parent_dir_name = item.parent.name

                if parent_dir_name == "3D Scan Data" and item.suffix == ".csv":
                    hits_data[hit_num]["scan"] = item
                elif parent_dir_name == "Load Stroke Data" and item.suffix == ".csv":
                    hits_data[hit_num]["load_stroke"] = item

    return global_files, hits_data


def process_linescanner_file(file):
    """
    Load raw linescanner file & process into pandas dataframe.
    """

    file_version = get_linescanner_file_version(file)

    if file_version == "csv-0.1.0":
        df = parse_csv_0_1_0(file)

    return df


def process_obj_file(file):
    """
    TODO
    """
    pass


def get_folder_structure_version(path):
    """
    This seems to change often, determine the layout of raw files for parsing.
    """
    pass


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
    x_values = df_even.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)
    z_values = df_odd.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1)
    a_axis_angles = df_odd.iloc[:, 2]

    df = pd.DataFrame({
        "timestamps_ms": timestamps,
        "temperature_C": temps,
        "x_mm": x_values,
        "z_mm": z_values,
        "a_axis_deg": a_axis_angles,
    })

    return df
