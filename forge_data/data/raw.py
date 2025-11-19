"""
TODO
"""

import datetime
import re
import sqlite3
from collections import defaultdict

import h5py
import numpy as np
import open3d as o3d
import pandas as pd

from forge_data.ue.api import mesh_from_dataframe

HIT_NUM_REGEX = re.compile(r"Hitpoint (\d+)")
TP_REGEX = re.compile(r"TP(\d+)")


def process_raw_directory(raw_path, save_path):
    """
    Process a dataset.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = save_path / f"{timestamp}.h5"
    sqlite_path = save_path / f"{timestamp}.sqlite"

    h5_conn = h5py.File(h5_path, "a")
    sqlite_conn = sqlite3.connect(sqlite_path)

    cursor = sqlite_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS obj_files (
            path TEXT PRIMARY KEY,
            data BLOB
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mesh_files (
            path TEXT PRIMARY KEY,
            vertices BLOB,
            faces BLOB
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS load_stroke_data (
            path TEXT PRIMARY KEY,
            data BLOB
        )
    """)

    sqlite_conn.commit()

    for item in raw_path.iterdir():
        if item.is_dir():
            if TP_REGEX.search(item.name):
                process_TP_directory(item, h5_conn, sqlite_conn)
            else:
                # Check one level deeper for TP directories
                for sub_item in item.iterdir():
                    if sub_item.is_dir() and TP_REGEX.search(sub_item.name):
                        process_TP_directory(sub_item, h5_conn, sqlite_conn)

    h5_conn.close()
    sqlite_conn.close()


def process_TP_directory(path, h5_conn, sqlite_conn):

    global_files, hits_data = discover_and_group_files(path)

    for global_file in global_files:
        if global_file.suffix == ".obj":
            process_obj_file(global_file, h5_conn, sqlite_conn, path)

    for hit_num, hit_files in hits_data.items():
        db_keybase = f"{path.parent.name}/{path.name}/H{hit_num:04}"

        if "scan" in hit_files:
            scan_file = hit_files["scan"]
            df = process_linescanner_file(scan_file)
            vertices, faces = mesh_from_dataframe([df])
            db_key = db_keybase + "/reconstructed_mesh"
            h5_conn.create_dataset(f"{db_key}/vertices", data=vertices)
            h5_conn.create_dataset(f"{db_key}/faces", data=faces)
            sqlite_conn.execute(
                "INSERT OR REPLACE INTO mesh_files (path, vertices, faces) VALUES (?, ?, ?)",
                (db_key, vertices.tobytes(), faces.tobytes()),
            )

        if "load_stroke" in hit_files:
            ls_file = hit_files["load_stroke"]
            df = process_load_stroke_file(ls_file)
            db_key = db_keybase + "/load_stroke"
            h5_conn.create_dataset(db_key, data=df.to_records(index=False))
            sqlite_conn.execute(
                "INSERT OR REPLACE INTO load_stroke_data (path, data) VALUES (?, ?)",
                (db_key, df.to_json().encode("utf-8")),
            )

    h5_conn.flush()
    sqlite_conn.commit()


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

    if file_version == "ue-csv-0.1.0":
        df = parse_ue_csv_0_1_0(file)

    return df


def process_load_stroke_file(file):
    """
    TODO
    """

    file_version = get_load_stroke_file_version(file)

    if file_version == "LS-csv-0.1.0":
        df = parse_ls_csv_0_1_0(file)

    return df


def process_obj_file(file, h5_conn, sqlite_conn, tp_path):
    """
    Process an OBJ file and save its mesh data to the databases.
    """
    mesh = o3d.io.read_triangle_mesh(str(file))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if faces is None:
        print(f"Warning: No triangular faces found in {file}. Skipping.")
        return

    db_key = f"{tp_path.parent.name}/{tp_path.name}/{file.name}"

    # Write to HDF5
    h5_conn.create_dataset(f"{db_key}/vertices", data=vertices)
    h5_conn.create_dataset(f"{db_key}/faces", data=faces)
    sqlite_conn.execute(
        "INSERT OR REPLACE INTO mesh_files (path, vertices, faces) VALUES (?, ?, ?)",
        (db_key, vertices.tobytes(), faces.tobytes()),
    )


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
        file_version = "ue-csv-0.1.0"
    else:
        file_version = None

    return file_version


def get_load_stroke_file_version(file):
    """
    TODO
    """

    if file.suffix == ".csv":
        df = pd.read_csv(file, header=None)

    # Determine if 11/17/25 version
    if (
        df.iloc[0, 0] == "Time Unix (ms)"
        and df.iloc[0, 1] == "Position (mm)"
        and df.iloc[0, 2] == "Force (kN)"
        and df.iloc[0, 3] == "Live Velocity (mm/s)"
        and df.iloc[0, 4] == "Thermal Cam Temp (c)"
        and df.iloc[0, 5] == "Pyrometer Temp (c)"
        and df.iloc[0, 6] == "Press Target Position (mm)"
        and df.iloc[0, 7] == "Press Target Velocity (mm/s)"
        and df.iloc[0, 8] == "X"
        and df.iloc[0, 9] == "Y"
        and df.iloc[0, 10] == "Z"
        and df.iloc[0, 11] == "A"
        and df.iloc[0, 12] == "Ram Side Tool Number"
        and df.iloc[0, 13] == "Static Side Tool Number"
        and df.iloc[0, 14] == "Hit Number"
    ):
        file_version = "LS-csv-0.1.0"
    else:
        file_version = None

    return file_version


def parse_ue_csv_0_1_0(file):
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


def parse_ls_csv_0_1_0(file):
    """
    TODO
    """
    # The first row is the header
    df = pd.read_csv(file, header=0)
    return df
