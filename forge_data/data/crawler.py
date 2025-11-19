"""
TODO
"""

import datetime
import re
import sqlite3
from collections import defaultdict

import h5py

from forge_data.data.parser import process_linescanner_file, process_load_stroke_file, process_obj_file
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
    """ """
    global_files, hits_data = discover_and_group_files(path)

    for global_file in global_files:
        if global_file.suffix == ".obj":
            vertices, faces = process_obj_file(global_file)
            db_key = f"{global_file.parent.parent.name}/{global_file.parent.name}/{global_file.name}"
            h5_conn.create_dataset(f"{db_key}/vertices", data=vertices)
            h5_conn.create_dataset(f"{db_key}/faces", data=faces)
            sqlite_conn.execute(
                "INSERT OR REPLACE INTO mesh_files (path, vertices, faces) VALUES (?, ?, ?)",
                (db_key, vertices.tobytes(), faces.tobytes()),
            )

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
