"""
TODO
"""

import datetime
import re
from collections import defaultdict

import h5py

from forge_data.data.parser import process_linescanner_file, process_load_stroke_file, process_obj_file
from forge_data.ue.api import mesh_from_dataframe

HIT_NUM_REGEX = re.compile(r"Hitpoint (\d+)")
TP_REGEX = re.compile(r"TP(\d+)")
TEMP_FILE_REGEX = re.compile(r".*t.*\.h5$", re.IGNORECASE)


def process_raw_directory(raw_path, save_path):
    """
    Process a dataset.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = save_path / f"{timestamp}.h5"

    h5_conn = h5py.File(h5_path, "a")

    for item in raw_path.iterdir():
        if item.is_dir():
            if TP_REGEX.search(item.name):
                process_TP_directory(item, h5_conn)
            else:
                # Check one level deeper for TP directories
                for sub_item in item.iterdir():
                    if sub_item.is_dir() and TP_REGEX.search(sub_item.name):
                        process_TP_directory(sub_item, h5_conn)

                    elif TEMP_FILE_REGEX.search(sub_item.name):
                        # print(f"Found temperature file: {sub_item.name}")
                        T_data_key = str(sub_item.relative_to(raw_path)).replace(".h5", "")
                        try:
                            with h5py.File(sub_item, "r") as source_h5:
                                h5_conn.copy(source_h5["/"], T_data_key)
                        except OSError:
                            print(f"Could not open or read {sub_item}")

    h5_conn.close()
    rebuild_h5_global_keys(h5_path)


def process_TP_directory(path, h5_conn):
    """ """
    global_files, hits_data = discover_and_group_files(path)

    for global_file in global_files:
        if global_file.suffix == ".obj":
            vertices, faces = process_obj_file(global_file)
            # TODO Change db_key to use Pathlib.relative_to()
            db_key = f"{global_file.parent.parent.name}/{global_file.parent.name}/{global_file.name}"
            h5_conn.create_dataset(f"{db_key}/vertices", data=vertices)
            h5_conn.create_dataset(f"{db_key}/faces", data=faces)

    for hit_num, hit_files in hits_data.items():
        db_keybase = f"{path.parent.name}/{path.name}/H{hit_num:04}"

        if "scan" in hit_files:
            scan_file = hit_files["scan"]
            df = process_linescanner_file(scan_file)
            vertices, faces = mesh_from_dataframe(df)
            db_key = db_keybase + "/reconstructed_mesh"
            h5_conn.create_dataset(f"{db_key}/vertices", data=vertices)
            h5_conn.create_dataset(f"{db_key}/faces", data=faces)

        if "load_stroke" in hit_files:
            ls_file = hit_files["load_stroke"]
            df = process_load_stroke_file(ls_file)
            db_key = db_keybase + "/load_stroke"
            h5_conn.create_dataset(db_key, data=df.to_records(index=False))

    h5_conn.flush()


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


def rebuild_h5_global_keys(h5_path):
    """
    Run through the h5 file and build "pointers" / global indices for pytorch dataloader such that we can easily call
    dataloader_object[100] and it pulls the correct load/stroke, thermal frames, and meshes
    """

    hit_pattern = re.compile(r"^H\d{4}$")
    found_paths = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Group):
            folder_name = name.split("/")[-1]
            if hit_pattern.match(folder_name):
                found_paths.append(name)

    with h5py.File(str(h5_path), "a") as f:
        f.visititems(visitor_func)
        count = len(found_paths)

        if "global_keyset" in f:
            del f["global_keyset"]

        dtype = h5py.special_dtype(vlen=str)
        dset = f.create_dataset("global_keyset", shape=(count,), maxshape=(None,), dtype=dtype)
        dset[:] = sorted(found_paths)

    print(found_paths)
    print(count)
