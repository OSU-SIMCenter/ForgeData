"""
Parses through a directory and generates an H5 / sqlite3 database.
"""

import argparse
import os
from pathlib import Path

from forge_data.data.raw import process_linescanner_file, process_TP_directory
from forge_data.ue.api import mesh_from_dataframe

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent
default_raw_path = project_root / "data" / "raw"
default_save_path = project_root / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=Path, default=default_raw_path)
    parser.add_argument("--save-path", type=Path, default=default_save_path)
    parser.add_argument("--format", type=str, choices=["h5", "sqlite"], default="h5")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # TODO: rm
    tp_path = Path(r"C:\Users\colto\Github\ForgeData\data\raw\2025_10_28 T18_01_24.E8 Tensile Bar 4140\TP1")
    process_TP_directory(tp_path)

    linescan_path = Path(
        r"C:\Users\colto\Github\ForgeData\data\raw\2025_10_28 T18_01_24.E8 Tensile Bar 4140\TP1\3D Scan Data\2025_10_28 T18_09_31.3D Scan at Hitpoint 31.csv"
    )
    df = process_linescanner_file(linescan_path)
    obj = mesh_from_dataframe([df])


if __name__ == "__main__":
    main()
