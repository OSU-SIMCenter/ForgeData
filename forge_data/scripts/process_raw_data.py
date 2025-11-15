"""
Parses through a directory and generates an H5 / sqlite3 database.
"""

import argparse
import os
from pathlib import Path

from forge_data.data.raw import process_raw_dataset

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent
default_raw_path = project_root / "data" / "raw"
default_save_path = project_root / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=Path, default=default_raw_path)
    parser.add_argument("--save-path", type=Path, default=default_save_path)
    parser.add_argument("--format", type=str, choices=["h5", "sqlite"], default="h5")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)


if __name__ == "__main__":
    main()
