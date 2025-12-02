"""
Test key builder for h5
"""

import argparse
from pathlib import Path

from forge_data.data.crawler import rebuild_h5_global_keys

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-path",
        type=Path,
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    rebuild_h5_global_keys(h5_path)


if __name__ == "__main__":
    main()
