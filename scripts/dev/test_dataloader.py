"""
Test key builder for h5
"""

import argparse
from pathlib import Path

from forge_data.torch_dataset.forge_dataset import ForgeDataset

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
    ds = ForgeDataset(h5_path)

    print(len(ds))
    ds.print_h5_structure()

    x0 = ds[3]
    print(x0)


if __name__ == "__main__":
    main()
