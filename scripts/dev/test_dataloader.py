"""
Test key builder for h5
"""

import argparse
from pathlib import Path

from forge_data.torch_dataset.forge_dataset import ForgeDataset

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-path",
        type=Path,
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    ds = ForgeDataset(h5_path)

    print(f"There are {len(ds)} data points in the database.")
    # ds.print_h5_structure()

    # example_datapoint = ds[99]
    # print(x0)

    # ds.plot_state_action(10)
    # Show meshes for every action in the database
    # for i in range(len(ds)):
    #     print(i)
    # ds.plot_state_action(10)

    ds.plot_load_stroke(0)
    ds.plot_thermal_frame(0)
    print(f"T_max of ds[0]: {ds[0].T_max}")
    print(f"T_avg of ds[0]: {ds[0].T_avg}")
    ds.plot_load_stroke(1)
    ds.plot_thermal_frame(1)
    print(f"T_max of ds[1]: {ds[1].T_max}")
    print(f"T_avg of ds[1]: {ds[1].T_avg}")

    # Show a thermal snapshot for every action in the database
    # for i in range(len(ds)):
    # ds.plot_thermal_frame(i)

    # T_path = project_root / "data"
    # T_path.mkdir(parents=True, exist_ok=True)
    # ds.save_thermal_video(T_path / "Temperature_snaps.mp4")


if __name__ == "__main__":
    main()
