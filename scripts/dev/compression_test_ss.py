"""
Generate stress strain curves from compression tests on 2025-12-24

Eq's from https://doi.org/10.1038/s41598-023-43129-3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge_data.torch_dataset.forge_dataset import ForgeDataset

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent
results_path = project_root / "data" / "results" / "compression_tests_ss"
results_path.mkdir(parents=True, exist_ok=True)

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

    # Since there are only 6 compression tests, pick start/end indices by hand
    contact_start_list = [423, 400, 412, 414, 397, 410]
    contact_end_list = [539, 506, 493, 493, 443, 458]

    if not (len(contact_start_list) == len(contact_end_list) == len(ds)):
        raise ValueError("Contact start/end lists must match dataset length.")

    for i in range(len(ds)):
        data_point = ds[i]
        load = data_point.load.cpu().numpy() * 1000 / 9.807  # [kgf]
        stroke = data_point.stroke.cpu().numpy()  # [mm]
        # ds.plot_load_stroke(i)
        # plt.figure()
        # plt.plot(load)
        # plt.show()

        load = load[contact_start_list[i] : contact_end_list[i]]
        stroke = stroke[contact_start_list[i] : contact_end_list[i]]
        # plt.figure()
        # plt.plot(load, color="tab:blue", linewidth=2, label="Load")
        # plt.figure()
        # plt.plot(stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        # plt.show()

        # Filter load/stroke to where load > 0

        # Convert to stress and strain
        # ε = ln((L0+ΔL)/L0)
        strain = np.log(stroke / stroke[0])
        # plt.figure()
        # plt.plot(strain, color="tab:red", linewidth=2, linestyle="--", label="Strain")
        # plt.show()
        diameter_0_mm = 5 / 8 * 25.4  # [mm]
        area = np.pi * (diameter_0_mm) ** 2 / 4  # in mm^2

        # σ = F/S
        # S = (π*d0^2)/4*L0/(L0+ΔL)
        S = area * stroke[0] / stroke  # [mm^2]
        stress = load / (S)  # [kgf/mm^2] = [MPa]
        # plt.figure()
        # plt.plot(-strain, stress, color="tab:blue", linewidth=2, linestyle="-", label="Stress/Strain")
        # plt.show()
        # print(f"data point path: {data_point.path}")
        df = pd.DataFrame({"strain": -strain, "stress_MPa": stress})
        df.to_csv(str(results_path / f"{data_point.path.split('/')[0]}.csv"))
        # stress_Pa = load / area_m2  # in Pascals
        # strain = stroke / 10.0  # assuming initial length of 10 mm

        # Save or plot stress-strain curve as needed
        # print(f"Data Point {i}:")
        # for s, e in zip(stress_Pa, strain, strict=False):
        # print(f"Strain: {e:.4f}, Stress: {s / 1e6:.2f} MPa")  # Print stress in MPa


if __name__ == "__main__":
    main()
