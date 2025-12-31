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

global clicked_points
clicked_points = []

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent
results_path = project_root / "data" / "results" / "compression_tests_ss"
results_path.mkdir(parents=True, exist_ok=True)


def calculate_slope(p1, p2):
    """Calculates slope between two tuples (x, y)."""
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return float('inf')
    return (y2 - y1) / (x2 - x1)


def on_click(event):
    if event.inaxes is None:
        return

    global clicked_points
    clicked_points.append((event.xdata, event.ydata))
    plt.plot(event.xdata, event.ydata, 'ro')
    plt.draw()

    if len(clicked_points) == 2:
        p1, p2 = clicked_points[0], clicked_points[1]
        slope = calculate_slope(p1, p2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--')
        plt.title(f"Slope: {slope:.4f}")
        plt.draw()
        print(f"Point 1: {p1}")
        print(f"Point 2: {p2}")
        print(f"Calculated Slope: {slope}")
        clicked_points.clear()


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
    contact_start_list = [571, 423, 401, 416, 417, 400, 413]
    contact_end_list = [620, 539, 506, 493, 493, 443, 458]

    # Hand-calculated machine compliance for removing from stroke data

    if not (len(contact_start_list) == len(contact_end_list) == len(ds)):
        raise ValueError("Contact start/end lists must match dataset length.")

    C_machine = -0.0000077053824362 # [mm/N]
    # C_machine = 0

    for i in range(len(ds)):
        data_point = ds[i]
        load = data_point.load.cpu().numpy() * 1000  # [N]
        stroke = data_point.stroke.cpu().numpy()  # [mm]
        ds.plot_load_stroke(i)
        plt.figure()
        plt.plot(load)
        plt.title("Load vs. Index (t)")
        plt.show()
        plt.figure()
        plt.plot(np.abs(stroke), load)
        plt.title("Load vs. Stroke")
        plt.show()

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
        l0 = stroke[0]
        delta_l = stroke - l0 - C_machine * load
        strain = np.log( (l0 + delta_l) / l0)
        # plt.figure()
        # plt.plot(strain, color="tab:red", linewidth=2, linestyle="--", label="Strain")
        # plt.show()
        diameter_0_mm = 5 / 8 * 25.4  # [mm]
        area = np.pi * (diameter_0_mm) ** 2 / 4  # [mm^2]

        # σ = F/S
        # S = (π*d0^2)/4*L0/(L0+ΔL)
        S = area * l0 / (l0 + delta_l)  # [mm^2]
        stress = load / (S)  # [N/mm^2] = [MPa]
        plt.figure(figsize=(8, 6))
        plt.plot(np.abs(strain), stress, color="tab:blue", linewidth=2, linestyle="-", label="True Stress-Strain (Corrected)")
        plt.title("True Stress vs. True Strain\n15-5PH Steel")
        plt.xlabel("True Strain")
        plt.ylabel("True Stress [MPa]")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"data point path: {data_point.path}")
        df = pd.DataFrame({"strain": -strain, "stress_MPa": stress})
        df.to_csv(str(results_path / f"{data_point.path.split('/')[0]}.csv"))

        fig, ax = plt.subplots()
        ax.set_title("Click two points to find the slope")

        ax.plot(np.abs(strain), stress)
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        plt.show()

if __name__ == "__main__":
    main()
