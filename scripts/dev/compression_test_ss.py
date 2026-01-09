"""
Generate stress strain curves from compression tests on 2025-12-24

Eq's from https://doi.org/10.1038/s41598-023-43129-3
"""

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge_data.torch_dataset.forge_dataset import ForgeDataset

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    contact_start_list = [571,420,420,416, 423, 401, 416, 417, 400, 413,424,415,415,0,419,417]
    contact_end_list = [620,467,466,457, 539, 506, 493, 493, 443, 458,473,463,462,-1,459,461,]

    if not (len(contact_start_list) == len(contact_end_list) == len(ds)):
        raise ValueError("Contact start/end lists must match dataset length.")

    # Hand-calculated machine compliance for removing from stroke data. This came from the baseline tests.
    C_machine = -0.0000077053824362 # [mm/N]
    # C_machine = 0
    C_machine = -8.252e-6  # [mm/N]

    for i in range(len(ds)):
        datapoint = ds[i]
        print(f"data point path: {datapoint.path}")

        t = datapoint.t.cpu().numpy()
        t = t - np.min(t)
        load = datapoint.load.cpu().numpy()
        stroke = datapoint.stroke.cpu().numpy()
        ds.plot_thermal_frame_callback(i)


        # Plot original load/stroke
        # fig, ax1 = plt.subplots(figsize=(6, 4))
        # ax1.set_xlabel("Time [s]", fontsize=12)
        # ax1.set_ylabel("Load [kN]", fontsize=12)
        # line1 = ax1.plot(t, load, color="tab:blue", linewidth=2, label="Load")
        # ax1.tick_params(axis="y")
        # ax1.grid(True, linestyle="--", alpha=0.7)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Stroke [mm]", fontsize=12)
        # line2 = ax2.plot(t, stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        # ax2.tick_params(axis="y")
        # lines = line1 + line2
        # labels = [ln.get_label() for ln in lines]
        # ax1.legend(lines, labels, loc="upper right")
        # plt.title("Load & Stroke vs. Time", fontsize=14)
        # plt.tight_layout()

        # fig, ax1 = plt.subplots(figsize=(6, 4))
        # ax1.set_xlabel("Index", fontsize=12)
        # ax1.set_ylabel("Load [kN]", fontsize=12)
        # line1 = ax1.plot(load, color="tab:blue", linewidth=2, label="Load")
        # ax1.tick_params(axis="y")
        # ax1.grid(True, linestyle="--", alpha=0.7)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Stroke [mm]", fontsize=12)
        # line2 = ax2.plot(stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        # ax2.tick_params(axis="y")
        # lines = line1 + line2
        # labels = [ln.get_label() for ln in lines]
        # ax1.legend(lines, labels, loc="upper right")
        # plt.title("Load & Stroke vs. Time", fontsize=14)
        # plt.tight_layout()

        # Clip and plot load/stroke
        t = t[contact_start_list[i] : contact_end_list[i]]
        load = load[contact_start_list[i] : contact_end_list[i]]
        stroke = stroke[contact_start_list[i] : contact_end_list[i]]

        # fig, ax1 = plt.subplots(figsize=(6, 4))
        # ax1.set_xlabel("Time [s]", fontsize=12)
        # ax1.set_ylabel("Load [kN]", fontsize=12)
        # line1 = ax1.plot(t, load, color="tab:blue", linewidth=2, label="Load")
        # ax1.tick_params(axis="y")
        # ax1.grid(True, linestyle="--", alpha=0.7)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Stroke [mm]", fontsize=12)
        # line2 = ax2.plot(t, stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        # ax2.tick_params(axis="y")
        # lines = line1 + line2
        # labels = [ln.get_label() for ln in lines]
        # ax1.legend(lines, labels, loc="upper right")
        # plt.title("Load & Stroke vs. Time", fontsize=14)
        # plt.tight_layout()

        # Plot clipped load/stroke without time
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.set_xlabel("Index", fontsize=12)
        ax1.set_ylabel("Load [kN]", fontsize=12)
        line1 = ax1.plot(load, color="tab:blue", linewidth=2, label="Load")
        ax1.tick_params(axis="y")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Stroke [mm]", fontsize=12)
        line2 = ax2.plot(stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        ax2.tick_params(axis="y")
        lines = line1 + line2
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="upper right")
        plt.title("Load & Stroke vs. Time", fontsize=14)
        plt.tight_layout()

        # Baseline plots
        if "(baseline)" in datapoint.path:
            # fig, ax1 = plt.subplots(figsize=(6, 4))
            # ax1.set_xlabel("Stroke [mm]", fontsize=12)
            # ax1.set_ylabel("Load [kN]", fontsize=12)
            # line1 = ax1.plot(stroke, load, color="tab:blue", linewidth=2, label="Load")
            # ax1.tick_params(axis="y")
            # ax1.grid(True, linestyle="--", alpha=0.7)
            # # ax2 = ax1.twinx()
            # # ax2.set_ylabel("Stroke [mm]", fontsize=12)
            # # line2 = ax2.plot(stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
            # # ax2.tick_params(axis="y")
            # # lines = line1 + line2
            # # labels = [ln.get_label() for ln in lines]
            # # ax1.legend(lines, labels, loc="upper right")
            # plt.title("Load vs. Stroke from baseline", fontsize=14)
            # plt.tight_layout()
            # cid = fig.canvas.mpl_connect('button_press_event', on_click)
            # plt.show()
            continue # Don't show stress/strain for baseline tests




        # ε = ln((L0+ΔL)/L0)
        # σ = F/S
        # S = (π*d0^2)/4*L0/(L0+ΔL)

        load *=1000
        l0 = stroke[0]  # [mm]
        delta_l = stroke - l0 - C_machine * load
        strain = np.log( (l0 + delta_l) / l0)
        diameter_0_mm = 5 / 8 * 25.4  # [mm]
        area = np.pi * (diameter_0_mm) ** 2 / 4  # [mm^2]
        S = area * l0 / (l0 + delta_l)  # [mm^2]
        stress = load / (S)  # [N/mm^2] = [MPa]


        # Plot the stress strain data
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.set_xlabel("true strain", fontsize=12)
        ax1.set_ylabel("true stress [MPa]", fontsize=12)
        line1 = ax1.plot(np.abs(strain), stress, color="tab:blue", linewidth=2, label="Stress")
        ax1.tick_params(axis="y")
        ax1.grid(True, linestyle="--", alpha=0.7)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Stroke [mm]", fontsize=12)
        # line2 = ax2.plot(stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        # ax2.tick_params(axis="y")
        # lines = line1 + line2
        # labels = [ln.get_label() for ln in lines]
        # ax1.legend(lines, labels, loc="upper right")
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.title("True stress/strain", fontsize=14)
        plt.tight_layout()



        plt.show()


        df = pd.DataFrame({"strain": -strain, "stress_MPa": stress})
        df.to_csv(str(results_path / f"{datapoint.path.split('/')[0]}.csv"))


if __name__ == "__main__":
    main()
