import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
import matplotlib
matplotlib.use('TkAgg')
from forge_data.torch_dataset.forge_dataset import ForgeDataset
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"], # Fallback chain
    "mathtext.fontset": "custom",
    "mathtext.rm": "Cambria",
    "mathtext.it": "Cambria:italic",
    "mathtext.bf": "Cambria:bold",
})
def plot_forge_data(datapoint, save_path="forge_plot.png", start_time=1767927550.0, end_time=1767927552.5):
    t_mech_raw = datapoint.t.numpy()
    t_therm_raw = datapoint.t_thermal.numpy()
    print(t_mech_raw[0])
    t_start = start_time
    t_end = t_mech_raw[-1] if end_time == -1 else end_time

    m_mask = (t_mech_raw >= t_start) & (t_mech_raw <= t_end)
    th_mask = (t_therm_raw >= t_start) & (t_therm_raw <= t_end)

    t = t_mech_raw[m_mask]
    load = datapoint.load.numpy()[m_mask]
    stroke = datapoint.stroke.numpy()[m_mask]

    t_thermal = t_therm_raw[th_mask]
    t = t_mech_raw[m_mask] - t_start
    t_thermal = t_therm_raw[th_mask] - t_start
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.75)

    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Load [kN]", color="tab:blue", fontsize=12)
    line1 = ax1.plot(t, load, color="tab:blue", linewidth=2, label="Load")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Stroke [mm]", color="tab:red", fontsize=12)
    line2 = ax2.plot(t, stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.15))
    ax3.set_ylabel("Temperature [°C]", color="tab:green", fontsize=12)
    ax3.tick_params(axis="y", labelcolor="tab:green")

    temp_lines = []
    styles = ["-.", "-", "--", ":", "-", "-."]
    green_shades = plt.cm.Greens(np.linspace(0.4, 1.0, 6))

    for i in range(6):
        attr_name = f"T_{i}_t"
        if hasattr(datapoint, attr_name):
            T_sliced = getattr(datapoint, attr_name).numpy()[th_mask]

            T_synced = np.interp(t, t_thermal, T_sliced)

            ln = ax3.plot(t, T_synced, color=green_shades[i], linewidth=1.2, 
                         linestyle=styles[i], alpha=0.8, label=f"Temp T{i}")
            temp_lines += ln

    lines = line1 + line2 + temp_lines
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize='x-small', ncol=2)

    plt.title(f"Forge Sample: Synchronized Load, Stroke & Temperature\n{datapoint.path}", fontsize=13)
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    leg = ax1.legend(
            lines, labels,
            loc="upper left",
            bbox_to_anchor=(0.5, 1.1),
            fontsize='x-small',
            ncol=4,
            frameon=False,
            framealpha=1.0,
        )
    leg.set_zorder(100)
    plt.close(fig)
    print(f"Synchronized plot saved to: {save_path}")



script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path, required=True)
    args = parser.parse_args()

    ds = ForgeDataset(args.h5_path)
    dp1 = ds[0]
    plot_forge_data(dp1)

if __name__ == "__main__":
    main()
