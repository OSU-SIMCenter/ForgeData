import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
from forge_data.torch_dataset.forge_dataset import ForgeDataset
import matplotlib.pyplot as plt

x0_slice = np.s_[:, 110:150, 35:40]
x1_slice = np.s_[:, 80:90, 53:70]
x2_slice = np.s_[:, 110:150, 53:70]
x3_slice = np.s_[:, 170:180, 53:70]
x4_slice = np.s_[:, 200:210, 53:70]
x5_slice = np.s_[:, 230:240, 53:70]

def plot_forge_data(datapoint, save_path="forge_plot.png", start_time=0, end_time=-1):
    # 1. Extract raw data
    t_mech_raw = datapoint.t.numpy()
    t_therm_raw = datapoint.t_thermal.numpy()
    
    # 2. Determine time boundaries
    t_start = start_time
    t_end = t_mech_raw[-1] if end_time == -1 else end_time
    
    # 3. Create masks based on TIME (not indices) to preserve alignment
    m_mask = (t_mech_raw >= t_start) & (t_mech_raw <= t_end)
    th_mask = (t_therm_raw >= t_start) & (t_therm_raw <= t_end)
    
    # Slice mechanical data
    t = t_mech_raw[m_mask]
    load = datapoint.load.numpy()[m_mask]
    stroke = datapoint.stroke.numpy()[m_mask]
    
    # Slice thermal data
    t_thermal = t_therm_raw[th_mask]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.75)

    # --- Load (Primary Axis) ---
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Load [kN]", color="tab:blue", fontsize=12)
    line1 = ax1.plot(t, load, color="tab:blue", linewidth=2, label="Load")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Stroke (Secondary Axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Stroke [mm]", color="tab:red", fontsize=12)
    line2 = ax2.plot(t, stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # --- Temperature (Tertiary Axis) ---
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
            # Get full thermal array and slice it by the time mask
            T_sliced = getattr(datapoint, attr_name).numpy()[th_mask]
            
            # Interpolate the sliced thermal data onto the sliced mechanical time-grid
            T_synced = np.interp(t, t_thermal, T_sliced)
            
            ln = ax3.plot(t, T_synced, color=green_shades[i], linewidth=1.2, 
                         linestyle=styles[i], alpha=0.8, label=f"Temp T{i}")
            temp_lines += ln

    # Combine legends
    lines = line1 + line2 + temp_lines
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize='x-small', ncol=2)

    plt.title(f"Forge Sample: Synchronized Load, Stroke & Temperature\n{datapoint.path}", fontsize=13)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Synchronized plot saved to: {save_path}")

def get_thermal_datasets(h5_file):
    all_thermal_data = []
    def visitor(name, node):
        if isinstance(node, h5py.Group) and (name.endswith("/t") or name == "t") and "frames" in node and "time" in node:
            all_thermal_data.append({"path": name, "frames": node, "time": node["time"]})
        return None
    h5_file.visititems(visitor)
    return all_thermal_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--output", type=str, default="thermal_slice.mp4")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index")
    args = parser.parse_args()

    datapoint_index = 0

    x0_slice = np.s_[:, 110:150, 35:40]
    x1_slice = np.s_[:, 80:90, 53:70]
    x2_slice = np.s_[:, 110:150, 53:70]
    x3_slice = np.s_[:, 170:180, 53:70]
    x4_slice = np.s_[:, 200:210, 53:70]
    x5_slice = np.s_[:, 230:240, 53:70]

    regions = [
        (x0_slice, "T0", (0, 255, 0)),    # Green
        (x1_slice, "T1", (255, 255, 0)),  # Cyan
        (x2_slice, "T2", (255, 0, 0)),    # Blue
        (x3_slice, "T3", (0, 0, 255)),    # Red
        (x4_slice, "T4", (128, 0, 128)),  # Purple
        (x5_slice, "T5", (255, 0, 255))   # Magenta
    ]

    with h5py.File(args.h5_path, "r") as f:
        thermal_datasets = get_thermal_datasets(f)
        if not thermal_datasets:
            print("No thermal datasets found.")
            return

        dataset = thermal_datasets[datapoint_index]
        frames = dataset["frames"]["frames"]
        times = dataset["time"]

        
        total_frames = len(frames)
        end_idx = total_frames if args.end == -1 else min(args.end, total_frames)
        start_idx = max(0, args.start)
        start_time = times[start_idx]
        end_time = times[end_idx]
        # Video Writer Setup
        height, width = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 27.0, (width, height))

        print(f"Processing frames {start_idx} to {end_idx}...")
        print("Time (s) | " + " | ".join([r[1] for r in regions]))

        for i in range(start_idx, end_idx):
            frame_u16 = frames[i]
            frame_c = frame_u16.astype(np.float32) / 10.0 - 100.0

            vis = cv2.normalize(frame_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

            temp_reports = []
            for slc, label, color in regions:
                y_s, y_e = slc[1].start, slc[1].stop
                x_s, x_e = slc[2].start, slc[2].stop
                
                roi_mean = np.mean(frame_c[y_s:y_e, x_s:x_e])
                temp_reports.append(f"{roi_mean:6.2f}C")

                x_text_pos = x_s
                if label == "T0":
                    x_text_pos = 0
                cv2.rectangle(colorized, (x_s, y_s), (x_e, y_e), color, 1)
                cv2.putText(colorized, f"{label}:{roi_mean:.1f}", (x_text_pos, y_s - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            print(f"{times[i]:8.4f} | " + " | ".join(temp_reports))

            cv2.putText(colorized, f"t: {times[i]:.3f}s", (10, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(colorized)
            
        out.release()
        print(f"\nExport complete: {args.output}")




        # Generate a load/stroke plot of the five regions
        ds = ForgeDataset(args.h5_path)
        same_sample = ds[datapoint_index]

        plot_forge_data(same_sample, save_path="forge_data_plot.png", start_time=start_time, end_time=end_time)

if __name__ == "__main__":
    main()