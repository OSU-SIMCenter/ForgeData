"""
Open an h5 and stream the thermal video
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent


def get_thermal_datasets(h5_file):
    """
    Returns a list of all temperature dataset pairs (frames, time) found in the h5 file.
    """
    all_thermal_data = []

    def visitor(name, node):
        # Look for groups that represent a thermal dataset
        if isinstance(node, h5py.Group) and (name.endswith("/t") or name == "t") and "frames" in node and "time" in node:
            all_thermal_data.append({
                "path": name,
                "frames": node["frames"],
                "time": node["time"]
            })
        return None

    h5_file.visititems(visitor)

    if not all_thermal_data:
        print("No thermal datasets found.")

    return all_thermal_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-path",
        type=Path,
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)

    with h5py.File(h5_path, "r") as f:
        thermal_datasets = get_thermal_datasets(f)

        if not thermal_datasets:
            print("No thermal datasets found.")
            return

        dataset = thermal_datasets[10]
        frames = dataset["frames"]
        time = dataset["time"]

        fps = 60
        delay_ms = int(1000 / fps)
        stride = 1

        h, w = frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(h5_path.with_suffix(".mp4")), fourcc, fps, (w, h))

        print(f"Camera resolution: {frames.shape[1:]}")

        for i in range(0, len(frames), stride):
            frame_u16 = frames[i]
            time_instance = time[i]

            frame_celsius = frame_u16.astype(np.float32) / 10.0 - 100.0

            vis = cv2.normalize(frame_celsius, None, 0, 255, cv2.NORM_MINMAX)
            vis = vis.astype(np.uint8)
            colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

            out.write(colorized)

            print(f"{time_instance:12.4f},    {np.max(frame_celsius):12.4f}")
            cv2.imshow("Thermal Video (press q/esc to quit)", colorized)

            key = cv2.waitKey(delay_ms)
            if key == 27 or key == 81 or key == 113:
                break

        cv2.destroyAllWindows()
        out.release()


if __name__ == "__main__":
    main()
