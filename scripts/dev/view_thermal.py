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
    This will return the first temp dataset it finds.
    """
    data = []

    def visitor(name, node):
        print(name)
        if isinstance(node, h5py.Group) and (name.endswith("/t") or name == "t"):
            data.append(node["frames"])
            data.append(node["time"])
            return True
        return None

    h5_file.visititems(visitor)
    return data[0], data[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-path",
        type=Path,
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    with h5py.File(h5_path, "r") as f:
        frames, time = get_thermal_datasets(f)

        fps = 60
        delay_ms = int(1000 / fps)
        stride = 20
        print(f"Camera resolution: {frames.shape[1:]}")

        for i in range(0, len(frames), stride):
            frame_u16 = frames[i]  # raw uint16 thermal frame
            time_instance = time[i]

            # Convert to Celsius
            frame_celsius = frame_u16.astype(np.float32) / 10.0 - 100.0

            vis = cv2.normalize(frame_celsius, None, 0, 255, cv2.NORM_MINMAX)
            vis = vis.astype(np.uint8)
            colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

            print(f"{time_instance:12.4f},    {np.max(frame_celsius):12.4f}")
            cv2.imshow("Thermal Video (press q/esc to quit)", colorized)

            key = cv2.waitKey(delay_ms)
            if key == 27 or key == 81 or key == 113:  # ESC q or Q key to stop
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
