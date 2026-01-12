import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np


def get_thermal_datasets(h5_file):
    all_thermal_data = []

    def visitor(name, node):
        if (
            isinstance(node, h5py.Group)
            and (name.endswith("/t") or name == "t")
            and "frames" in node
            and "time" in node
        ):
            all_thermal_data.append({"path": name, "frames": node, "time": node["time"]})
        return None

    h5_file.visititems(visitor)
    return all_thermal_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        thermal_datasets = get_thermal_datasets(f)
        if not thermal_datasets:
            return

        dataset = thermal_datasets[0]
        frames = dataset["frames"]["frames"]
        times = dataset["time"]

        num_frames = len(frames)
        window_name = "Thermal Analyzer - Space: Play/Pause | Q: Quit"
        cv2.namedWindow(window_name)

        state = {"index": 0, "paused": True, "mouse_pos": (0, 0)}

        def on_trackbar(val):
            state["index"] = val

        def on_mouse(event, x, y, flags, param):
            state["mouse_pos"] = (x, y)

        cv2.createTrackbar("Frame", window_name, 0, num_frames - 1, on_trackbar)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            # Load current frame
            idx = state["index"]
            frame_u16 = frames[idx]
            frame_c = frame_u16.astype(np.float32) / 10.0 - 100.0

            vis = cv2.normalize(frame_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

            mx, my = state["mouse_pos"]
            if 0 <= mx < colorized.shape[1] and 0 <= my < colorized.shape[0]:
                temp = frame_c[my, mx]

                # Updated string to include pixel coordinates
                t_str = f"({mx}, {my}) {temp:.2f} C"

                # Draw text with shadow for readability
                cv2.putText(colorized, t_str, (mx + 15, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(colorized, t_str, (mx + 15, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                colorized,
                f"Time: {times[idx]:.4f}s | Frame: {idx}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow(window_name, colorized)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):  # Toggle pause
                state["paused"] = not state["paused"]
            elif key == ord("q") or key == 27:
                break

            if not state["paused"]:
                state["index"] = (state["index"] + 1) % num_frames
                cv2.setTrackbarPos("Frame", window_name, state["index"])

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
