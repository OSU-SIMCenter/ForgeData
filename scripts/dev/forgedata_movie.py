"""
Validate ForgeData with movie of action sequence.
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

    import cv2
    import numpy as np

    out_dir = project_root / "data" / "processed" / "movie_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_size = (1024, 1024)

    for i in range(len(ds)):
        try:
            img_x, img_y = ds.plot_state_action(i, return_image=True, window_size=target_size)
        except Exception as e:
            print(f"Failed to render meshes for idx {i}: {e}")
            img_x = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            img_y = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        try:
            img_load = ds.plot_load_stroke(i, return_image=True, figsize=(6, 4))
            # normalize to target size
            img_load = cv2.resize(img_load, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Failed to render load/stroke for idx {i}: {e}")
            img_load = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        try:
            img_thermal = ds.plot_thermal_frame(i, return_image=True)
            img_thermal = cv2.resize(img_thermal, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Failed to render thermal for idx {i}: {e}")
            img_thermal = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Request a tall temperature-field image the same height as the combined 2x2 grid
        try:
            tall_size = (target_size[0], target_size[1] * 2)
            img_temperature_field = ds.plot_temperature_field(i, return_image=True, window_size=tall_size)
        except Exception as e:
            print(f"Failed to render temp field for idx {i}: {e}")
            img_temperature_field = np.zeros((target_size[1] * 2, target_size[0], 3), dtype=np.uint8)

        # Stack into 2x2 grid: [img_x | img_y]
        #                      [load  | thermal]
        top = np.hstack([img_x, img_y])
        bottom = np.hstack([img_load, img_thermal])
        combined = np.vstack([top, bottom])

        # Now stitch the tall temperature-field image on the RHS of the combined image
        final = np.hstack([combined, img_temperature_field])

        # Save final combined image
        combined_path = out_dir / f"{i:06d}.png"

        # OpenCV expects BGR
        cv2.imwrite(str(combined_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        if i % max(1, (len(ds) // 10)) == 0:
            print(f"Saved frame {i}/{len(ds)} -> {combined_path}")

if __name__ == "__main__":
    main()
