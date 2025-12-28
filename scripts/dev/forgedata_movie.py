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

        # Overlay datapoint path in top-left corner
        try:
            sample = ds[i]
            path_text = str(sample.path)
        except Exception:
            path_text = f"idx:{i:06d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        margin = 8
        max_width = final.shape[1] - 2 * margin
        (text_w, text_h), baseline = cv2.getTextSize(path_text, font, font_scale, thickness)
        while text_w > max_width and font_scale > 0.4:
            font_scale -= 0.1
            (text_w, text_h), baseline = cv2.getTextSize(path_text, font, font_scale, thickness)
        display = path_text
        if text_w > max_width:
            # Truncate with ellipsis
            ellipsis = "…"
            max_chars = len(path_text)
            while text_w > max_width and max_chars > 0:
                max_chars -= 1
                display = path_text[:max_chars] + ellipsis
                (text_w, text_h), baseline = cv2.getTextSize(display, font, font_scale, thickness)
        x0 = margin
        y0 = margin
        rect_tl = (x0 - 4, y0 - 4)
        rect_br = (x0 + text_w + 4, y0 + text_h + 4)
        cv2.rectangle(final, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
        text_org = (x0, y0 + text_h)
        cv2.putText(final, display, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save final combined image
        combined_path = out_dir / f"{i:06d}.png"

        # OpenCV expects BGR
        cv2.imwrite(str(combined_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        if i % max(1, (len(ds) // 10)) == 0:
            print(f"Saved frame {i}/{len(ds)} -> {combined_path}")

if __name__ == "__main__":
    main()
