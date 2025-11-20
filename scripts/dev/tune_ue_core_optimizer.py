"""
Parses through a directory and generates an H5 / sqlite3 database.
"""

import argparse
import ast
import os
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from forge_data.data.parser import process_linescanner_file
from forge_data.ue.api import ReconConfig
from forge_data.ue.core import Recon

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent
default_raw_path = project_root / "data" / "raw"
default_save_path = project_root / "data" / "results"
os.makedirs(default_save_path, exist_ok=True)


def parse_input(prompt, default_value):
    """
    Helper to handle user input with defaults and type evaluation.
    """
    user_str = input(f"{prompt} (default: {default_value}):\n> ")

    if not user_str.strip():
        return default_value

    try:
        return ast.literal_eval(user_str)
    except (ValueError, SyntaxError):
        print(f"Error parsing input. Falling back to default: {default_value}")
        return default_value


def plot_pointcloud(pcd, save_path):
    """ """
    if pcd.shape[0] > 100000:
        print(f"Subsampling {pcd.shape[0]} points to 100k for display.")
        indices = np.random.choice(pcd.shape[0], 100000, replace=False)
        pcd = pcd[indices]

    fig = go.Figure(
        data=[go.Scatter3d(x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], mode="markers", marker={"size": 2, "opacity": 0.8})]
    )

    fig.update_layout(
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        title="Reconstructed Point Cloud (Refresh page to update)",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    fig.write_html(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=Path, default=default_raw_path)
    parser.add_argument("--save-path", type=Path, default=default_save_path)
    parser.add_argument("--format", type=str, choices=["h5", "sqlite"], default="h5")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    data_path = Path(args.raw_data_path)
    df = process_linescanner_file(data_path)
    av = [1.0, 0.0, 0.0, 290]
    ob = ((0, 1), (-0.1, 0.1), (-0.1, 0.1), (200, 380))
    first_run = True
    try:
        while True:
            av = parse_input("Input axis_angle_vector", av)
            ob = parse_input("Input optimization_bounds", ob)
            recon_args = ReconConfig(default_axis_angle_vector=av, default_optimization_bounds=ob)
            recon_args.default_axis_angle_vector = av
            recon_args.default_optimization_bounds = ob
            recon_args.error_comp = False

            recon = Recon(recon_args)
            recon.preprocess(df.copy())
            recon.process()
            pcd = recon.post_process()
            fig_save_path = default_save_path / "temp_reconstruction.html"
            plot_pointcloud(pcd, fig_save_path)

            if first_run:
                webbrowser.open("file://" / fig_save_path)
                first_run = False
            else:
                print("Plot updated. Refresh your browser.")

            print("-" * 30)

    except KeyboardInterrupt:
        print("Done...")


if __name__ == "__main__":
    main()
