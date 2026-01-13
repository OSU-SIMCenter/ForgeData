"""
Chamfer distance etc. for two obj files
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent

samples = 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-path", type=Path)
    parser.add_argument("--m2-path", type=Path)
    args = parser.parse_args()

    mesh1 = o3d.io.read_triangle_mesh(str(args.m1_path))
    mesh2 = o3d.io.read_triangle_mesh(str(args.m2_path))

    pcd1 = mesh1.sample_points_uniformly(number_of_points=samples)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=samples)
    np.asarray(pcd1.points)
    np.asarray(pcd2.points)

    dists_1to2 = pcd1.compute_point_cloud_distance(pcd2)
    dists_2to1 = pcd2.compute_point_cloud_distance(pcd1)

    chamfer_dist = np.mean(np.square(dists_1to2)) + np.mean(np.square(dists_2to1))
    hausdorff_dist = max(max(dists_1to2), max(dists_2to1))  # https://en.wikipedia.org/wiki/Hausdorff_distance

    output = (
        f"Metrics for {args.m1_path.name} vs {args.m2_path.name}:\n"
        f"Mesh 1: {args.m1_path.resolve()}\n"
        f"Mesh 2: {args.m2_path.resolve()}\n"
        f"{'-' * 40}\n"
        f"Chamfer Distance:    {chamfer_dist:.6f} [mm2]\n"
        f"Hausdorff Distance:  {hausdorff_dist:.6f} [mm]\n"
    )

    print(output)
    output_file = args.m1_path.parent / "geom_comp_metrics.txt"
    try:
        output_file.write_text(output)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to disk: {e}")

if __name__ == "__main__":
    main()
