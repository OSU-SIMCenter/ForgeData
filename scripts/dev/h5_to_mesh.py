"""
Pull an OBJ file from an H5 and write to disk.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from forge_data.torch_dataset.forge_dataset import ForgeDataset

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path)
    args = parser.parse_args()

    ds = ForgeDataset(args.h5_path)

    print(f"There are {len(ds)} data points in the database.")

    mesh = ds[5].y

    v = mesh.vertices.cpu().numpy()
    f = mesh.faces.cpu().numpy()

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
    o3d.io.write_triangle_mesh("output_mesh.obj", o3d_mesh)


if __name__ == "__main__":
    main()
