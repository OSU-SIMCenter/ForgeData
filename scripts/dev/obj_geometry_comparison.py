"""
Chamfer distance etc. for two obj files
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
import scienceplots

plt.style.use('science')

script_dir = Path(__file__).resolve()
project_root = script_dir.parent.parent.parent

samples = 1000000
def raycast(x, theta, omega):
    omega = pv.wrap(omega)
    pcd = []

    # --- raycast ---
    for pro in itertools.product(x, theta):
        x_i = pro[0]
        theta_i = pro[1]

        origin = np.array([x_i, 0.0, 0.0])
        end_point = np.array([x_i, 100*np.cos(theta_i), 100*np.sin(theta_i)])
        point, _ = omega.ray_trace(origin, end_point, first_point=True)
        if len(point) > 0:
            pcd.append(point)
        else:
            pcd.append(np.array([x_i, 0.0, 0.0]))

    pcd = np.asarray(pcd)
    return pcd


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


    v = np.asarray(mesh1.vertices)*10
    f = np.array(mesh1.triangles)
    faces = np.hstack(np.c_[np.full(len(f), 3), f])
    pv_mesh1 = pv.PolyData(v, faces)
    pv_mesh1 = pv.PolyData(v, faces)

    v = np.asarray(mesh2.vertices)*10
    f = np.array(mesh2.triangles)
    faces = np.hstack(np.c_[np.full(len(f), 3), f])
    pv_mesh2 = pv.PolyData(v, faces)


    print(f"Mesh Bounds: {pv_mesh1.bounds}")
    plotter = pv.Plotter(title="3D Mesh Preview")
    plotter.add_mesh(pv_mesh1, color="lightblue", show_edges=True, label="Input Mesh")
    plotter.add_mesh(pv_mesh2, color="red", show_edges=True, label="Input Mesh")

    plotter.show_grid()
    theta = 3*np.pi/2
    pcd1 = raycast(np.linspace(2, 98, 10000), [theta], pv_mesh1)
    pcd2 = raycast(np.linspace(2, 98, 10000), [theta], pv_mesh2)

    start_node = np.array([[pcd1[0, 0], 0.0, 0.0]])
    end_node = np.array([[pcd1[-1, 0], 0.0, 0.0]])
    pcd1 = np.concatenate([start_node, pcd1, end_node], axis=0)
    start_node = np.array([[pcd2[0, 0], 0.0, 0.0]])
    end_node = np.array([[pcd2[-1, 0], 0.0, 0.0]])
    pcd2 = np.concatenate([start_node, pcd2, end_node], axis=0)
    radius1 = np.linalg.norm(pcd1[:, 1:3], axis=1)
    radius2 = np.linalg.norm(pcd2[:, 1:3], axis=1)

    HUGE_FONT = 12
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Cambria"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": HUGE_FONT,
        "ytick.labelsize": HUGE_FONT,
        "legend.fontsize": HUGE_FONT,
        "axes.labelsize": HUGE_FONT + 4,
    })

    plt.figure(figsize=(8, 4))
    plt.plot(pcd1[:, 0], radius1, label="Exp.", color='blue', linewidth=1.5)
    plt.plot(pcd2[:, 0], radius2, label="Sim.", color='red', linewidth=1.5, linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(pcd1[:, 0].min(), pcd1[:, 0].max())
    plt.ylim(0, radius1.max() * 1.2)
    plt.xlabel("Axial Position $x$ [mm]")
    plt.ylabel("Radius $r$ [mm]")
    plt.legend(frameon=False, loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(pad=1.5)



    x_val1 = 80.0
    x_val2 = 96.0
    r_val1 = np.interp(x_val1, pcd1[:, 0], radius1)
    r_val2 = np.interp(x_val2, pcd2[:, 0], radius2)

    plt.vlines(x=x_val1, ymin=0, ymax=r_val1, colors='black', linestyles='dotted', linewidth=1.5)
    plt.vlines(x=x_val2, ymin=0, ymax=r_val2, colors='black', linestyles='dotted', linewidth=1.5)
    plt.savefig("GeoComp A.png", bbox_inches='tight')


    def generate_comparison_plots(pv_mesh1, pv_mesh2, x_slice=5.0):
        x_range = np.linspace(0.1, 9.8, 500)
        pcd_top1 = raycast(x_range, [0], pv_mesh1)
        pcd_top2 = raycast(x_range, [0], pv_mesh2)

        r_top1 = np.linalg.norm(pcd_top1[:, 1:3], axis=1)
        r_top2 = np.linalg.norm(pcd_top2[:, 1:3], axis=1)

        theta_range = np.linspace(0, 2*np.pi, 200)

        pcd_rad1 = raycast([x_slice], theta_range, pv_mesh1)
        pcd_rad2 = raycast([x_slice], theta_range, pv_mesh2)

        r_rad1 = np.linalg.norm(pcd_rad1[:, 1:3], axis=1)
        r_rad2 = np.linalg.norm(pcd_rad2[:, 1:3], axis=1)
        y1, z1 = pcd_rad1[:, 1], pcd_rad1[:, 2]
        y2, z2 = pcd_rad2[:, 1], pcd_rad2[:, 2]
        fig = plt.figure(figsize=(5, 5))

        plt.plot(y1, z1, color='blue', label='Experimental', linewidth=1.5)
        plt.plot(y2, z2, color='red', linestyle='--', label='Simulated', linewidth=1.5)

        plt.gca().set_aspect('equal', adjustable='datalim')

        plt.gca().set_xlabel("$y$ [mm]")
        plt.gca().set_ylabel("$z$ [mm]")
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout(pad=4.0)

    generate_comparison_plots(pv_mesh1, pv_mesh2, x_slice=80.0)
    generate_comparison_plots(pv_mesh1, pv_mesh2, x_slice=96.5)
    plt.show()

if __name__ == "__main__":
    main()
