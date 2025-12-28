import time
from functools import wraps

import numpy as np
import open3d as o3d
import scipy.optimize
from scipy.spatial.transform import Rotation

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"    [ue.core] {func.__name__} took {execution_time:.4f} seconds")
        return result

    return wrapper


class Recon:
    def __init__(self, args):
        """
        TODO
        """

        self.args = args
        self.n_error_calls = 0
        self.generate_stock_pcd(0.625 * 25.4, "cylindrical")

        self.i = 0
        self.j = 0

    # @timeit
    def reconstruct_pcd_axis_angle(self, V, *args, **kwargs):
        self.j += 1

        rotation_axis = V[0:3]
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        radial_offset = V[3]
        (scanner_data, return_pcd) = args
        (_, x, z, th) = scanner_data
        n_rotations = th.shape[0]

        reconstructed_points = []

        z = z - radial_offset
        for i in range(n_rotations):
            rotation_vec = Rotation.from_rotvec(-th[i] * rotation_axis)
            scan = np.stack((x[i], np.zeros_like(x[i]), z[i]), axis=1)
            mask = ~np.isnan(scan[:, 0]) & ~np.isnan(scan[:, 2])
            scan = scan[mask]
            transformed_points = rotation_vec.apply(scan)
            reconstructed_points.append(transformed_points)

        reconstructed_pcd = np.vstack(reconstructed_points)

        if not return_pcd:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(reconstructed_pcd)
            return np.mean(self.target_pcd.compute_point_cloud_distance(pcd1))

        return reconstructed_pcd

    def reconstruct_pcd(self, scanner_data):
        if self.args.error_comp:
            res = scipy.optimize.minimize(
                self.reconstruct_pcd_axis_angle,
                np.array(self.args.default_axis_angle_vector),
                args=(scanner_data, False),
                method="SLSQP",
                tol=1e-6,
                bounds=self.args.default_optimization_bounds,
            )
            xf = res.x
        else:
            xf = np.array(self.args.default_axis_angle_vector)
        # print(f"Final vector: {xf}")
        reconstructed_pcd = self.reconstruct_pcd_axis_angle(xf, *(scanner_data, True))
        return reconstructed_pcd, xf

    # @timeit
    def preprocess(self, df):
        """
        TODO
        """

        # Split the dataframe into different scans when theta rolls over.
        reset_indices = df.index[df["a_axis_deg"].diff() < -180].tolist()
        df_list = np.split(df, reset_indices)  # TODO: Rm, causes FutureWarning
        self.scanner_data_list = []
        for df in df_list:
            t = np.stack(df["timestamps_ms"].to_numpy())
            # T = np.stack(df["temperature_C"].to_numpy())
            x = np.stack(df["x_mm"].to_numpy())
            z = np.stack(df["z_mm"].to_numpy())
            th = np.stack(df["a_axis_deg"].to_numpy())

            # rm zero points
            mask = (x == 0) & (z == 0)
            x = np.where(mask, np.nan, x)
            z = np.where(mask, np.nan, z)
            x = x - np.nanmin(x)  # Flip x axis and slide to origin for convenience...

            th = np.deg2rad(th)

            scanner_data = (
                t,
                x,
                z,
                th,
            )
            self.scanner_data_list.append(scanner_data)

    # @timeit
    def process(self):
        """
        TODO
        """

        scanner_data = self.scanner_data_list[0]
        pcd, _ = self.reconstruct_pcd(scanner_data)
        pcds = []
        no_overlap_pcds = []
        pcds.append(pcd)
        no_overlap_pcds.append(pcd)
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([pcd0, self.target_pcd], point_show_normal=True)

        # TODO: Have Brian remove offset from raw data & save machine coordinates from scans.

        # for i, scan_data in enumerate(self.scanner_data_list[1:], start=1):
        #     self.i = i

        #     # Create variables needed by error metric for RMSE
        #     t_pcd = pcds[i - 1]
        #     self.current_overlap_start = np.nanmax(t_pcd[:, 0])
        #     overlap_end = self.scanner_data_list[1:][0].min()

        #     mask = (t_pcd[:, 0] > overlap_end)
        #     # t_pcd = ...

        #     target_pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points)
        #     # self.target_pcd = ...

        #     pcd, _ = self.multiscan_stitching_error_compensation(scan_data)

        #     # Remove overlap from pointclouds for final pcd. Of course we still need
        #     # that overlap for the error compensation.
        #     mask = pcd[:, 0] >= overlap_end
        #     pcd_no_overlap = pcd[mask]
        #     no_overlap_pcds.append(pcd_no_overlap)

        #     pcds.append(pcd)

        self.finished_recon_pcd = np.vstack(no_overlap_pcds)

        return self.finished_recon_pcd

    # @timeit
    def post_process(self):
        pcd = self.finished_recon_pcd
        part_length = np.nanmax(pcd[:, 0]) - np.nanmin(pcd[:, 0])
        part_min = np.nanmin(pcd[:, 0])

        # --- Clip relative percentages off the origin and end ---
        clip_origin_fraction = 0.01
        clip_end_fraction = 0.05
        clip_origin = clip_origin_fraction * part_length
        clip_end = clip_end_fraction * part_length
        mask = (pcd[:, 0] >= part_min + clip_origin) & (pcd[:, 0] <= part_length - clip_end)
        pcd = pcd[mask]
        pcd[:, 0] = pcd[:, 0] - pcd[:, 0].min()
        part_min = np.nanmin(pcd[:, 0])

        # --- Statistical outlier removal ---
        pcdo = o3d.geometry.PointCloud()
        pcdo.points = o3d.utility.Vector3dVector(pcd)
        pcdo, _ = pcdo.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = np.asarray(pcdo.points)

        # --- Cap origin (x ~ clipped start) ---
        slice_thickness_origin = 2.0  # mm, enough points for triangulation
        mask = (pcd[:, 0] >= part_min + clip_origin) & (pcd[:, 0] <= part_min + clip_origin + slice_thickness_origin)
        startpoints = pcd[mask]
        startpoints_yz = startpoints[:, [1, 2]]
        origin_filled_points = grid_ring_with_points(startpoints_yz)
        x = (part_min + np.zeros_like(origin_filled_points[:, 0])).reshape(-1, 1)
        origin_filled_points_3d = np.hstack((x, origin_filled_points))

        # --- Cap end (x ~ clipped end) ---
        new_length = np.nanmax(pcd[:, 0])
        self.new_length = new_length
        slice_thickness_end = 1.0  # mm
        mask = pcd[:, 0] >= new_length - slice_thickness_end
        endpoints = pcd[mask]
        endpoints_yz = endpoints[:, [1, 2]]
        filled_points = grid_ring_with_points(endpoints_yz)
        x = (new_length + np.zeros_like(filled_points[:, 0])).reshape(-1, 1)
        filled_points_3d = np.hstack((x, filled_points))

        pcd = np.vstack([pcd, origin_filled_points_3d, filled_points_3d])

        return pcd

    # @timeit
    def pcd_to_mesh(self, pcd0):
        """
        TODO
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd0)
        radius = 1
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        np_points = np.array(pcd.points)
        np_normals = np.array(pcd.normals)

        # Vector r = [x,y,z] - [x,0,0] = [0, -y, -z]
        # norm dot r = [0, -ny*y, -nz*z]
        norm_dot_radius = -(np_points[:, 1] * np_normals[:, 1] + np_points[:, 2] * np_normals[:, 2])
        mask_centers_flip = norm_dot_radius > 0  # If the norm points toward the x-axis, you'll want to flip it.
        # Convert True=1 False=0 to True=-1 and False=1 for multiply
        mask_centers_flip_mult = 1 - 2 * mask_centers_flip.astype(int)
        temp_normal = np_normals[:, 0] * mask_centers_flip_mult  # We only worry about x dir for next two masks

        # If the x value of the point is near the origin and the normal is pointing in the +x direction, flip
        mask_x0 = (np.abs(np_points[:, 0]) < 0.1) & (temp_normal > 0.8)

        # If the x value of the point is near the part's end and the normal is pointing in the -x direction, flip
        mask_xL = (np.abs(np_points[:, 0] - self.new_length) < 0.1) & (temp_normal < -0.8)
        # Flip from mask centers and flip from either maskx0 or maskxl
        final_flip_mask = mask_centers_flip ^ (mask_x0 | mask_xL)
        np_normals[final_flip_mask] *= -1

        pcd.normals = o3d.utility.Vector3dVector(np_normals)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        mesh = self.call_o3d(pcd)

        return mesh

    # @timeit
    def call_o3d(self, pcd):
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh_pts = np.asarray(mesh.vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.uniform(size=(len(mesh_pts), 3)))
        # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        return mesh

    def generate_stock_pcd(self, diameter, stock_type):
        """
        TODO: Take in stock.obj for this
        """
        r = diameter / 2  # [mm]
        stock_length = 20  # [mm]
        n_points = 10000
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = np.random.uniform(0, stock_length, n_points)
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        points = np.stack([x, y, z], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        self.target_pcd = pcd


def grid_ring_with_points(ring, points_per_unit_area=10):
    from scipy.spatial import Delaunay

    tri = Delaunay(ring)

    filled_points = []
    for triangle in tri.simplices:
        pts = ring[triangle]
        pts_in_triangle = sample_points_in_triangle(pts, points_per_unit_area)
        filled_points.append(pts_in_triangle)

    return np.vstack(filled_points)


def sample_points_in_triangle(pts, points_per_unit_area=200):
    """
    Generate points uniformly inside a triangle defined by 3 vertices.

    Args:
        pts: 3x2 array of triangle vertices
        points_per_unit_area: desired point density (points per unit area)

    Returns:
        Array of points inside the triangle
    """
    # Calculate triangle area using cross product
    # Area = 0.5 * |(b-a) x (c-a)|
    a, b, c = pts
    area = 0.5 * abs(np.cross(b - a, c - a))

    # Determine number of points based on area and desired density
    num_points = max(1, int(area * points_per_unit_area))

    sampled = []
    for _ in range(num_points):
        r1, r2 = np.random.rand(2)
        # Ensure the point is inside the triangle
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        point = pts[0] + r1 * (pts[1] - pts[0]) + r2 * (pts[2] - pts[0])
        sampled.append(point)

    return np.array(sampled)
