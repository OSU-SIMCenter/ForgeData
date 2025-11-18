import numpy as np
import open3d as o3d
import scipy.optimize
from scipy.spatial.transform import Rotation

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class Recon:
    def __init__(self, args):
        """
        TODO
        """

        self.args = args
        self.default_axis_angle_vector = np.array([0.0, 0.0, -0.0, -1.06619719, -0.00775896, 0.04988556, 301.72860016])
        self.n_error_calls = 0

        self.default_optimization_bounds = (
            (0, 100),
            (0, 100),
            (0, 100),
            (-1, 1),
            (-0.1, 0.1),
            (-0.1, 0.1),
            (200, 380),
        )

        # Multiscan variables, default init is for single scan
        self.i = 0
        self.j = 0
        self.current_scan_offset = 0
        self.current_error_metric_function = self.error_metric

    def error_metric(self, reconstructed_pcd):
        # Use the stored error mesh
        mesh = self.error_mesh
        self.n_error_calls += 1

        # Create a boolean mask for points where x is within the smallest 10% of the range.
        x_max = np.max(reconstructed_pcd[:, 0], axis=0)
        x_min = np.min(reconstructed_pcd[:, 0], axis=0)
        x_range = x_max - x_min
        mask = reconstructed_pcd[:, 0] <= (x_min + x_range * 0.40)
        masked_reconstructed_pcd = reconstructed_pcd[mask]

        # Setup raycasting scene and get the closest points on the stock mesh.
        pcd_o3d = o3d.core.Tensor(masked_reconstructed_pcd, dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        # Convert legacy mesh to the tensor-based geometry.
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        closest_points = scene.compute_closest_points(pcd_o3d)["points"].numpy()

        # Compute RMSE as the error metric.
        distances = masked_reconstructed_pcd - closest_points
        rmse = np.sqrt(((distances) ** 2).mean())

        return rmse

    def reconstruct_pcd_axis_angle(self, V, *args, **kwargs):
        self.j += 1

        rotation_axis = V[3:6]
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        radial_offset = V[6]
        (scanner_data, return_pcd) = args
        (_, x, z, th) = scanner_data
        n_rotations = th.shape[0]

        reconstructed_points = []

        z = z - radial_offset
        for i in range(n_rotations):
            rotation_vec = Rotation.from_rotvec(th[-i] * rotation_axis)
            scan = np.stack((x[i], np.zeros_like(x[i]), z[i]), axis=1)
            mask = ~np.isnan(scan[:, 0]) & ~np.isnan(scan[:, 2])
            scan = scan[mask]
            transformed_points = rotation_vec.apply(scan)
            reconstructed_points.append(transformed_points)

        reconstructed_pcd = np.vstack(reconstructed_points)
        reconstructed_pcd[:, 0] = reconstructed_pcd[:, 0] + self.current_scan_offset

        if return_pcd:
            return reconstructed_pcd
        rmse = self.current_error_metric_function(reconstructed_pcd)
        return rmse

    def reconstruct_pcd_error_compensated(self, scanner_data):

        args = (scanner_data, False)
        if self.args.error_comp:
            res = scipy.optimize.minimize(
                self.reconstruct_pcd_axis_angle, self.default_axis_angle_vector, args=args, method="SLSQP", tol=1e-6
            )
            xf = res.x
        else:
            xf = self.default_axis_angle_vector

        reconstructed_pcd = self.reconstruct_pcd_axis_angle(xf, *(scanner_data, True))
        return reconstructed_pcd, xf

    def preprocess(self, df_list):
        """
        TODO
        """

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

    def process(self):
        """
        TODO
        """

        # scan_stepover = 40  # mm stepped through for each scan

        scanner_data = self.scanner_data_list[0]
        pcd, _ = self.reconstruct_pcd_error_compensated(scanner_data)
        pcds = []
        no_overlap_pcds = []
        pcds.append(pcd)
        no_overlap_pcds.append(pcd)

        # for i, scan_data in enumerate(self.scanner_data_list[1:], start=1):
        #     self.i = i
        #     self.current_scan_offset = scan_stepover * i

        #     # Create variables needed by error metric for RMSE
        #     self.truth_pcd = pcds[i - 1]
        #     self.current_overlap_start = np.nanmax(self.truth_pcd[:, 0]) - scan_stepover  # [mm]
        #     self.current_overlap_end = self.current_overlap_start + scan_stepover

        #     pcd, _ = self.multiscan_stitching_error_compensation(scan_data)

        #     # Remove overlap from pointclouds for final pcd. Of course we still need
        #     # that overlap for the error compensation.
        #     mask = pcd[:, 0] >= self.current_overlap_end
        #     pcd_no_overlap = pcd[mask]
        #     no_overlap_pcds.append(pcd_no_overlap)

        #     pcds.append(pcd)

        self.finished_recon_pcd = np.vstack(no_overlap_pcds)

        return self.finished_recon_pcd

    def post_process(self):
        pcd = self.finished_recon_pcd
        part_length = np.nanmax(pcd[:, 0])
        part_min = np.nanmin(pcd[:, 0])

        # --- Clip relative percentages off the front and back ---
        clip_front_fraction = 0.03
        clip_end_fraction = 0.05
        clip_front = clip_front_fraction * part_length
        clip_end = clip_end_fraction * part_length
        mask = (pcd[:, 0] >= part_min + clip_front) & (pcd[:, 0] <= part_length - clip_end)
        pcd = pcd[mask]

        # --- Statistical outlier removal ---
        pcdo = o3d.geometry.PointCloud()
        pcdo.points = o3d.utility.Vector3dVector(pcd)
        pcdo, _ = pcdo.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = np.asarray(pcdo.points)

        # --- Fill front (x ~ clipped start) ---
        slice_thickness_front = 2.0  # mm, enough points for triangulation
        mask = (pcd[:, 0] >= part_min + clip_front) & (pcd[:, 0] <= part_min + clip_front + slice_thickness_front)
        startpoints = pcd[mask]
        startpoints_yz = startpoints[:, [1, 2]]
        origin_filled_points = grid_ring_with_points(startpoints_yz)
        x = (part_min + clip_front + np.zeros_like(origin_filled_points[:, 0])).reshape(-1, 1)
        origin_filled_points_3d = np.hstack((x, origin_filled_points))

        # --- Fill end (x ~ clipped end) ---
        new_length = np.nanmax(pcd[:, 0])
        self.new_length = new_length
        slice_thickness_end = 1.0  # mm
        mask = pcd[:, 0] >= new_length - slice_thickness_end
        endpoints = pcd[mask]
        endpoints_yz = endpoints[:, [1, 2]]
        filled_points = grid_ring_with_points(endpoints_yz)
        x = (new_length + np.zeros_like(filled_points[:, 0])).reshape(-1, 1)
        filled_points_3d = np.hstack((x, filled_points))

        # --- Combine original points with filled front and back ---
        pcd = np.vstack([pcd, origin_filled_points_3d, filled_points_3d])

        return pcd

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

        max_x = np.max(np_points[:, 0])
        min_x = np.min(np_points[:, 0])
        n_center_points = 10
        centers = np.linspace(min_x, max_x, n_center_points)
        centers = np.stack([centers, np.zeros(n_center_points), np.zeros(n_center_points)], axis=-1)
        centers = np.stack([np_points[:, 0], np.zeros(len(np_points)), np.zeros(len(np_points))], axis=1)

        for i in range(len(np_points)):
            # If pointing towards the x-axis, flip the normal
            points_to_center = centers[i] - np_points[i]
            dot_products = np.dot(points_to_center, np_normals[i])
            if np.any(dot_products > 0):
                np_normals[i] = -np_normals[i]

            # If near x=0 and pointing +x, flip
            point = np_points[i]
            normal = np_normals[i]
            condition1 = np.isclose(point[0], 0, rtol=0, atol=0.1)  # Absolute isnear
            condition2 = np.dot(normal, np.array([1, 0, 0])) > 0.9
            if condition1 and condition2:
                np_normals[i] = -np_normals[i]

            # If near x=part_length and pointing -x, flip
            condition1 = np.isclose(point[0], self.new_length, rtol=0, atol=0.1)  # Absolute isnear
            condition2 = np.dot(normal, np.array([-1, 0, 0])) > 0.9
            if condition1 and condition2:
                np_normals[i] = -np_normals[i]

        pcd.normals = o3d.utility.Vector3dVector(np_normals)

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh_pts = np.asarray(mesh.vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.uniform(size=(len(mesh_pts), 3)))
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        return mesh


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
