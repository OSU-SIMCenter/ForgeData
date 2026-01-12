from pathlib import Path
from typing import NamedTuple

import cv2
import h5py
import matplotlib

matplotlib.use("TkAgg")  # OpenCV busted default, this fixes error in plot_load_stroke on Ubuntu
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch


class MeshData(NamedTuple):
    vertices: torch.tensor
    faces: torch.tensor


class ForgeSample(NamedTuple):
    """
    TODO Docstring
    """

    x: MeshData  # Current State (Mesh) [mm]
    a: torch.Tensor  # Action, (x-axis [mm], theta [deg], radius [mm])
    y: MeshData  # Next State (Mesh) [mm]
    t: torch.Tensor  # time for load/stroke data [s]
    load: torch.Tensor  # Press load [kN]
    stroke: torch.Tensor  # Press position [mm]
    T_max: torch.Tensor  # Max temperature of workpiece during action [degC]
    T_avg: torch.Tensor  # Mean temperature of workpiece during action [degC]
    T_frame: torch.Tensor  # Thermal image during action [degC]
    T_field: torch.Tensor  # Temperature values of each node in mesh x [degC] n_nodes x 1
    t_thermal: torch.Tensor # time for thermal video feed
    T_0_t: torch.Tensor  # Average temperature in pixel set 0
    T_1_t: torch.Tensor  # Average temperature in pixel set 1
    T_2_t: torch.Tensor  # Average temperature in pixel set 2
    T_3_t: torch.Tensor  # Average temperature in pixel set 3
    T_4_t: torch.Tensor
    T_5_t: torch.Tensor
    path: str  # Metadata (Debug path)


class ForgeDataset(torch.utils.data.Dataset):
    """
    Pytorch data primitive to easily access Agility Forge data.

    TODO:
        Add blacklist
        Thermal frames, average/max workpiece temperature
        Could rearrange the h5 and have less processing per data point in this class, but it won't matter.
        Probably device should be passed into dataset
    """

    def __init__(self, *args, **kwargs):
        self.h5_path = args[0]
        self.h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            self.global_keyset = f["global_keyset"].asstr()[:]
            # Pull scan 0 out, and x offset, so we can map LS data to stock action

    def __len__(self):
        return len(self.global_keyset)

    def __getitem__(self, idx):
        """
        TODO
        """
        f = self._get_h5_file()

        curr_path = self.global_keyset[idx]
        curr_group = f[curr_path]

        ls_data = curr_group["load_stroke"][:]

        path_ = Path(curr_path)
        folder_name = path_.name
        parent_path_ = path_.parent

        try:
            xv = None
            xf = None
            if folder_name == "H0000":
                # Pull the stock .obj file instead of the previous scan, there is no stock scan
                parent_group = f[parent_path_.as_posix()]
                obj_key = None
                for key in parent_group:
                    if key.endswith(".obj"):
                        obj_key = key
                        break

                if obj_key:
                    xv = torch.from_numpy(parent_group[obj_key]["vertices"][:]).float()
                    xf = torch.from_numpy(parent_group[obj_key]["faces"][:]).long()
            else:
                curr_hit_num = int(folder_name[1:])
                prev_hit_name = f"H{curr_hit_num - 1:04d}"
                prev_path = f"{parent_path_.as_posix()}/{prev_hit_name}"
                prev_group = f[prev_path]
                xv = torch.from_numpy(prev_group["reconstructed_mesh/vertices"][:]).float()
                xf = torch.from_numpy(prev_group["reconstructed_mesh/faces"][:]).long()
            if xv is None or xf is None:
                x = None
            x = MeshData(vertices=xv, faces=xf)
        except Exception:
            x = None

        a_x = ls_data["X pos referenced to target part butt"][0]
        a_theta = ls_data["A"][0]
        a_depth = ls_data["Press Target Position (mm)"][0] / 2  # TODO: This is actually not right at the moment
        action = np.array([a_x, a_theta, a_depth], dtype=np.float64)
        action = torch.from_numpy(action)

        try:
            yv = torch.from_numpy(np.array(curr_group["reconstructed_mesh/vertices"], dtype=np.float64))
            yf = torch.from_numpy(np.array(curr_group["reconstructed_mesh/faces"], dtype=np.int32))
            y = MeshData(vertices=yv, faces=yf)
        except Exception:
            # Missing or malformed reconstructed_mesh
            y = None

        time = torch.from_numpy(ls_data["Time Unix (ms)"] / 1000)
        load = torch.from_numpy(ls_data["Force (kN)"])
        stroke = torch.from_numpy(ls_data["Position (mm)"])

        # Get the max temperature of workpiece while the load is max
        idx_max_load = np.argmax(ls_data["Force (kN)"])
        time_max_load = ls_data["Time Unix (ms)"][idx_max_load] / 1000

        try:
            thermal_group = f[str(path_.parent.parent) + "/t"]
            thermal_timeline = np.array(thermal_group["time"])
            idx_closest = np.argmin(np.abs(thermal_timeline - time_max_load))
            thermal_frame = np.array(thermal_group["frames"][idx_closest])
            temperature = thermal_frame.astype(np.float32) / 10.0 - 100.0
            T_max = torch.tensor(np.max(temperature))
            mask = temperature > 700
            T_avg = torch.tensor(np.mean(temperature[mask]))
            T_frame = torch.from_numpy(temperature)
            thermal_timeline = torch.from_numpy(thermal_timeline)
            t_thermal = thermal_timeline

            # Sorry for hardcoding, TODO fix later
            x0_slice = np.s_[:, 110:150, 35:40]
            x1_slice = np.s_[:, 80:90, 53:70]
            x2_slice = np.s_[:, 110:150, 53:70]
            x3_slice = np.s_[:, 170:180, 53:70]
            x4_slice = np.s_[:, 200:210, 53:70]
            x5_slice = np.s_[:, 230:240, 53:70]
            raw_data_0 = thermal_group["frames"][x0_slice]
            raw_data_1 = thermal_group["frames"][x1_slice]
            raw_data_2 = thermal_group["frames"][x2_slice]
            raw_data_3 = thermal_group["frames"][x3_slice]
            raw_data_4 = thermal_group["frames"][x4_slice]
            raw_data_5 = thermal_group["frames"][x5_slice]
            T_0_history = raw_data_0.astype(np.float32) / 10.0 - 100.0
            T_1_history = raw_data_1.astype(np.float32) / 10.0 - 100.0
            T_2_history = raw_data_2.astype(np.float32) / 10.0 - 100.0
            T_3_history = raw_data_3.astype(np.float32) / 10.0 - 100.0
            T_4_history = raw_data_4.astype(np.float32) / 10.0 - 100.0
            T_5_history = raw_data_5.astype(np.float32) / 10.0 - 100.0
            T_0_t = torch.from_numpy(np.mean(T_0_history, axis=(1, 2)))
            T_1_t = torch.from_numpy(np.mean(T_1_history, axis=(1, 2)))
            T_2_t = torch.from_numpy(np.mean(T_2_history, axis=(1, 2)))
            T_3_t = torch.from_numpy(np.mean(T_3_history, axis=(1, 2)))
            T_4_t = torch.from_numpy(np.mean(T_4_history, axis=(1, 2)))
            T_5_t = torch.from_numpy(np.mean(T_5_history, axis=(1, 2)))
        except Exception as e:
            print(e)
            # No thermal frames available for this datapoint
            T_max = None
            T_avg = None
            T_frame = None
            t_thermal = None
            T_0_t = None
            T_1_t = None
            T_2_t = None
            T_3_t = None
            T_4_t = None
            T_5_t = None

        # If a good x mesh is available, and a thermal frame is available, output a temperature field over the nodes of
        # x. This temperature field will assume uniform temperatures in the x direction.
        T_field = self._get_temperature_field(x, T_frame, a_x) if (T_frame is not None) and (x is not None) and (x.vertices is not None) else None

        # NOTE I think for batched dataloader you need ForgeData mesh tensors to always have the same size, or use some
        # collate function
        return ForgeSample(
            x=x,
            a=action,
            y=y,
            t=time,
            load=load,
            stroke=stroke,
            T_max=T_max,
            T_avg=T_avg,
            T_frame=T_frame,
            T_field=T_field,
            t_thermal=t_thermal,
            T_0_t=T_0_t,
            T_1_t=T_1_t,
            T_2_t=T_2_t,
            T_3_t=T_3_t,
            T_4_t=T_4_t,
            T_5_t=T_5_t,
            path=curr_path,
        )

    def print_h5_structure(self):
        f = self._get_h5_file()

        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        f.visititems(print_attrs)

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def plot_state_action(self, idx, return_image=False, window_size=(512, 512)):
        """
        Generate plots of the pre- and post-hit meshes and action vectors.

        If `return_image=True` this returns a tuple `(img_x, img_y)` where each is an RGB
        uint8 numpy array with shape (H,W,3). Otherwise the function shows an interactive
        plot (previous behavior) but without mesh edges.
        """

        def make_pv_mesh(verts, faces):
            faces = faces.astype(np.int32)
            if faces.ndim == 2 and faces.shape[1] == 3:
                padding = np.full((faces.shape[0], 1), 3)
                faces = np.hstack([padding, faces]).flatten()
            return pv.PolyData(verts, faces)

        datapoint = self[idx]

        # Build meshes if available
        mesh_x = None
        mesh_y = None
        if datapoint.x is not None:
            xv = datapoint.x.vertices.cpu().numpy()
            xf = datapoint.x.faces.cpu().numpy()
            mesh_x = make_pv_mesh(xv, xf)
        if datapoint.y is not None:
            yv = datapoint.y.vertices.cpu().numpy()
            yf = datapoint.y.faces.cpu().numpy()
            mesh_y = make_pv_mesh(yv, yf)

        action = datapoint.a.cpu().numpy()
        x_pos = action[0] + 57.32  # TODO: TALK TO BRIAN THIS NUMBER IS WRONG
        theta = action[1] + 105  # TODO: TALK TO BRIAN THIS NUMBER IS WRONG
        action[2]

        # print(f"action tuple: {(x_pos, theta, hit_radius)}")

        theta_rad = np.deg2rad(theta)
        r = 20
        yy = r * np.cos(theta_rad)
        z = r * np.sin(theta_rad)

        start_point = np.array([x_pos, yy, z])
        direction = np.array([x_pos, 0, 0]) - start_point

        arrow = pv.Arrow(
            start=start_point,
            direction=direction,
            scale=10.0,  # Length of arrow
            shaft_radius=0.01,
            tip_radius=0.05,
            tip_length=0.25,
        )
        a2s = np.array([start_point[0], -start_point[1], -start_point[2]])
        a2d = np.array([x_pos, 0, 0]) - a2s
        arrow2 = pv.Arrow(
            start=a2s,
            direction=a2d,
            scale=10.0,  # Length of arrow
            shaft_radius=0.01,
            tip_radius=0.05,
            tip_length=0.25,
        )

        if return_image:
            # Render each mesh separately (no edges) and return RGB images
            imgs = []
            for mesh, color, opacity in [(mesh_x, "lightblue", 1.0), (mesh_y, "grey", 1.0)]:
                if mesh is None:
                    # create blank image
                    h, w = window_size[1], window_size[0]
                    imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
                    continue
                pl = pv.Plotter(off_screen=True, window_size=window_size)
                pl.add_mesh(mesh, color=color, opacity=opacity, show_edges=False)
                pl.add_mesh(arrow, color="red")
                pl.add_mesh(arrow2, color="red")
                pl.show_grid(grid="back", location="outer", color="lightgrey", font_size=10)
                pl.add_axes()
                pl.camera_position = "iso"
                pl.set_background("white")
                img = pl.screenshot(return_img=True)
                pl.close()
                # Ensure RGB 3 channels
                if img is None:
                    imgs.append(np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8))
                else:
                    if img.shape[2] == 4:
                        img = img[..., :3]
                    imgs.append(img.astype(np.uint8))
            return tuple(imgs)
        else:
            # Interactive combined view (keeps old behavior minus edges)
            pl = pv.Plotter()
            if mesh_x is not None:
                pl.add_mesh(mesh_x, color="red", opacity=0.3, show_edges=False, label="State X (Pre-hit)")
            if mesh_y is not None:
                pl.add_mesh(mesh_y, color="lightblue", show_edges=False, label="State Y (Post-hit)")
            pl.add_mesh(arrow, color="red", label="Action Vector")
            pl.add_mesh(arrow2, color="red")
            pl.show_grid(grid="back", location="outer", color="lightgrey", font_size=10)
            pl.add_axes()
            pl.camera_position = "iso"
            pl.set_background("white")
            pl.show(title=f"Forge Sample {idx}")

    def plot_load_stroke(self, idx, return_image=False, figsize=(6, 4)):
        datapoint = self[idx]

        t = datapoint.t.cpu().numpy()
        t = t - np.min(t)
        load = datapoint.load.cpu().numpy()
        stroke = datapoint.stroke.cpu().numpy()

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Time [s]", fontsize=12)
        ax1.set_ylabel("Load [kN]", fontsize=12)
        line1 = ax1.plot(t, load, color="tab:blue", linewidth=2, label="Load")
        ax1.tick_params(axis="y")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Stroke [mm]", fontsize=12)
        line2 = ax2.plot(t, stroke, color="tab:red", linewidth=2, linestyle="--", label="Stroke")
        ax2.tick_params(axis="y")
        lines = line1 + line2
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="upper right")
        plt.title("Load & Stroke vs. Time", fontsize=14)
        plt.tight_layout()

        if return_image:
            # Render figure to RGB numpy array
            fig.canvas.draw()
            rgba_buffer = fig.canvas.buffer_rgba()
            img = np.asarray(rgba_buffer)[:, :, :3]
            plt.close(fig)
            return img
        else:
            plt.show()

    def plot_thermal_frame(self, idx, return_image=False):
        datapoint = self[idx]
        if datapoint.T_frame is None:
            if return_image:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                print("No thermal frame available for this datapoint")
                return

        T_frame = datapoint.T_frame.cpu().numpy()
        vis = cv2.normalize(T_frame, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)
        colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        # Convert BGR->RGB for consistency
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

        if return_image:
            return colorized_rgb
        else:
            cv2.imshow("Thermal Video (press q/esc to quit)", colorized)
            cv2.waitKey(0)

    def plot_thermal_frame_callback(self, idx, return_image=False):
        datapoint = self[idx]
        if datapoint.T_frame is None:
            return np.zeros((256, 256, 3), dtype=np.uint8) if return_image else print("No thermal frame available for this datapoint")

        T_raw = datapoint.T_frame.cpu().numpy()
        vis = cv2.normalize(T_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        if return_image:
            return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

        def show_temp(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                img_copy = colorized.copy()
                temp_val = T_raw[y, x] # Note: numpy uses [row, col] -> [y, x]
                text = f"{temp_val:.2f} C"
                cv2.putText(img_copy, text, (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Thermal Video", img_copy)

        cv2.namedWindow("Thermal Video")
        cv2.setMouseCallback("Thermal Video", show_temp)

        cv2.imshow("Thermal Video", colorized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_temperature_field(self, idx, return_image=False, window_size=(512, 1024), clim=None):
        """
        Render mesh `x` colored by `T_field` (per-vertex temperature) and return an RGB image when
        `return_image=True`. `window_size` is (width, height) to allow tall renders for RHS stitching.
        """
        datapoint = self[idx]

        if datapoint.T_field is None or datapoint.x is None:
            if return_image:
                return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
            else:
                print("No temperature field available for this datapoint")
                return

        T_field = datapoint.T_field.cpu().numpy()
        verts = datapoint.x.vertices.cpu().numpy()
        faces = datapoint.x.faces.cpu().numpy()

        # Convert faces to PyVista format
        faces_arr = faces.astype(np.int32)
        if faces_arr.ndim == 2 and faces_arr.shape[1] == 3:
            padding = np.full((faces_arr.shape[0], 1), 3)
            faces_arr = np.hstack([padding, faces_arr]).flatten()

        pv_mesh = pv.PolyData(verts, faces_arr)
        pv_mesh.point_data["Temperature [degC]"] = T_field

        if clim is None:
            valid = T_field[~np.isnan(T_field)] if np.any(~np.isnan(T_field)) else T_field
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin, vmax = clim

        if return_image:
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            pl.add_mesh(pv_mesh, scalars="Temperature [degC]", cmap="jet", clim=[vmin, vmax], show_edges=False)
            pl.add_scalar_bar(title="Temperature (°C)")
            pl.show_grid(grid="back", location="outer", color="lightgrey", font_size=10)
            pl.add_axes()
            pl.set_background("white")
            pl.camera_position = "iso"
            img = pl.screenshot(return_img=True)
            pl.close()
            if img is None:
                return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
            if img.shape[2] == 4:
                img = img[..., :3]
            return img.astype(np.uint8)
        else:
            plotter = pv.Plotter()
            plotter.add_mesh(pv_mesh, scalars="Temperature [degC]", cmap="jet", clim=[vmin, vmax], show_edges=False)
            plotter.add_scalar_bar(title="Temperature (°C)")
            plotter.show_grid(grid="back", location="outer", color="lightgrey", font_size=10)
            plotter.add_axes()
            plotter.set_background("white")
            plotter.camera_position = "iso"
            plotter.show()

    def save_thermal_video(self, path):
        fps = 8
        first_frame = self[0].T_frame.numpy()
        h, w = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        for i in range(len(self)):
            datapoint = self[i]
            T_frame = datapoint.T_frame.numpy()
            vis = cv2.normalize(T_frame, None, 0, 255, cv2.NORM_MINMAX)
            vis = vis.astype(np.uint8)
            colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            out.write(colorized)
            if i % (len(self) // 10) == 0:
                print(f"Processed frame {i}/{len(self)}")
        out.release()

    def _get_temperature_field(self, x, thermal_frame, a_x):
        """Maps thermal frame data to the mesh nodes, assuming uniform x-axis temperature.

        Args:
            x: The current workpiece mesh data containing vertices and faces.
            thermal_frame: A 2D tensor representing the thermal image [degC].

        Returns:
            A tensor representing the temperature field mapped over the nodes of x.
        """
        vis_dbg = False

        # Calibrate camera by hand, it's rigidly mounted
        pixels_per_mm = (207 - 16) / 50.8
        mm_per_pixel = 1 / pixels_per_mm

        # Filter out cold background temperatures
        threshold_temp = 800.0
        thermal_workpiece = torch.where(thermal_frame > threshold_temp, thermal_frame, torch.tensor(float("nan")))
        vis_np = thermal_workpiece.detach().cpu().numpy()
        T_profile_1d = np.nanmean(vis_np, axis=1)
        T_profile_1d[np.isnan(T_profile_1d)] = 800.0

        if vis_dbg:
            mask = ~np.isnan(vis_np)
            vis_display = np.zeros_like(vis_np, dtype=np.uint8)
            if np.any(mask):
                valid_min = vis_np[mask].min()
                valid_max = vis_np[mask].max()
                vis_display[mask] = ((vis_np[mask] - valid_min) / (valid_max - valid_min + 1e-6) * 255).astype(np.uint8)
            colorized = cv2.applyColorMap(vis_display, cv2.COLORMAP_JET)
            colorized[~mask] = 0
            cv2.imshow("Thermal Workpiece (NaN is Black)", colorized)
            cv2.waitKey(0)

            print(f"T_profile_1d.shape: {T_profile_1d.shape}")
            plt.figure()
            plt.plot(T_profile_1d)
            plt.show()

        # We know that x value on the mesh a_x is located at pixel y=128 in the thermal frame
        reference_y = 128
        num_pixels = T_profile_1d.shape[0]
        pixel_indices = np.arange(num_pixels)
        pixel_coords_mm = (pixel_indices - reference_y) * mm_per_pixel + a_x + 57.32
        T_profile_1d = np.flip(T_profile_1d, axis=0)

        mesh_points = x.vertices.cpu().numpy()
        vertex_temps = np.full(len(mesh_points), 800.0)  # Default/Ambient temp
        vertex_temps = np.interp(
            mesh_points[:, 0],
            pixel_coords_mm,
            T_profile_1d,
            # left=700.0,
            # right=700.0
        )

        if vis_dbg:
            faces = x.faces.cpu().numpy()
            pv_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
            point_cloud = x.vertices.cpu().numpy()
            pv_mesh = pv.PolyData(point_cloud, pv_faces)
            pv_mesh.point_data["Temperature [degC]"] = vertex_temps
            plotter = pv.Plotter()
            plotter.add_mesh(
                pv_mesh,
                scalars="Temperature [degC]",
                cmap="jet",
                clim=[800, 1050],  # Based on your previous snippet
                show_edges=False,
            )
            plotter.add_scalar_bar(title="Temperature (°C)")
            plotter.show_grid(grid="back", location="outer", color="lightgrey", font_size=10)
            plotter.add_axes()
            plotter.set_background("white")
            plotter.show()

        return torch.from_numpy(vertex_temps)

    def __del__(self):
        """
        Cleanup hdf5, invoked when ForgeDataset instance is garbage collected
        """
        if self.h5_file is not None:
            self.h5_file.close()
