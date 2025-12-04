from pathlib import Path
from typing import NamedTuple

import cv2
import h5py
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
    a: torch.Tensor  # Action (Hitpoint), (x-axis [mm], theta [deg], depth [mm] (this is radius of hitpoint))
    y: MeshData  # Next State (Mesh) [mm]
    load: torch.Tensor  # Press load [kN]
    stroke: torch.Tensor  # Press position [mm]
    T_max: torch.Tensor  # Max temperature of workpiece during action [degC]
    T_avg: torch.Tensor  # Mean temperature of workpiece during action [degC]
    T_frame: torch.Tensor  # Thermal image [C]
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
        x = MeshData(vertices=xv, faces=xf)

        a_x = ls_data["X pos referenced to target part butt"][0]
        a_theta = ls_data["A"][0]
        a_depth = ls_data["Press Target Position (mm)"][0] / 2  # TODO: This is actually not right at the moment
        action = np.array([a_x, a_theta, a_depth], dtype=np.float64)
        action = torch.from_numpy(action)

        yv = torch.from_numpy(np.array(curr_group["reconstructed_mesh/vertices"], dtype=np.float64))
        yf = torch.from_numpy(np.array(curr_group["reconstructed_mesh/faces"], dtype=np.int32))
        y = MeshData(vertices=yv, faces=yf)

        load = torch.from_numpy(ls_data["Force (kN)"])
        stroke = torch.from_numpy(ls_data["Position (mm)"])

        # Get the max temperature of workpiece while the load is max
        idx_max_load = np.argmax(ls_data["Force (kN)"])
        time_max_load = ls_data["Time Unix (ms)"][idx_max_load] / 1000

        thermal_group = f[str(path_.parent.parent) + "/t"]
        thermal_timeline = np.array(thermal_group["time"])
        idx_closest = np.argmin(np.abs(thermal_timeline - time_max_load))
        thermal_frame = np.array(thermal_group["frames"][idx_closest])
        temperature = thermal_frame.astype(np.float32) / 10.0 - 100.0
        T_max = torch.tensor(np.max(temperature))
        mask = temperature > 700
        T_avg = torch.tensor(np.mean(temperature[mask]))
        T_frame = torch.from_numpy(temperature)

        # NOTE I think for batched dataloader you need ForgeData mesh tensors to always have the same size, or use some
        # collate function
        return ForgeSample(
            x=x,
            a=action,
            y=y,
            load=load,
            stroke=stroke,
            T_max=T_max,
            T_avg=T_avg,
            T_frame=T_frame,
            path=curr_path,
        )

    def print_h5_structure(self):
        with h5py.File(self.h5_path, "r") as f:

            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")

            f.visititems(print_attrs)

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def plot_state_action_(self, idx):
        """
        TODO
            Generate a plot of this state-action, show hit vector on resulting mesh.
        """
        def make_pv_mesh(verts, faces):
                    """
                    PyVista (VTK) expects faces to be a flat array where each face 
                    is prefixed by the number of points (e.g., [3, v1, v2, v3, ...])
                    """
                    faces = faces.astype(np.int32)
                    # If faces are (N, 3), pad them with the number 3
                    if faces.ndim == 2 and faces.shape[1] == 3:
                        padding = np.full((faces.shape[0], 1), 3)
                        faces = np.hstack([padding, faces]).flatten()
                    return pv.PolyData(verts, faces)
        datapoint = self[idx]

        xv = datapoint.x.vertices.cpu().numpy()
        xf = datapoint.x.faces.cpu().numpy()
        mesh_x = make_pv_mesh(xv, xf)

        yv = datapoint.y.vertices.cpu().numpy()
        yf = datapoint.y.faces.cpu().numpy()
        mesh_y = make_pv_mesh(yv, yf)

        action = datapoint.a.cpu().numpy()
        x_pos = action[0] + 57.32  # TODO: TALK TO BRIAN THIS NUMBER IS WRONG
        theta = action[1] + 105 # TODO: TALK TO BRIAN THIS NUMBER IS WRONG
        hit_radius = action[2]

        print(f"action tuple: {(x_pos, theta, hit_radius)}")

        theta_rad = np.deg2rad(theta)
        r = 20
        y = r * np.cos(theta_rad)
        z = r * np.sin(theta_rad)

        start_point = np.array([x_pos, y, z])
        direction = np.array([x_pos, 0, 0]) - start_point

        # shaft_radius and tip_radius allow us to visualize the 'hit_radius'
        arrow = pv.Arrow(
            start=start_point,
            direction=direction,
            scale=10.0, # Length of arrow
            shaft_radius=.01,
            tip_radius=.05,
            tip_length=0.25
        )
        a2s = np.array([start_point[0], -start_point[1], -start_point[2]])
        a2d = np.array([x_pos, 0, 0]) - a2s
        arrow2 = pv.Arrow(
            start=a2s,
            direction=a2d,
            scale=10.0, # Length of arrow
            shaft_radius=.01,
            tip_radius=.05,
            tip_length=0.25
        )

        pl = pv.Plotter()
        pl.add_mesh(mesh_x, color="red", opacity=0.3, label="State X (Pre-hit)")
        pl.add_mesh(mesh_y, color="lightblue", show_edges=True, label="State Y (Post-hit)")
        pl.add_mesh(arrow, color="red", label="Action Vector")
        pl.add_mesh(arrow2, color="red")
        pl.show_grid(
            grid='back',
            location='outer',
            color='lightgrey',
            font_size=10
        )
        pl.add_axes()
        # stats = (
        #     f"Step: {idx}\n"
        #     f"Load: {to_numpy(self.load):.2f} kN\n"
        #     f"Stroke: {to_numpy(self.stroke):.2f} mm\n"
        #     f"Max Temp: {to_numpy(self.T_max):.1f} C"
        # )
        # pl.add_text(stats, position='upper_left', font_size=10, color='black')
        pl.set_background('white')
        pl.show(title=f"Forge Sample {idx}")


    def plot_thermal_frame(self, idx):
        datapoint = self[idx]
        T_frame = datapoint.T_frame.numpy()
        vis = cv2.normalize(T_frame, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)
        colorized = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imshow("Thermal Video (press q/esc to quit)", colorized)
        cv2.waitKey(0)

    def __del__(self):
        """
        Cleanup hdf5, invoked when ForgeDataset instance is garbage collected
        """
        if self.h5_file is not None:
            self.h5_file.close()
