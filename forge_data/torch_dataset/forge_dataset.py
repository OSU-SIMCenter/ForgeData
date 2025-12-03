from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
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
            parent_group = f[parent_path_]
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
        # datapoint = self[idx]
        pass

    def plot_thermal_frame(self, idx):
        pass

    def __del__(self):
        """
        Cleanup hdf5, invoked when ForgeDataset instance is garbage collected
        """
        if self.h5_file is not None:
            self.h5_file.close()
