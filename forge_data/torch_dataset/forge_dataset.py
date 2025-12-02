import h5py
import numpy as np
import torch


class ForgeDataset(torch.utils.data.Dataset):
    """
    Pytorch data primitive to easily access Agility Forge data.

    TODO:
        Add blacklist
        Thermal frames, average/max workpiece temperature
        Could rearrange the h5 and have less processing per data point in this class, but it won't matter.
    """

    def __init__(self, *args, **kwargs):
        self.h5_path = args[0]
        self.h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            self.global_keyset = f["global_keyset"].asstr()[:]
            # Pull scan 0 out, and x offset, so we can map LS data to stock action

    def __len__(self):
        return len(self.global_keyset)

    def  __getitem__(self, idx):
        """
        TODO
        """
        f = self._get_h5_file()

        curr_path = self.global_keyset[idx]
        curr_group = f[curr_path]

        ls_data = curr_group['load_stroke'][:]
        # print(ls_data)
        action = self._get_action_from_load_stroke(ls_data)

        pass

    def print_h5_structure(self):
        with h5py.File(self.h5_path, "r") as f:

            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")

            f.visititems(print_attrs)

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def _get_action_from_load_stroke(ls):
        """
        Pull action out of load stroke data
        TODO
        """
        return np.array(.1, 90, 0.01)

    def plot_state_action_(self, idx):
        """
        TODO
            Generate a plot of this state-action, show hit vector on resulting mesh.
        """
        # datapoint = self[idx]
        pass
