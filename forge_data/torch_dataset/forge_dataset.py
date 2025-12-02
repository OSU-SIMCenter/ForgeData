import h5py
import torch


class ForgeDataset(torch.utils.data.Dataset):
    """
    Pytorch data primitive to easily access Agility Forge data.
    """

    def __init__(self, *args, **kwargs):
        self.h5_file = args[0]

        with h5py.File(self.h5_file, "r") as f:
            self.sample_keys = f["global_keyset"].asstr()[:]

    def __len__(self):
        return len(self.sample_keys)

    def print_h5_structure(self):
        with h5py.File(self.h5_file, "r") as f:

            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")

            f.visititems(print_attrs)
