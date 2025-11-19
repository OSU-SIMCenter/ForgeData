import re

import h5py
import torch


class ForgeDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.h5_file = args[0]
        # pattern = re.compile(r"^sample_\d{4}-\d{2}-\d{2}_[^_]+_\d{4}$")
        # self.save_meshes = kwargs.get('save_meshes', False)
        # self.save_path_base = kwargs.get('save_path', './data/inspection')
        # with h5py.File(self.h5_file, 'r') as f:
        #     self.sample_keys = [k for k in f if pattern.match(k)]

    def __len__(self):
        return len(self.sample_keys)

    def print_h5_structure(self):
        with h5py.File(self.h5_file, 'r') as f:
            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")

            f.visititems(print_attrs)
