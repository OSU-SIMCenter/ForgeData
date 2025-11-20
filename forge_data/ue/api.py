import numpy as np
import pandas as pd

from forge_data.ue.core import Recon


def mesh_from_dataframe(df: pd.DataFrame, args=None) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    recon = Recon(DefaultArgs())
    recon.preprocess(df)
    recon.process()
    pcd = recon.post_process()
    mesh = recon.pcd_to_mesh(pcd)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    return vertices, faces


class DefaultArgs:
    def __init__(self):
        self.visualize = False
        self.verbosity = "error"
        self.error_comp = False
